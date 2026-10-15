// Qwen3.8-Flash-Next's per-request host work (FamilyRuntime): its PLE rows,
// hashed from each request's tokens and read from the GGUF shard through the
// PleStore into the staging the graphs gather from, and the QSA block records
// of each step's rows.

#include "model/FamilyRuntime.hpp"

#include "metal/abi/Qwen4.h"
#include "model/ModelFactory.hpp"
#include "model/PleStore.hpp"
#include "model/Qwen4Exp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace splash::model {
namespace {

constexpr uint64_t kDefaultCacheMiB = 256;

uint64_t cacheBytesFromEnvironment() {
  const char *value = std::getenv("SPLASH_PLE_CACHE_MB");
  if (!value || !*value) return kDefaultCacheMiB << 20;
  return std::strtoull(value, nullptr, 10) << 20;
}

bool envIsZero(const char *name) {
  const char *value = std::getenv(name);
  return value && std::strcmp(value, "0") == 0;
}

class Qwen4Family final : public FamilyRuntime {
public:
  explicit Qwen4Family(const Qwen4ExpWeights &weights)
      : hasher_(weights.pleTable.hash),
        store_(PleSource(weights.pleTable.path.string(), weights.pleTable.dataOffset, weights.pleTable.rows,
                         weights.pleTable.rowBytes),
               cacheBytesFromEnvironment()),
        heads_(hasher_.heads()), rowBytes_(weights.pleTable.rowBytes), log_(std::getenv("SPLASH_PLE_LOG") != nullptr),
        prefetchEnabled_(!envIsZero("SPLASH_PLE_PREFETCH")) {
    std::fprintf(stderr, "PLE rows read from %s; row cache %llu MiB of host RAM (outside the Metal budget)\n",
                 weights.pleTable.path.filename().c_str(),
                 static_cast<unsigned long long>(cacheBytesFromEnvironment() >> 20));
  }

  ~Qwen4Family() override {
    dropPrefetch();
    if (log_) logStats("final");
  }

  void attach(const FamilyScratch &scratch) override { scratch_ = scratch; }

  void onActivate(uint64_t requestId, uint32_t /*stateLane*/, std::span<const uint32_t> prompt) override {
    history_[requestId].assign(prompt.begin(), prompt.end());
  }

  void onEnd(uint64_t requestId) override {
    if (prefetch_ && prefetch_->requestId == requestId) dropPrefetch();
    history_.erase(requestId);
  }

  void beforePrefillSequence(uint64_t requestId, uint64_t position, std::span<const uint32_t> inputTokens,
                             uint32_t rowBegin) override {
    const auto rows = static_cast<uint32_t>(inputTokens.size());
    const std::vector<uint32_t> &history = historyOf(requestId);
    // The single prefetch slot stays with the first sequence of a packed batch; the others gather synchronously.
    const bool mine = !prefetch_ || prefetch_->requestId == requestId;
    if (!mine || !takePrefetch(requestId, position, inputTokens, uint64_t{rowBegin} * heads_))
      gather(position, inputTokens, history, scratch_.prefillRows, scratch_.prefillSlots,
             uint64_t{rowBegin} * heads_, uint64_t{rowBegin} * heads_, rows, rows);
    if (mine) startPrefetch(requestId, position + rows, rows, history);
    writeBlocks(scratch_.prefillBlocks, uint64_t{rowBegin} * sizeof(Qwen4QsaBlock), position, rows);
  }

  // The QSA blocks a step's rows complete depend on the position alone, and an MTP draft pools them before the
  // proposals (hence the PLE rows) exist.
  void beforeDecodeLane(uint64_t /*requestId*/, uint32_t lane, uint32_t /*stateLane*/, uint64_t position,
                        uint32_t /*anchor*/) override {
    writeBlocks(scratch_.decodeBlocks.at(lane), 0, position, ExecutionLimits::targetVerifyRows);
  }

  void beforeVerifyLane(uint64_t requestId, uint32_t lane, uint64_t position, std::span<const uint32_t> rowTokens,
                        uint32_t retainedRows) override {
    const auto rows = static_cast<uint32_t>(rowTokens.size());
    gather(position, rowTokens, historyOf(requestId), scratch_.decodeRows.at(lane), scratch_.decodeSlots.at(lane), 0,
           uint64_t{lane} * rows * heads_, rows, std::min(retainedRows, rows));
  }

  void afterCommit(uint64_t requestId, std::optional<uint32_t> previousAnchor,
                   std::span<const uint32_t> selected) override {
    const auto found = history_.find(requestId);
    if (found == history_.end()) return;
    std::vector<uint32_t> &history = found->second;
    if (previousAnchor) history.push_back(*previousAnchor);
    history.insert(history.end(), selected.begin(), selected.end() - 1);
  }

private:
  // A request the family never saw (a warmup) has an empty history.
  const std::vector<uint32_t> &historyOf(uint64_t requestId) const {
    static const std::vector<uint32_t> none;
    const auto found = history_.find(requestId);
    return found == history_.end() ? none : found->second;
  }

  // Chunk c+1's rows, read on a worker while chunk c runs on the GPU. The worker owns its buffers and the ids
  // (hashed on the engine thread), so it touches only the store, whose use is serialized by storeMutex_.
  struct Prefetch {
    uint64_t requestId = 0, position = 0;
    std::vector<uint32_t> tokens, ids, slots;
    std::vector<uint8_t> staging;
    std::future<PleStore::Result> result;
  };

  void discard(const PleStore::Result &result) {
    ++total_.discarded;
    total_.discardedBytes += result.stats.bytes;
  }

  void dropPrefetch() {
    if (!prefetch_) return;
    if (prefetch_->result.valid()) discard(prefetch_->result.get());
    prefetch_.reset();
  }

  // Guesses the next chunk is as long as this one; takePrefetch checks the guess and falls back to a sync gather.
  void startPrefetch(uint64_t requestId, uint64_t next, uint32_t rows, const std::vector<uint32_t> &history) {
    dropPrefetch();
    if (!prefetchEnabled_ || next >= history.size()) return;
    const auto count = static_cast<uint32_t>(std::min<uint64_t>(rows, history.size() - next));
    Prefetch &p = prefetch_.emplace();
    p.requestId = requestId;
    p.position = next;
    p.tokens.assign(history.begin() + next, history.begin() + next + count);
    p.ids.resize(uint64_t{count} * heads_);
    p.slots.resize(p.ids.size());
    p.staging.resize(p.ids.size() * rowBytes_);
    hashIds(next, p.tokens, history, count, count, p.ids.data());
    p.result = std::async(std::launch::async, [this, &p] {
      std::lock_guard lock(storeMutex_);
      return store_.fetch(p.ids, p.staging, p.slots);
    });
  }

  // Copies the prefetched rows of this chunk into the step's staging, or returns false (nothing written) so the
  // caller gathers synchronously: no prefetch, a different chunk than guessed, a failed read, or too small buffers.
  bool takePrefetch(uint64_t requestId, uint64_t position, std::span<const uint32_t> tokens, uint64_t first) {
    if (!prefetch_) return false;
    Prefetch &p = *prefetch_;
    const PleStore::Result result = p.result.get();
    const uint64_t count = tokens.size() * heads_, unique = uint64_t{result.stats.hits} + result.stats.misses;
    // A shorter chunk than guessed is a prefix: each row's ids depend on earlier tokens only, and its slots index
    // into the guess's unique rows, which are all staged.
    const bool usable = result.ok() && p.requestId == requestId && p.position == position &&
                        tokens.size() <= p.tokens.size() &&
                        std::equal(tokens.begin(), tokens.end(), p.tokens.begin()) &&
                        scratch_.prefillSlots.sizeBytes() >= (first + count) * sizeof(uint32_t) &&
                        scratch_.prefillRows.sizeBytes() >= (first + unique) * rowBytes_;
    if (usable) {
      auto *slotWords = static_cast<uint32_t *>(scratch_.prefillSlots.contents()) + first;
      for (uint64_t i = 0; i < count; ++i) slotWords[i] = p.slots[i] + static_cast<uint32_t>(first);
      std::memcpy(static_cast<uint8_t *>(scratch_.prefillRows.contents()) + first * rowBytes_, p.staging.data(),
                  unique * rowBytes_);
      addStats(result.stats);
      ++total_.taken;
    } else {
      discard(result);
    }
    prefetch_.reset();
    return usable;
  }

  void addStats(const PleStore::Stats &stats) {
    total_.hits += stats.hits;
    total_.misses += stats.misses;
    total_.dupes += stats.dupes;
    total_.bytes += stats.bytes;
    total_.ns += stats.ns;
    if (log_ && ++fetches_ % 64 == 0) logStats("running");
  }

  void hashIds(uint64_t position, std::span<const uint32_t> tokens, const std::vector<uint32_t> &history,
               uint32_t rows, uint32_t real, uint32_t *ids) const {
    for (uint32_t row = 0; row < real; ++row)
      hasher_.rows(
          position + row, tokens[row],
          [&](uint64_t at) -> std::optional<uint32_t> {
            if (at >= position) return tokens[at - position];
            if (at < history.size()) return history[at];
            return std::nullopt;
          },
          ids + uint64_t{row} * heads_);
    for (uint32_t row = real; row < rows; ++row)
      std::copy_n(ids, heads_, ids + uint64_t{row} * heads_);
  }

  // The PLE rows of `tokens` at `position` on, whose predecessors come from
  // `history` (the tokens before the first row) and the rows themselves: the
  // table's unique rows into `rows * heads` slots of `staging` from slot
  // `first`, and each (row, head)'s slot, plus `slotBase` (the staging's
  // place in the buffer the graph gathers from), into `slots`. Rows from `real` on are never read: they take their
  // first row's rows, which costs no further fetch.
  void gather(uint64_t position, std::span<const uint32_t> tokens, const std::vector<uint32_t> &history,
              const metal::MetalBuffer &staging, const metal::MetalBuffer &slots, uint64_t first, uint64_t slotBase,
              uint32_t rows, uint32_t real) {
    const uint64_t count = uint64_t{rows} * heads_;
    if (slots.sizeBytes() < (first + count) * sizeof(uint32_t) ||
        staging.sizeBytes() < (first + count) * rowBytes_)
      throw std::logic_error("PLE staging is smaller than a step's rows (" + std::to_string(staging.sizeBytes()) + " B for " +
                             std::to_string(first + count) + " rows of " + std::to_string(rowBytes_) + " B)");
    ids_.resize(count);
    hashIds(position, tokens, history, rows, real, ids_.data());
    auto *slotWords = static_cast<uint32_t *>(slots.contents()) + first;
    auto *stagedRows = static_cast<uint8_t *>(staging.contents()) + first * rowBytes_;
    const PleStore::Result result = [&] {
      std::lock_guard lock(storeMutex_);
      return store_.fetch(ids_, {stagedRows, count * rowBytes_}, {slotWords, count});
    }();
    if (!result.ok()) throw std::runtime_error("PLE rows could not be read: " + result.error);
    if (slotBase)
      for (uint64_t i = 0; i < count; ++i) slotWords[i] += static_cast<uint32_t>(slotBase);
    addStats(result.stats);
  }

  // The records of the QSA blocks whose last token is one of `rows` rows from
  // `position`: their first token's position and, for text, RoPE position.
  // ponytail: text-only RoPE positions; Flash-Next serves no vision tower.
  static void writeBlocks(const metal::MetalBuffer &buffer, uint64_t offset, uint64_t position, uint32_t rows) {
    auto *records = reinterpret_cast<Qwen4QsaBlock *>(static_cast<uint8_t *>(buffer.contents()) + offset);
    uint32_t count = 0;
    for (uint64_t last = position; last < position + rows; ++last) {
      if (last % QWEN4_QSA_BLOCK != QWEN4_QSA_BLOCK - 1) continue;
      const auto first = static_cast<uint32_t>(last + 1 - QWEN4_QSA_BLOCK);
      if (offset + (count + 1) * sizeof(Qwen4QsaBlock) > buffer.sizeBytes())
        throw std::logic_error("QSA block records are smaller than a step's blocks");
      records[count++] = {first, {first, first, first}};
    }
  }

  void logStats(const char *label) const {
    std::fprintf(stderr,
                 "ple[%s]: fetches=%llu unique hits=%llu misses=%llu dupes=%llu read=%.1f MiB fetch=%.1f ms "
                 "(%.3f ms/fetch) prefetch taken=%llu discarded=%llu (%.1f MiB)\n",
                 label, static_cast<unsigned long long>(fetches_), static_cast<unsigned long long>(total_.hits),
                 static_cast<unsigned long long>(total_.misses), static_cast<unsigned long long>(total_.dupes),
                 double(total_.bytes) / (1 << 20), double(total_.ns) / 1e6,
                 fetches_ ? double(total_.ns) / 1e6 / double(fetches_) : 0.0,
                 static_cast<unsigned long long>(total_.taken), static_cast<unsigned long long>(total_.discarded),
                 double(total_.discardedBytes) / (1 << 20));
  }

  struct Totals {
    uint64_t hits = 0, misses = 0, dupes = 0, bytes = 0, ns = 0, taken = 0, discarded = 0, discardedBytes = 0;
  };
  PleHasher hasher_;
  PleStore store_;
  uint32_t heads_, rowBytes_;
  bool log_, prefetchEnabled_;
  FamilyScratch scratch_;
  std::unordered_map<uint64_t, std::vector<uint32_t>> history_;
  std::vector<uint32_t> ids_;
  Totals total_;
  uint64_t fetches_ = 0;
  std::mutex storeMutex_;
  std::optional<Prefetch> prefetch_; // last: joined before the members it uses go
};

} // namespace

uint64_t familyHostBytes(const ModelPackage &package) {
  return std::holds_alternative<Qwen4ExpWeights>(package.target) ? cacheBytesFromEnvironment() : 0;
}

std::unique_ptr<FamilyRuntime> makeFamilyRuntime(const ModelPackage &package) {
  const auto *weights = std::get_if<Qwen4ExpWeights>(&package.target);
  if (!weights) return nullptr;
  return std::make_unique<Qwen4Family>(*weights);
}

} // namespace splash::model
