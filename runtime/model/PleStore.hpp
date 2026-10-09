#pragma once

// Qwen3.8-Flash-Next per-layer-embedding (PLE) rows read from the GGUF shard on
// SSD: the hash that picks a token's rows, a CLOCK row cache, a pread pool and
// the fetch that stages the unique rows of a step. Platform neutral; the
// engine thread owns a PleStore and calls it alone.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace splash {

inline constexpr uint32_t pleRowBytes = 90; // 160 values = 5 IQ4_NL blocks
inline constexpr uint32_t pleMaximumHeads = 16;

struct PleHashParams {
  uint32_t ngram = 3; // 2..4
  uint32_t headsPerNgram = 8;
  uint32_t eos = 0;
  uint64_t multipliers[4]{};
  std::vector<uint32_t> vocabularies; // (ngram - 1) * headsPerNgram
  std::vector<uint32_t> offsets;
  uint32_t heads() const { return (ngram - 1) * headsPerNgram; }
};

// Same hash as the qwen4_ple_hash kernel: the n-gram of a token and its
// predecessors, cut to EOS at an EOS or the sequence start, mixed by xor of
// products, one row per head.
class PleHasher {
public:
  explicit PleHasher(PleHashParams params);
  uint32_t heads() const { return params_.heads(); }
  // before(p) returns the token at position p, or nullopt when there is none.
  template <class Before>
  void rows(uint64_t position, uint32_t token, Before &&before, uint32_t *out) const {
    uint64_t context[4] = {token, params_.eos, params_.eos, params_.eos};
    bool cut = false;
    for (uint32_t back = 1; back < params_.ngram; ++back) {
      const std::optional<uint32_t> previous = position >= back ? before(position - back) : std::nullopt;
      cut = cut || !previous || *previous == params_.eos;
      context[back] = cut ? params_.eos : *previous;
    }
    for (uint32_t gram = 2; gram <= params_.ngram; ++gram) {
      uint64_t mixed = context[0] * params_.multipliers[0];
      for (uint32_t j = 1; j < gram; ++j) mixed ^= context[j] * params_.multipliers[j];
      for (uint32_t g = 0; g < params_.headsPerNgram; ++g) {
        const uint32_t head = (gram - 2) * params_.headsPerNgram + g;
        out[head] = static_cast<uint32_t>(mixed % params_.vocabularies[head] + params_.offsets[head]);
      }
    }
  }

private:
  PleHashParams params_;
};

// The table inside a GGUF shard, read in place. Throws on open failure.
class PleSource {
public:
  // validate returns an error message, empty when the shard is acceptable.
  using Validate = std::function<std::string(const PleSource &)>;
  PleSource(const std::string &path, uint64_t dataOffset, uint64_t rows, const Validate &validate = {});
  PleSource(PleSource &&other) noexcept;
  PleSource(const PleSource &) = delete;
  PleSource &operator=(const PleSource &) = delete;
  PleSource &operator=(PleSource &&) = delete;
  ~PleSource();

  int fd() const { return fd_; }
  uint64_t rows() const { return rows_; }
  uint64_t rowOffset(uint64_t row) const { return dataOffset_ + row * pleRowBytes; }
  // Empty when the open file and the path still are the shard opened.
  std::string checkIdentity() const;

private:
  struct Identity {
    uint64_t device = 0, inode = 0, size = 0;
    int64_t mtimeSec = 0, mtimeNsec = 0;
    bool operator==(const Identity &) const = default;
  };
  std::string path_;
  int fd_ = -1;
  uint64_t dataOffset_ = 0, rows_ = 0;
  Identity identity_;
};

// CLOCK cache of rows keyed by row id: fixed arrays, linear-probing index at
// most 60% full, about 100 B per row. No locks.
class PleRowCache {
public:
  explicit PleRowCache(uint64_t bytes); // 0 disables
  uint32_t capacity() const { return capacity_; }
  uint32_t size() const { return size_; }
  // The row, marking it referenced, or null.
  const uint8_t *find(uint32_t id);
  // Evicts by CLOCK when full; a hot row survives its first sweep, a cold one does not.
  void insert(uint32_t id, const uint8_t *row, bool hot);
  void clear();

private:
  uint32_t probe(uint32_t id) const;
  void erase(uint32_t slot);
  uint32_t capacity_ = 0, size_ = 0, hand_ = 0, mask_ = 0;
  std::vector<uint32_t> index_; // slot + 1, 0 empty
  std::vector<uint32_t> keys_;
  std::vector<uint8_t> state_; // 0 free, 1 cold, 2 hot
  std::vector<uint8_t> rows_;
};

// Reads rows with a pool of threads; the caller works too.
class PleReader {
public:
  struct Miss {
    uint32_t id;
    uint32_t slot; // output row index
  };
  static constexpr uint32_t inlineMisses = 8;
  explicit PleReader(uint32_t threads);
  PleReader(const PleReader &) = delete;
  PleReader &operator=(const PleReader &) = delete;
  ~PleReader();
  // Reads each miss's row into out + slot * pleRowBytes. Returns the bytes
  // read from the device, or an error message in `error`.
  uint64_t read(const PleSource &source, std::span<const Miss> misses, uint8_t *out, std::string &error);

private:
  struct Job {
    const PleSource *source = nullptr;
    const Miss *misses = nullptr;
    size_t count = 0;
    uint8_t *out = nullptr;
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::string error;
  };
  static void work(Job &job);
  void run();
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_, done_;
  Job *job_ = nullptr;
  uint64_t generation_ = 0;
  uint32_t active_ = 0;
  bool stop_ = false;
};

class PleStore {
public:
  struct Stats {
    uint32_t hits = 0, misses = 0, dupes = 0; // hits and misses count unique rows
    uint64_t bytes = 0;                       // read from the device
    uint64_t ns = 0;
  };
  struct Result {
    Stats stats;
    std::string error; // empty on success
    bool ok() const { return error.empty(); }
  };
  PleStore(PleSource source, uint64_t cacheBytes, uint32_t threads = 32);

  // Writes the unique rows of ids, in ascending id order, to staging and each
  // id's index among them to slots. An out-of-range id fails before anything
  // is written; a failed read leaves staging undefined and poisons the store.
  Result fetch(std::span<const uint32_t> ids, std::span<uint8_t> staging, std::span<uint32_t> slots);
  bool poisoned() const { return poisoned_; }
  void clearCache() { cache_.clear(); }

private:
  PleSource source_;
  PleRowCache cache_;
  PleReader reader_;
  bool poisoned_ = false;
  std::vector<uint64_t> order_;
  std::vector<PleReader::Miss> misses_;
  std::vector<uint32_t> uniqueIds_;
};

} // namespace splash
