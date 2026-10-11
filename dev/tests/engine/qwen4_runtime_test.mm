// Qwen3.8-Flash-Next end to end on a small synthetic model (the compiled GDN, attention and MoE kernel shapes,
// 4 layers, 8 experts, sparse QSA past 32 tokens): the PLE rows a step gathers from the store's staging equal
// the full table's, and the Runtime's prefill, chunked prefill and decode (verify) graphs agree on their
// logits and selections.
#import <Foundation/Foundation.h>

#include "GgufFixtures.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "metal/MetalBackend.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/PleStore.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/Runtime.hpp"
#include "ops/Embedding.hpp"
#include "ops/PageStorage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace splash;
using namespace gguf_fixtures;
using splash::test::require;

namespace {

constexpr uint32_t kVocabulary = 256;

model::Qwen4ExpLayout layoutOfTarget() {
  model::Qwen4ExpLayout layout;
  layout.layers = 4;
  layout.hiddenSize = 256;
  layout.vocabularySize = kVocabulary;
  layout.maximumContextTokens = 4096;
  layout.experts = 8;
  layout.expertsPerToken = 2;
  layout.indexerTopBlocks = 8;
  layout.pleHeadsPerNgram = 2;
  // Tokens no selection reaches, so a stop never ends a request.
  layout.stopTokens = {kVocabulary, kVocabulary + 1};
  layout.maskToken = kVocabulary - 1;
  layout.hiddenCaptureLayers = {3};
  return layout;
}

std::vector<uint8_t> tableBytes(const model::Qwen4PleTable &table) {
  std::vector<uint8_t> bytes(table.rows * table.rowBytes);
  std::ifstream file(table.path, std::ios::binary);
  file.seekg(static_cast<std::streamoff>(table.dataOffset));
  file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(file.good(), "the PLE table could not be read");
  return bytes;
}

metal::MetalBuffer sharedBuffer(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
}

// The rows of 160 values the embedding gather returns for `slots` of a table, as bf16 bits.
std::vector<uint16_t> gather(metal::MetalBackend &backend, const metal::MetalBuffer &rows, uint32_t tableRows,
                             const std::vector<uint32_t> &slots, uint32_t format) {
  metal::MetalBuffer indices = sharedBuffer(backend, slots.size() * 4, "ple-test-indices");
  std::memcpy(indices.contents(), slots.data(), slots.size() * 4);
  metal::MetalBuffer output = sharedBuffer(backend, slots.size() * 160 * 2, "ple-test-output");
  metal::CommandGraph graph;
  ops::Embedding::add(graph, indices, ops::EmbeddingWeights(tableRows, 160, ops::NativeRows(rows, format)),
                      output, static_cast<uint32_t>(slots.size()));
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const auto *values = static_cast<const uint16_t *>(output.contents());
  return {values, values + slots.size() * 160};
}

// What a step gathers through the store's staging equals what the same ids gather from the whole table.
void checkPleStaging(metal::MetalBackend &backend, const model::Qwen4PleTable &table) {
  const std::vector<uint8_t> bytes = tableBytes(table);
  metal::MetalBuffer whole = sharedBuffer(backend, bytes.size(), "ple-test-table");
  std::memcpy(whole.contents(), bytes.data(), bytes.size());
  std::mt19937 rng(7);
  // 4 rows of 4 heads, with repeats within and across rows.
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < 16; ++i) ids.push_back(rng() % table.rows);
  ids[5] = ids[1];
  ids[11] = ids[1];
  ids[14] = ids[2];
  PleStore store(PleSource(table.path.string(), table.dataOffset, table.rows, table.rowBytes), 64 << 10);
  for (const uint64_t cache : {0, 1}) {
    static_cast<void>(cache);
    metal::MetalBuffer staging = sharedBuffer(backend, ids.size() * table.rowBytes, "ple-test-staging");
    std::vector<uint32_t> slots(ids.size());
    // The second pass answers from the row cache.
    const PleStore::Result result =
        store.fetch(ids, {static_cast<uint8_t *>(staging.contents()), ids.size() * table.rowBytes}, slots);
    require(result.ok(), "the PLE store failed to fetch: " + result.error);
    require(cache == 0 ? result.stats.misses > 0 : result.stats.hits > 0 && result.stats.misses == 0,
            "the second fetch did not come from the cache");
    require(result.stats.dupes == 3, "the repeated ids were not deduplicated");
    const auto viaStaging = gather(backend, staging, static_cast<uint32_t>(ids.size()), slots, table.format);
    const auto viaTable = gather(backend, whole, static_cast<uint32_t>(table.rows), ids, table.format);
    require(viaStaging == viaTable, "gathering from the staging differs from gathering from the whole table");
    require(std::any_of(viaTable.begin(), viaTable.end(), [](uint16_t v) { return v != 0; }),
            "the gathered PLE rows are all zero");
  }
  std::printf("ple staging gather == full-table gather: ok\n");
}

// The same check on a Q8_0 table (170-byte rows across page boundaries), written here with random int8 quants.
void checkQ80PleStaging(metal::MetalBackend &backend) {
  constexpr uint64_t rows = 600, offset = 4096 + 123;
  const std::filesystem::path path = std::filesystem::temp_directory_path() / ("ple-q80-" + std::to_string(getpid()));
  std::mt19937 rng(11);
  std::vector<uint8_t> bytes(offset + rows * pleMaxRowBytes);
  for (uint64_t at = offset; at < bytes.size(); ++at) {
    const uint64_t inBlock = (at - offset) % 34;
    bytes[at] = inBlock == 0 ? 0x66 : inBlock == 1 ? 0x2E : static_cast<uint8_t>(rng()); // d = 0.1, then quants
  }
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
  model::Qwen4PleTable table;
  table.path = path;
  table.dataOffset = offset;
  table.rows = rows;
  table.rowBytes = pleMaxRowBytes;
  table.format = GGUF_FMT_Q80;
  checkPleStaging(backend, table);
  std::filesystem::remove(path);
}

// A family that stages the table's first row for every (row, head): the PLE layer gathers row 0 everywhere,
// whatever the tokens are.
class FirstRowFamily final : public model::FamilyRuntime {
public:
  explicit FirstRowFamily(std::vector<uint8_t> row) : row_(std::move(row)) {}
  void attach(const model::FamilyScratch &scratch) override {
    for (const metal::MetalBuffer &buffer : {scratch.prefillRows, scratch.decodeRows.at(0)})
      for (uint64_t at = 0; at + pleRowBytes <= buffer.sizeBytes(); at += pleRowBytes)
        std::memcpy(static_cast<uint8_t *>(buffer.contents()) + at, row_.data(), pleRowBytes);
    // The decode lanes' staging is one packed buffer; every lane's slice is filled the same way.
    for (const metal::MetalBuffer &buffer : scratch.decodeRows)
      for (uint64_t at = 0; at + pleRowBytes <= buffer.sizeBytes(); at += pleRowBytes)
        std::memcpy(static_cast<uint8_t *>(buffer.contents()) + at, row_.data(), pleRowBytes);
  }

private:
  std::vector<uint8_t> row_;
};

// One decode cycle of a request: what it emitted and how many of its proposals the verify accepted.
struct Cycle final {
  std::vector<uint32_t> outputs;
  uint32_t drafted = 0, accepted = 0;
};

// The family the model installs, which also records what each verify stages: the rows' tokens and the PLE rows
// and slots the family left for them.
class SpyFamily final : public model::FamilyRuntime {
public:
  struct Call final {
    uint64_t request = 0;
    uint32_t lane = 0;
    uint64_t position = 0;
    std::vector<uint32_t> tokens, slots;
    std::vector<uint8_t> staged;
    uint32_t retained = 0;
  };

  SpyFamily(std::unique_ptr<model::FamilyRuntime> inner, uint32_t heads) : inner_(std::move(inner)), heads_(heads) {}
  void attach(const model::FamilyScratch &scratch) override {
    scratch_ = scratch;
    inner_->attach(scratch);
  }
  void onActivate(uint64_t request, uint32_t lane, std::span<const uint32_t> prompt) override {
    inner_->onActivate(request, lane, prompt);
  }
  void onEnd(uint64_t request) override { inner_->onEnd(request); }
  void beforePrefillSequence(uint64_t request, uint64_t position, std::span<const uint32_t> input,
                             uint32_t rowBegin) override {
    inner_->beforePrefillSequence(request, position, input, rowBegin);
  }
  void beforeDecodeLane(uint64_t request, uint32_t lane, uint32_t stateLane, uint64_t position,
                        uint32_t anchor) override {
    inner_->beforeDecodeLane(request, lane, stateLane, position, anchor);
  }
  void beforeVerifyLane(uint64_t request, uint32_t lane, uint64_t position, std::span<const uint32_t> rowTokens,
                        uint32_t retainedRows) override {
    inner_->beforeVerifyLane(request, lane, position, rowTokens, retainedRows);
    Call call{request, lane, position, {rowTokens.begin(), rowTokens.end()}, {}, {}, retainedRows};
    const uint64_t count = uint64_t{heads_} * rowTokens.size();
    const auto *slots = static_cast<const uint32_t *>(scratch_.decodeSlots.at(lane).contents());
    call.slots.assign(slots, slots + count);
    const auto *rows = static_cast<const uint8_t *>(scratch_.decodeRows.at(lane).contents());
    call.staged.assign(rows, rows + count * pleRowBytes);
    calls.push_back(std::move(call));
  }
  void afterCommit(uint64_t request, std::optional<uint32_t> previous, std::span<const uint32_t> selected) override {
    inner_->afterCommit(request, previous, selected);
  }

  std::vector<Call> calls;

private:
  std::unique_ptr<model::FamilyRuntime> inner_;
  uint32_t heads_;
  model::FamilyScratch scratch_;
};

struct Run final {
  uint64_t id = 0;
  std::vector<uint32_t> prompt;
  std::vector<uint32_t> tokens;
  std::vector<ops::TokenLogprobs> logprobs;
  std::vector<Cycle> cycles;
  uint32_t drafted = 0, accepted = 0;
};

class Session final {
public:
  // `replacement` takes the place of the model's family runtime.
  Session(metal::MetalBackend &backend, const model::ModelPackage &package,
          std::unique_ptr<model::FamilyRuntime> replacement = nullptr)
      : operators_(backend.capabilities()) {
    const kv::Layout layout = package.targetKvLayout(kv::Format::Int8);
    const uint32_t budget = std::max(128U, layout.minimumExtentPages());
    const uint32_t extent = layout.extentPagesFor(budget);
    const uint32_t count = budget - budget % extent;
    const metal::AllocationAdmission admit = [](uint64_t, const std::function<void()> &allocate) {
      allocate();
      return metal::AllocationResult{};
    };
    pages_ = std::make_unique<kv::PageStorage>(backend, admit, layout, count, extent);
    for (uint32_t index = 0; index < count / extent; ++index)
      require(static_cast<bool>(pages_->allocateExtent(index)), "a KV extent was refused");
    states_ = std::make_unique<model::QwenStateStorage>(backend, admit, package.stateLayout(), nullptr);
    model::RuntimeContext context{backend, package, *pages_, *states_, operators_};
    context.prefillRows = 256;
    runtime_ = std::make_unique<model::Runtime>(context);
    if (replacement) runtime_->installFamily(std::move(replacement));
  }

  // Raw logits at ids 0..254 of the position after `prompt`, prefilled in chunks of `chunk` rows.
  std::vector<float> score(const std::vector<uint32_t> &prompt, uint32_t chunk) {
    std::vector<uint32_t> ids(kVocabulary - 1);
    std::iota(ids.begin(), ids.end(), 0);
    ModelRequest request;
    request.id = ++nextId_;
    request.prompt = prompt;
    request.scoreTokens = ids;
    begin(request);
    ModelStepResult result;
    for (uint32_t at = 0; at < prompt.size(); at += chunk)
      result = prefill(request.id, prompt, at, std::min<uint32_t>(chunk, prompt.size() - at));
    runtime_->end(request.id);
    require(result.failure.empty() && result.scoreLogits.size() == ids.size(), "the score request failed");
    return result.scoreLogits;
  }

  // `count` greedy tokens after `prompt`: the first prefill's selection, then decode cycles.
  Run generate(const std::vector<uint32_t> &prompt, uint32_t count, uint32_t logprobs = 0, uint32_t chunk = 0) {
    return generateTogether({prompt}, count, logprobs, chunk).front();
  }

  // The same for several prompts at once, decoded in one batch of that width.
  std::vector<Run> generateTogether(const std::vector<std::vector<uint32_t>> &prompts, uint32_t count,
                                    uint32_t logprobs = 0, uint32_t chunk = 0) {
    std::vector<ModelRequest> requests(prompts.size());
    std::vector<Run> runs(prompts.size());
    std::vector<uint64_t> positions;
    for (size_t i = 0; i < prompts.size(); ++i) {
      requests[i].id = ++nextId_;
      requests[i].prompt = prompts[i];
      requests[i].maxNewTokens = count;
      requests[i].logprobs = logprobs;
      begin(requests[i]);
      runs[i].id = requests[i].id;
      runs[i].prompt = prompts[i];
      const auto size = static_cast<uint32_t>(prompts[i].size());
      for (uint32_t at = 0; at < size; at += chunk ? chunk : size) {
        const ModelStepResult first = prefill(requests[i].id, prompts[i], at, std::min(chunk ? chunk : size, size - at));
        require(first.failure.empty() && first.outputTokens.empty(), "the prompt's prefill did not leave an anchor");
      }
      positions.push_back(prompts[i].size());
    }
    const auto done = [&] {
      return std::all_of(runs.begin(), runs.end(), [&](const Run &run) { return run.tokens.size() >= count; });
    };
    while (!done()) {
      BatchPlan plan{.kind = WorkKind::Decode};
      std::vector<std::vector<uint32_t>> tables;
      std::vector<ModelBatchItem> items;
      for (size_t i = 0; i < prompts.size(); ++i) {
        plan.items.push_back({requests[i].id, 0});
        tables.push_back(pageTable(requests[i].id));
      }
      for (size_t i = 0; i < prompts.size(); ++i) {
        items.push_back({requests[i].id, positions[i], 0, tables[i]});
        items.back().pageTableRevision = 1;
      }
      auto results = polled_ ? poll(plan, items) : runtime_->decode(plan, items);
      require(results.size() == prompts.size(), "a decode batch lost a lane");
      for (size_t i = 0; i < prompts.size(); ++i) {
        require(results[i].failure.empty() && !results[i].outputTokens.empty(),
                "a decode cycle failed: " + results[i].failure);
        Run &run = runs[i];
        run.tokens.insert(run.tokens.end(), results[i].outputTokens.begin(), results[i].outputTokens.end());
        run.logprobs.insert(run.logprobs.end(), results[i].outputLogprobs.begin(), results[i].outputLogprobs.end());
        run.drafted += results[i].draftedTokens;
        run.accepted += results[i].acceptedDraftTokens;
        run.cycles.push_back({results[i].outputTokens, results[i].draftedTokens, results[i].acceptedDraftTokens});
        positions[i] += results[i].outputTokens.size() - results[i].outputTokensWithoutKv;
      }
    }
    for (size_t i = 0; i < prompts.size(); ++i) {
      runtime_->end(requests[i].id);
      runs[i].tokens.resize(count);
    }
    return runs;
  }

  // Decode through the ticket the way the engine does: poll it until it is ready, then take its results.
  void pollTickets(bool polled) { polled_ = polled; }

private:
  std::vector<ModelStepResult> poll(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    const std::unique_ptr<ModelBatchTicket> ticket = runtime_->submit(plan, items, {});
    while (true) {
      static_cast<void>(ticket->takeMaskRequests());
      if (ticket->ready()) break;
      std::this_thread::yield();
    }
    return ticket->wait();
  }

  void begin(const ModelRequest &request) {
    runtime_->beginColdRequest(request, static_cast<uint32_t>(request.id % 4));
  }

  // A request's pages: 16 of the pool's, its own range.
  std::vector<uint32_t> pageTable(uint64_t id) const {
    std::vector<uint32_t> pages(16);
    std::iota(pages.begin(), pages.end(), static_cast<uint32_t>(id % 4) * 16);
    return pages;
  }

  ModelStepResult prefill(uint64_t id, const std::vector<uint32_t> &prompt, uint32_t at, uint32_t rows) {
    BatchPlan plan{.kind = WorkKind::Prefill, .items = {{id, rows}}};
    const std::vector<uint32_t> pages = pageTable(id);
    ModelBatchItem item{id, at, rows, pages};
    item.pageTableRevision = 1;
    item.inputTokens = std::span<const uint32_t>(prompt).subspan(at, rows);
    auto results = runtime_->prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    require(results.size() == 1 && results[0].consumedPromptTokens == rows, "a prefill chunk failed");
    return std::move(results[0]);
  }

  ops::ExecutionPlans operators_;
  std::unique_ptr<kv::PageStorage> pages_;
  std::unique_ptr<model::QwenStateStorage> states_;
  std::unique_ptr<model::Runtime> runtime_;
  uint64_t nextId_ = 100;
  bool polled_ = false;
};

struct Closeness final {
  double cosine = 0, maximumAbsolute = 0, scale = 0;
};

std::vector<float> centered(std::vector<float> values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  for (float &value : values) value -= static_cast<float>(mean);
  return values;
}

// How close two logit vectors are once their common offset (which the random weights make large) is gone.
Closeness compare(const std::vector<float> &left, const std::vector<float> &right) {
  const std::vector<float> a = centered(left), b = centered(right);
  double dot = 0, na = 0, nb = 0, maximum = 0, scale = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
    maximum = std::max(maximum, std::fabs(double(a[i]) - b[i]));
    scale = std::max(scale, std::fabs(double(a[i])));
  }
  return {dot / std::sqrt(na * nb), maximum, scale};
}

uint32_t argmax(const std::vector<float> &values) {
  return static_cast<uint32_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

// The MTP head of a Flash-Next target as unsloth ships it: its block 4 is a full-attention block (here the target's
// last one, weights and all) with the nextn tensors beside it, whose values are random.
void writeMtpHead(const std::filesystem::path &path, const SmallTarget &target) {
  const model::gguf::TargetGeometry &g = target.geometry;
  const uint64_t streams = g.streamWidth();
  std::vector<Tensor> head;
  for (const Tensor &tensor : target.tensors) {
    if (tensor.name.starts_with("blk.3.")) {
      head.push_back(tensor);
      head.back().name = "blk.4." + tensor.name.substr(6);
    } else if (tensor.name.starts_with("output_hc_")) {
      head.push_back(tensor);
      head.back().name = "blk.4.nextn.hc_head_" + tensor.name.substr(10);
    }
  }
  // Norm values near one come from their name; the tensors take their real names after.
  const auto add = [&](const std::string &name, std::vector<uint64_t> dims, uint32_t type) {
    uint64_t elements = 1;
    for (uint64_t dim : dims) elements *= dim;
    const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
    head.push_back({name, std::move(dims), type, test_gguf::Bytes(elements / traits.blockElements * traits.blockBytes)});
  };
  const size_t nextn = head.size();
  add("blk.4.nextn.eh_proj.weight", {2ull * g.hiddenSize, g.hiddenSize}, kQ8_0);
  add("blk.4.nextn.enorm_norm.weight", {g.hiddenSize}, model::ggml::kF32);
  add("blk.4.nextn.hnorm_norm.weight", {streams}, model::ggml::kF32);
  std::vector<Tensor> fresh(head.begin() + static_cast<std::ptrdiff_t>(nextn), head.end());
  randomizeRealistic(fresh, 2200);
  for (size_t i = 0; i < fresh.size(); ++i) {
    head[nextn + i].data = std::move(fresh[i].data);
    auto &name = head[nextn + i].name;
    if (const size_t at = name.find("_norm"); at != std::string::npos) name.erase(at, 5);
  }
  std::vector<test_gguf::Key> keys = metadata(g);
  keys[1] = test_gguf::uint32Key("qwen4exp.block_count", g.layers + 1);
  model::gguf::TargetGeometry noPle = g;
  noPle.pleNgram = 0;
  qwen4Metadata(keys, noPle, 1);
  splash::test::writeFile(path, test_gguf::file(keys, head));
}

// The expected value of a PLE row from the whole table.
void checkStaging(const std::vector<SpyFamily::Call> &calls, const std::vector<Run> &runs, const model::Qwen4PleTable &table,
                  const std::vector<uint8_t> &rows, uint32_t verifyRows) {
  PleHasher hasher(table.hash);
  const uint32_t heads = hasher.heads();
  size_t checked = 0;
  for (const SpyFamily::Call &call : calls) {
    const auto run = std::find_if(runs.begin(), runs.end(), [&](const Run &r) { return r.id == call.request; });
    if (run == runs.end()) continue;
    std::vector<uint32_t> sequence = run->prompt;
    sequence.insert(sequence.end(), run->tokens.begin(), run->tokens.end());
    require(call.position + 1 <= sequence.size() && sequence[call.position] == call.tokens[0],
            "a verify's first row is not the request's anchor");
    require(call.tokens.size() == verifyRows && call.slots.size() == uint64_t{verifyRows} * heads,
            "a verify staged another number of rows");
    const uint64_t base = uint64_t{call.lane} * verifyRows * heads;
    std::vector<uint32_t> ids(heads);
    for (uint32_t row = 0; row < verifyRows; ++row) {
      // Rows past the retained ones are never read; they only point into the staging.
      if (row >= call.retained) {
        for (uint32_t head = 0; head < heads; ++head)
          require((call.slots[uint64_t{row} * heads + head] - base + 1) * pleRowBytes <= call.staged.size(),
                  "a staged slot lies outside the staging");
        continue;
      }
      hasher.rows(
          call.position + row, call.tokens[row],
          [&](uint64_t at) -> std::optional<uint32_t> {
            if (at >= call.position) return call.tokens[at - call.position];
            return sequence[at];
          },
          ids.data());
      for (uint32_t head = 0; head < heads; ++head) {
        const uint64_t slot = call.slots[uint64_t{row} * heads + head] - base;
        require((slot + 1) * pleRowBytes <= call.staged.size(), "a staged slot lies outside the staging");
        require(std::equal(call.staged.begin() + slot * pleRowBytes, call.staged.begin() + (slot + 1) * pleRowBytes,
                           rows.begin() + uint64_t{ids[head]} * pleRowBytes),
                "a verify row's staged PLE row differs from the whole table's");
        ++checked;
      }
    }
  }
  require(checked > 0, "no verify row was checked");
  std::printf("staged PLE rows of %zu verify rows' heads == the whole table's\n", checked);
}

// Each cycle's accepted count is the longest prefix of its proposals that the target's tokens (the emitted ones,
// then the next cycle's anchor) repeat, and the proposals reach the verify rows as they were drafted.
void checkAcceptance(const Run &run, const std::vector<SpyFamily::Call> &calls, uint32_t steps) {
  require(calls.size() == run.cycles.size(), "a decode cycle staged no verify rows");
  size_t distinct = 0, accepted = 0, partial = 0;
  std::vector<char> seen(kVocabulary);
  for (size_t cycle = 0; cycle + 2 < run.cycles.size(); ++cycle) {
    const Cycle &now = run.cycles[cycle];
    const std::vector<uint32_t> &proposals = calls[cycle].tokens;
    require(proposals[0] == now.outputs[0], "a verify's first row is not the cycle's anchor");
    require(now.drafted == steps, "a cycle drafted another number of proposals");
    std::vector<uint32_t> target(now.outputs.begin() + 1, now.outputs.end());
    target.push_back(run.cycles[cycle + 1].outputs[0]);
    uint32_t expected = 0;
    while (expected < steps && expected < target.size() && proposals[expected + 1] == target[expected]) ++expected;
    require(now.accepted == expected && now.outputs.size() == expected + 1,
            "a cycle accepted " + std::to_string(now.accepted) + " proposals, the target's tokens repeat " +
                std::to_string(expected));
    for (uint32_t i = 1; i <= steps; ++i) distinct += !std::exchange(seen[proposals[i]], char{1});
    accepted += expected;
    partial += expected > 0 && expected < steps;
  }
  std::printf("MTP: %zu cycles, %zu distinct proposal tokens, accepted %zu, cycles with a partial acceptance %zu\n",
              run.cycles.size(), distinct, accepted, partial);
  require(distinct > 1, "the MTP head proposes one token only");
  require(accepted > 0, "no MTP proposal was ever accepted");
}

void run(const std::string &metallib) {
  const splash::test::TemporaryDirectory directory("splash-qwen4-runtime");
  setenv("SPLASH_WEIGHT_CACHE", (directory.path() / "cache").c_str(), 1);
  // Lookups from two matching tokens, so a repeating continuation proposes and verifies several rows.
  setenv("SPLASH_LOOKUP_MIN", "2", 1);
  setenv("SPLASH_PLE_CACHE_MB", "1", 1);
  metal::MetalBackend backend(metallib);

  SmallTarget target = smallQwen4Target(true);
  const model::Qwen4ExpLayout layout = layoutOfTarget();
  const model::gguf::TargetGeometry geometry = model::ggufTargetGeometry(layout);
  const auto dir = directory.path() / "model";
  std::filesystem::create_directories(dir);
  std::vector<test_gguf::Key> keys = metadata(target.geometry);
  qwen4Metadata(keys, target.geometry);
  const size_t half = target.tensors.size() / 2;
  const std::vector<Tensor> first(target.tensors.begin(), target.tensors.begin() + half),
      second(target.tensors.begin() + half, target.tensors.end());
  const auto parts = [&](uint32_t no) {
    std::vector<test_gguf::Key> k = no == 0 ? keys : std::vector<test_gguf::Key>{};
    k.push_back(test_gguf::uint32Key("split.no", no));
    k.push_back(test_gguf::uint32Key("split.count", 2));
    k.push_back(test_gguf::uint32Key("split.tensors.count", uint32_t(target.tensors.size())));
    return k;
  };
  splash::test::writeFile(dir / "t-00001-of-00002.gguf", test_gguf::file(parts(0), first));
  splash::test::writeFile(dir / "t-00002-of-00002.gguf", test_gguf::file(parts(1), second));
  model::GgufTargetLoader loader(backend, model::findTargetGgufs(dir), geometry, {});
  loader.prepare();

  model::ModelPackage package;
  package.descriptor = model::makeModelDescriptor(
      "qwen4-test", layout, model::nullDraftLayout(layout.hiddenSize, layout.vocabularySize, layout.capturedHiddenSize()),
      ops::VisionLayout{.outputHiddenSize = layout.hiddenSize});
  package.descriptor.proposals = model::ProposalSource::None;
  package.descriptor.targetSource = model::TargetSource::Gguf;
  package.descriptor.visionSource = model::VisionSource::None;
  package.target = model::loadQwen4ExpWeights(backend, layout, loader);
  package.draft.layout = package.descriptor.draft;
  const model::Qwen4PleTable table = std::get<model::Qwen4ExpWeights>(package.target).pleTable;

  checkPleStaging(backend, table);
  checkQ80PleStaging(backend);
  {
    ops::ExecutionPlans operators(backend.capabilities());
    const model::ModelMemoryPlan plan = model::plannedRuntimeMemory(backend.capabilities(), package, operators);
    require(plan.hostResidentBytes == (uint64_t{1} << 20) && plan.sharedPrefillPlannedAllocatedBytes > 0 &&
                plan.sharedDecodePlannedAllocatedBytes > 0,
            "the memory plan does not count the PLE row cache as host RAM");
  }

  Session session(backend, package);
  std::mt19937 rng(11);
  const auto randomPrompt = [&](uint32_t length) {
    std::vector<uint32_t> tokens(length);
    for (uint32_t &token : tokens) token = rng() % (kVocabulary - 2);
    return tokens;
  };
  const std::vector<uint32_t> prompt = randomPrompt(100);

  // Finite, and a function of the prompt.
  const std::vector<float> whole = session.score(prompt, 100);
  require(std::all_of(whole.begin(), whole.end(), [](float v) { return std::isfinite(v); }), "non-finite logits");
  const Closeness different = compare(whole, session.score(randomPrompt(100), 100));
  std::printf("another prompt: cosine %.4f max |diff| %.3f of centered scale %.3f\n", different.cosine,
              different.maximumAbsolute, different.scale);
  require(different.maximumAbsolute > 0.05 * different.scale, "the logits do not depend on the prompt");

  // Chunked prefill: the PLE history, GDN state, KV and QSA index keys carry across chunks.
  double noise = 0;
  for (const uint32_t chunk : {33U, 17U, 64U}) {
    const Closeness close = compare(whole, session.score(prompt, chunk));
    std::printf("chunked prefill (%u rows): cosine %.6f max |diff| %.5f of scale %.3f\n", chunk, close.cosine,
                close.maximumAbsolute, close.scale);
    require(close.cosine > 0.9999 && close.maximumAbsolute < 0.02 * close.scale,
            "chunked prefill does not match one chunk");
    noise = std::max(noise, close.maximumAbsolute);
  }

  // The PLE rows matter: without them the same prompt's logits move well past the chunking noise.
  const std::vector<uint8_t> tableRows = tableBytes(table);
  Session unstaged(backend, package,
                   std::make_unique<FirstRowFamily>(std::vector<uint8_t>(tableRows.begin(), tableRows.begin() + pleRowBytes)));
  const Closeness without = compare(whole, unstaged.score(prompt, 100));
  std::printf("PLE row 0 for every position: max |diff| %.3f (chunking noise %.4f)\n", without.maximumAbsolute, noise);
  require(without.maximumAbsolute > 10 * noise, "the logits do not depend on the PLE rows");

  // Decode against prefill: each token the decode cycles selected is the argmax a prefill of the tokens
  // before it computes, and the logprobs they report give the same logit differences.
  const Run decoded = session.generate(prompt, 40, 8);
  std::printf("decoded %zu tokens, drafted %u accepted %u:", decoded.tokens.size(), decoded.drafted,
              decoded.accepted);
  for (const uint32_t token : decoded.tokens) std::printf(" %u", token);
  std::printf("\n");
  require(decoded.logprobs.size() >= decoded.tokens.size(), "decode reported no logprobs");
  std::vector<uint32_t> context = prompt;
  double worstLogitGap = 0;
  for (size_t i = 0; i < decoded.tokens.size(); ++i) {
    const std::vector<float> logits = session.score(context, 128);
    // A near tie may go either way: the two graphs round differently, by about 0.1 on logits of this size.
    const float best = logits[argmax(logits)];
    require(decoded.tokens[i] < logits.size() && logits[decoded.tokens[i]] > best - 0.3F,
            "decode selected " + std::to_string(decoded.tokens[i]) + " at token " + std::to_string(i) +
                ", the prefill's argmax is " + std::to_string(argmax(logits)) + " by " +
                std::to_string(best - logits[decoded.tokens[i] < logits.size() ? decoded.tokens[i] : 0]));
    // The reported alternatives' logprob gaps equal the prefill's logit gaps.
    const ops::TokenLogprobs &reported = decoded.logprobs[i];
    for (size_t j = 1; j < reported.topIds.size(); ++j) {
      if (reported.topIds[0] >= logits.size() || reported.topIds[j] >= logits.size()) continue;
      const double gap = double(reported.topLogprobs[0]) - reported.topLogprobs[j];
      const double expected = double(logits[reported.topIds[0]]) - logits[reported.topIds[j]];
      worstLogitGap = std::max(worstLogitGap, std::fabs(gap - expected));
    }
    context.push_back(decoded.tokens[i]);
  }
  std::printf("decode agrees with the prefill's argmax at all %zu tokens; largest logit-gap difference %.4f\n",
              decoded.tokens.size(), worstLogitGap);
  require(worstLogitGap < 0.4, "the decode's logit gaps differ from the prefill's");

  // Lanes: two requests decoded in one batch select what each selects alone (every lane's staged PLE rows,
  // QSA block records and states are its own).
  const std::vector<uint32_t> shorter = randomPrompt(70);
  const Run alone = session.generate(shorter, 16);
  const std::vector<Run> together = session.generateTogether({prompt, shorter}, 16);
  require(std::equal(together[0].tokens.begin(), together[0].tokens.end(), decoded.tokens.begin()) &&
              together[1].tokens == alone.tokens,
          "a request decoded in a batch of two selects other tokens than alone");
  std::printf("batch of two == each alone\n");

  // The MTP head drafts: the same model with the head, whose proposals the verify checks. The tokens it selects
  // are the draft-less decode's (a near tie may go either way); its drafts are what a verify consumed.
  writeMtpHead(dir / "mtp.gguf", target);
  model::GgufMtpLoader head(backend, dir / "mtp.gguf", geometry, {});
  head.prepare();
  model::Qwen4ExpLayout withMtp = layout;
  withMtp.mtpLayers = 1;
  model::ModelPackage mtpPackage;
  mtpPackage.descriptor = model::makeModelDescriptor(
      "qwen4-mtp-test", withMtp,
      model::nullDraftLayout(layout.hiddenSize, layout.vocabularySize, layout.capturedHiddenSize()),
      ops::VisionLayout{.outputHiddenSize = layout.hiddenSize});
  mtpPackage.descriptor.proposals = model::ProposalSource::Mtp;
  mtpPackage.descriptor.targetSource = model::TargetSource::Gguf;
  mtpPackage.descriptor.visionSource = model::VisionSource::None;
  mtpPackage.target = model::loadQwen4ExpWeights(backend, withMtp, loader, &head);
  mtpPackage.draft.layout = mtpPackage.descriptor.draft;
  const uint32_t heads = layout.pleHeads();
  auto spyOwner = std::make_unique<SpyFamily>(model::makeFamilyRuntime(mtpPackage), heads);
  SpyFamily &spy = *spyOwner;
  Session drafting(backend, mtpPackage, std::move(spyOwner));
  constexpr uint32_t steps = 3;
  const std::vector<uint8_t> wholeTable = tableBytes(table);

  setenv("SPLASH_LOOKUP_MIN", "0", 1);
  const Run mtp = drafting.generate(prompt, 40, 8);
  std::printf("MTP decode: %zu tokens in %zu cycles, drafted %u accepted %u:", mtp.tokens.size(), mtp.cycles.size(),
              mtp.drafted, mtp.accepted);
  for (const uint32_t token : mtp.tokens) std::printf(" %u", token);
  std::printf("\n");
  checkAcceptance(mtp, spy.calls, steps);
  checkStaging(spy.calls, {mtp}, table, wholeTable, 8);
  const auto sameAsDraftless = [&](const Run &other, const char *what) {
    for (size_t i = 0; i < other.tokens.size(); ++i) {
      if (other.tokens[i] == decoded.tokens[i]) continue;
      // The verify rows round differently from the draft-less cycle's: only a near tie may flip.
      std::vector<uint32_t> before = prompt;
      before.insert(before.end(), decoded.tokens.begin(), decoded.tokens.begin() + static_cast<std::ptrdiff_t>(i));
      const std::vector<float> logits = session.score(before, 128);
      const float best = logits[argmax(logits)];
      std::printf("%s differs at token %zu: %u against %u, logits %.3f and %.3f of best %.3f\n", what, i,
                  other.tokens[i], decoded.tokens[i], logits[other.tokens[i]], logits[decoded.tokens[i]], best);
      require(logits[other.tokens[i]] > best - 0.3F && logits[decoded.tokens[i]] > best - 0.3F,
              std::string(what) + " left the draft-less decode before a near tie");
      break;
    }
  };
  sameAsDraftless(mtp, "MTP decode");
  std::printf("MTP decode == draft-less decode (or a near tie)\n");

  // A budget that ends mid-cycle leaves the last cycle's later rows inactive in the MoE (they select no expert):
  // the tokens and logprobs it retains are the long run's.
  const auto samePrefix = [&](Session &run, const Run &whole, const char *what) {
    for (const uint32_t budget : {2U, 3U, 5U, 6U, 9U, 11U}) {
      const Run part = run.generate(prompt, budget, 8);
      require(part.tokens.size() == budget && part.logprobs.size() >= budget &&
                  std::equal(part.tokens.begin(), part.tokens.end(), whole.tokens.begin()),
              std::string(what) + " with a budget of " + std::to_string(budget) + " selects other tokens");
      for (uint32_t i = 0; i < budget; ++i)
        require(part.logprobs[i].topIds == whole.logprobs[i].topIds &&
                    part.logprobs[i].topLogprobs == whole.logprobs[i].topLogprobs,
                std::string(what) + " with a budget of " + std::to_string(budget) + " reports other logprobs");
    }
  };
  setenv("SPLASH_LOOKUP_MIN", "2", 1);
  samePrefix(session, decoded, "draft-less decode");
  setenv("SPLASH_LOOKUP_MIN", "0", 1);
  samePrefix(drafting, mtp, "MTP decode");
  std::printf("budgets ending mid-cycle == the long run's prefix\n");

  // Chunked prefill carries the head's streams and KV across chunks, and a batch keeps each lane's own: the same
  // prompts propose the same tokens.
  const size_t spyBefore = spy.calls.size();
  const Run chunked = drafting.generate(prompt, 24, 0, 33);
  require(std::equal(chunked.tokens.begin(), chunked.tokens.end(), mtp.tokens.begin()),
          "chunked prefill changed the MTP decode's tokens");
  // The chunks round the head's KV differently, so a later proposal may flip at a near tie; the first proposal,
  // which every carried stream and KV row of the prompt feeds, may not.
  size_t compared = 0, flipped = 0;
  for (size_t cycle = 0; cycle + 1 < std::min(chunked.cycles.size(), mtp.cycles.size()); ++cycle, ++compared) {
    const std::vector<uint32_t> &whole = spy.calls[cycle].tokens, &parts = spy.calls[spyBefore + cycle].tokens;
    require(whole[1] == parts[1], "chunked prefill changed the MTP head's first proposal");
    flipped += whole != parts;
  }
  std::printf("MTP: chunked prefill (33) flipped a later proposal in %zu of %zu cycles\n", flipped, compared);
  require(flipped * 5 <= compared, "chunked prefill changed the MTP head's proposals");
  const std::vector<Run> batch = drafting.generateTogether({prompt, shorter}, 16);
  const Run single = drafting.generate(shorter, 16);
  require(batch[0].tokens == std::vector<uint32_t>(mtp.tokens.begin(), mtp.tokens.begin() + 16) &&
              batch[1].tokens == single.tokens,
          "a batch of two changed the MTP decode's tokens");
  checkStaging(spy.calls, {mtp, chunked, batch[0], batch[1], single}, table, wholeTable, 8);
  const auto proposalsOf = [&](uint64_t id) {
    std::vector<std::vector<uint32_t>> proposals;
    for (const SpyFamily::Call &call : spy.calls)
      if (call.request == id) proposals.push_back(call.tokens);
    return proposals;
  };
  for (const auto &[left, right] : {std::pair{batch[0].id, mtp.id}, std::pair{batch[1].id, single.id}}) {
    const auto a = proposalsOf(left), b = proposalsOf(right);
    for (size_t cycle = 0; cycle + 1 < std::min(a.size(), b.size()); ++cycle)
      require(a[cycle] == b[cycle], "a batch of two changed the MTP head's proposals");
  }
  std::printf("MTP: chunked prefill and a batch of two propose what the one request does\n");

  // A long enough match proposes in place of the head: those cycles skip the draft command and verify the lookup's
  // rows, the others keep the head's; the tokens are the draft-less decode's.
  setenv("SPLASH_LOOKUP_MIN", "2", 1);
  const size_t hybridBefore = spy.calls.size();
  const Run hybrid = drafting.generate(prompt, 40, 8);
  const size_t lookups = std::count_if(hybrid.cycles.begin(), hybrid.cycles.end(),
                                       [](const Cycle &c) { return c.drafted != steps; });
  std::printf("MTP + lookup: %zu cycles, %zu lookup cycles\n", hybrid.cycles.size(), lookups);
  require(lookups > 0 && lookups < hybrid.cycles.size(), "the hybrid decode did not mix lookup and MTP cycles");
  require(spy.calls.size() - hybridBefore == hybrid.cycles.size(), "a hybrid cycle staged no verify rows");
  sameAsDraftless(hybrid, "MTP + lookup decode");
  checkStaging(spy.calls, {hybrid}, table, wholeTable, 8);
  setenv("SPLASH_LOOKUP_MIN", "0", 1);

  // The engine polls the ticket instead of waiting for it: the verify is staged and submitted from the poll.
  drafting.pollTickets(true);
  const Run polled = drafting.generate(prompt, 40, 8);
  require(polled.tokens == mtp.tokens && polled.drafted == mtp.drafted && polled.accepted == mtp.accepted,
          "a polled MTP ticket decodes other tokens than a waited one");
  std::printf("polled ticket == waited ticket\n");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::fprintf(stderr, "usage: qwen4-runtime METALLIB\n");
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::fprintf(stderr, "qwen4 runtime test failed: %s\n", error.what());
      return 1;
    }
    std::printf("qwen4 runtime tests passed\n");
    return 0;
  }
}
