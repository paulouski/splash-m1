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
  std::vector<uint8_t> bytes(table.rows * pleRowBytes);
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
                             const std::vector<uint32_t> &slots) {
  metal::MetalBuffer indices = sharedBuffer(backend, slots.size() * 4, "ple-test-indices");
  std::memcpy(indices.contents(), slots.data(), slots.size() * 4);
  metal::MetalBuffer output = sharedBuffer(backend, slots.size() * 160 * 2, "ple-test-output");
  metal::CommandGraph graph;
  ops::Embedding::add(graph, indices, ops::EmbeddingWeights(tableRows, 160, ops::NativeRows(rows, GGUF_FMT_IQ4NL)),
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
  PleStore store(PleSource(table.path.string(), table.dataOffset, table.rows), 64 << 10);
  for (const uint64_t cache : {0, 1}) {
    static_cast<void>(cache);
    metal::MetalBuffer staging = sharedBuffer(backend, ids.size() * pleRowBytes, "ple-test-staging");
    std::vector<uint32_t> slots(ids.size());
    // The second pass answers from the row cache.
    const PleStore::Result result =
        store.fetch(ids, {static_cast<uint8_t *>(staging.contents()), ids.size() * pleRowBytes}, slots);
    require(result.ok(), "the PLE store failed to fetch: " + result.error);
    require(cache == 0 ? result.stats.misses > 0 : result.stats.hits > 0 && result.stats.misses == 0,
            "the second fetch did not come from the cache");
    require(result.stats.dupes == 3, "the repeated ids were not deduplicated");
    const auto viaStaging = gather(backend, staging, static_cast<uint32_t>(ids.size()), slots);
    const auto viaTable = gather(backend, whole, static_cast<uint32_t>(table.rows), ids);
    require(viaStaging == viaTable, "gathering from the staging differs from gathering from the whole table");
    require(std::any_of(viaTable.begin(), viaTable.end(), [](uint16_t v) { return v != 0; }),
            "the gathered PLE rows are all zero");
  }
  std::printf("ple staging gather == full-table gather: ok\n");
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

struct Run final {
  std::vector<uint32_t> tokens;
  std::vector<ops::TokenLogprobs> logprobs;
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
  Run generate(const std::vector<uint32_t> &prompt, uint32_t count, uint32_t logprobs = 0) {
    return generateTogether({prompt}, count, logprobs).front();
  }

  // The same for several prompts at once, decoded in one batch of that width.
  std::vector<Run> generateTogether(const std::vector<std::vector<uint32_t>> &prompts, uint32_t count,
                                    uint32_t logprobs = 0) {
    std::vector<ModelRequest> requests(prompts.size());
    std::vector<Run> runs(prompts.size());
    std::vector<uint64_t> positions;
    for (size_t i = 0; i < prompts.size(); ++i) {
      requests[i].id = ++nextId_;
      requests[i].prompt = prompts[i];
      requests[i].maxNewTokens = count;
      requests[i].logprobs = logprobs;
      begin(requests[i]);
      const ModelStepResult first = prefill(requests[i].id, prompts[i], 0, static_cast<uint32_t>(prompts[i].size()));
      require(first.failure.empty() && first.outputTokens.empty(), "the prompt's prefill did not leave an anchor");
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
      auto results = runtime_->decode(plan, items);
      require(results.size() == prompts.size(), "a decode batch lost a lane");
      for (size_t i = 0; i < prompts.size(); ++i) {
        require(results[i].failure.empty() && !results[i].outputTokens.empty(),
                "a decode cycle failed: " + results[i].failure);
        Run &run = runs[i];
        run.tokens.insert(run.tokens.end(), results[i].outputTokens.begin(), results[i].outputTokens.end());
        run.logprobs.insert(run.logprobs.end(), results[i].outputLogprobs.begin(), results[i].outputLogprobs.end());
        run.drafted += results[i].draftedTokens;
        run.accepted += results[i].acceptedDraftTokens;
        positions[i] += results[i].outputTokens.size() - results[i].outputTokensWithoutKv;
      }
    }
    for (size_t i = 0; i < prompts.size(); ++i) {
      runtime_->end(requests[i].id);
      runs[i].tokens.resize(count);
    }
    return runs;
  }

private:
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
