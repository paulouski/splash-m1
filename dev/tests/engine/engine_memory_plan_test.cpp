#include "engine/MemoryPlan.hpp"
#include "TestModel.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

DeviceCapabilities device(uint64_t workingSet = 12 * kGiB) {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = workingSet;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  result.supportsPlacementSparse = true;
  return result;
}

ModelMemoryProfile model() {
  return test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
}

void testUnifiedElasticBudget() {
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.maximumBatchWidth == 4 &&
              budget.kvSparseMappingBatchPages == 128 &&
              budget.kvExtentPages == 128,
          "execution geometry did not reach memory planning");
  require(budget.fixedRuntimeBytes == model().fixedRuntimeBytes(),
          "active state was incorrectly precharged as fixed memory");
  require(budget.dynamicBudgetBytes ==
              budget.hardBudgetBytes - budget.fixedRuntimeBytes,
          "state and KV do not share one dynamic budget");
  require(budget.minimumDynamicBytes ==
                  budget.activeStateCellBytes + budget.kvExtentBytes &&
              budget.minimumRequiredBytes ==
                  budget.fixedRuntimeBytes + budget.minimumDynamicBytes,
          "minimum B1 plus one physical extent is incorrect");
  require(budget.kvVirtualPages % 128 == 0 && budget.kvVirtualPages >= 128 &&
              budget.kvVirtualBytes ==
                  uint64_t{budget.kvVirtualPages} * budget.kvPageBytes &&
              budget.kvVirtualTokens == uint64_t{budget.kvVirtualPages} * 32,
          "KV virtual address space is inconsistent");
  require(plan.maximumContextTokens() ==
              std::min<uint64_t>(model().maximumContextTokens,
                                 budget.kvVirtualTokens -
                                     model::ExecutionLimits::speculativeScratchTokens),
          "advertised context exceeds elastic KV capacity");
  const std::string json = plan.toStatusJson();
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":128") != std::string::npos,
          "elastic state/KV budget is missing from memory status");
}

void testBf16BudgetAndStatus() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  const auto bf16 = requireEngineMemoryPlan(device(), profile);
  const auto int8 = requireEngineMemoryPlan(device(), model());
  const auto &budget = bf16.breakdown();
  require(budget.kvPageBytes == profile.targetKvLayout.bytesPerModelPage() &&
              budget.kvPageBytes > int8.breakdown().kvPageBytes &&
              budget.kvSparseMappingBatchPages == 1 && budget.kvExtentPages == 64,
          "BF16 planning did not use its payload size and sparse alignment");
  require(budget.kvVirtualBytes <= budget.dynamicBudgetBytes &&
              budget.kvVirtualPages < int8.breakdown().kvVirtualPages,
          "BF16 virtual capacity exceeded the shared budget");
  const auto json = bf16.toStatusJson();
  require(json.find("\"kv_format\":\"bf16\"") != std::string::npos &&
              json.find("\"kv_scale_value_bytes\":0") != std::string::npos &&
              json.find("\"q8_page_bytes\"") == std::string::npos,
          "BF16 memory status reported INT8 scales or pages");
  const uint64_t minimum = budget.minimumRequiredBytes;
  require(!evaluateEngineMemoryPlan(device(), profile, minimum - 1).plan,
          "BF16 startup admitted less than its minimum resident footprint");
}

void testUserCeilingAndFailure() {
  EngineMemoryPlan automatic = requireEngineMemoryPlan(device(), model());
  const uint64_t ceiling =
      automatic.breakdown().minimumRequiredBytes + 64 * kMiB;
  EngineMemoryPlan limited =
      requireEngineMemoryPlan(device(), model(), ceiling);
  require(limited.breakdown().hardBudgetBytes == ceiling,
          "explicit memory ceiling was ignored");
  require(requireEngineMemoryPlan(device(), model(), 16 * kGiB)
                  .breakdown().hardBudgetBytes ==
              automatic.breakdown().hardBudgetBytes,
          "explicit memory ceiling overrode the safe working set");

  auto failed = evaluateEngineMemoryPlan(
      device(), model(), automatic.breakdown().minimumRequiredBytes - 1);
  require(!failed.plan &&
              failed.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "budget smaller than B1 plus one extent was accepted");
}

// The disk tier's KV pages stage through a ring of Metal memory, which the
// governor charges beside the weights. The plan sets it aside, so one lone
// request can still map every KV page the advertised context promises.
void testDiskTierKvStagingIsBudgeted() {
  const EngineMemoryPlan without = requireEngineMemoryPlan(device(), model());
  ModelMemoryProfile tiered = model();
  // 128 16 KiB-aligned page slots plus the copy table rounded up to 16 KiB.
  const uint64_t ring =
      128 * tiered.targetKvLayout.bytesPerModelPage() + 16 * 1024;
  tiered.footprint.kvStagingBytes = ring;
  const EngineMemoryPlan with = requireEngineMemoryPlan(device(), tiered);
  const auto &budget = with.breakdown();
  require(without.breakdown().fixedRuntimeBytes + ring +
                  budget.activeStateCellBytes + budget.kvVirtualBytes <=
              budget.hardBudgetBytes,
          "the advertised context cannot be mapped beside the KV staging ring");
  require(with.maximumContextTokens() < without.maximumContextTokens(),
          "a budget-limited context did not shrink by the KV staging ring");
  require(budget.kvStagingBytes == ring &&
              budget.fixedRuntimeBytes ==
                  without.breakdown().fixedRuntimeBytes + ring,
          "KV staging was not planned as fixed runtime memory");
  // A budget that fits everything but the ring is refused by the plan, not
  // by a warmup allocation.
  const auto tight = evaluateEngineMemoryPlan(
      device(), tiered, without.breakdown().minimumRequiredBytes);
  require(!tight.plan &&
              tight.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget without room for the KV staging ring was accepted");
  const std::string staging = "\"kv_staging_bytes\":" + std::to_string(ring);
  require(with.toStatusJson().find(staging + ",\"fixed_runtime_bytes\"") !=
                  std::string::npos &&
              with.toStatusJson().find(staging + "}}") != std::string::npos &&
              budget.describe().find("disk tier KV staging: " +
                                     std::to_string(ring)) != std::string::npos,
          "KV staging is missing from the memory plan status");
  require(without.breakdown().kvStagingBytes == 0 &&
              without.toStatusJson().find("\"kv_staging_bytes\":0,") !=
                  std::string::npos,
          "a plan without the disk tier reported KV staging");
}

void testHardBudgetBoundaries() {
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(0) == 0 &&
              EngineMemoryPolicy::hardBudgetBytes(kGiB, 1) == 0,
          "insufficient working set underflowed the preflight ceiling");
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(EngineMemoryPolicy::hardBudgetBytes(maximum, maximum) ==
              maximum - EngineMemoryPolicy::workingSetMarginBytes(maximum),
          "maximum working set overflowed the preflight ceiling");
}

// What memory holds below the plan's budget, where the host has less: the
// plan made there, within the configured limit, and nothing where one request
// does not fit.
void testContextTokensWithin() {
  const EngineMemoryPlan plan = requireEngineMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  const uint64_t ceiling = budget.minimumRequiredBytes + 64 * kMiB;
  const EngineMemoryPlan limited = requireEngineMemoryPlan(device(), model(), ceiling);
  require(plan.contextTokensWithin(16 * kGiB) == plan.maximumContextTokens() &&
              plan.contextTokensWithin(ceiling) == limited.maximumContextTokens() &&
              limited.maximumContextTokens() < plan.maximumContextTokens() &&
              limited.contextTokensWithin(16 * kGiB) == limited.maximumContextTokens() &&
              !plan.contextTokensWithin(budget.minimumRequiredBytes - 1) &&
              !plan.contextTokensWithin(0),
          "the context memory holds is not the plan's within the host's memory");
}

void testModelProvidedKvGeometry() {
  ModelMemoryProfile compact = model();
  compact.name = "compact-test-model";
  compact.targetKvLayout = {10, 2, 256};
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), compact);
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.kvPageBytes == 332'800 &&
              budget.kvSparseMappingBatchPages == 256 &&
              budget.kvExtentPages == 512,
          "memory plan ignored model-provided Q8 geometry");
  require(plan.toStatusJson().find("\"attention_layers\":10") !=
              std::string::npos &&
              plan.toStatusJson().find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

// Ternary-Bonsai-2-27B with its DFlash draft: prepared target 7,251,165,184 B,
// target + draft 8,517,206,016 B. The prefill arena is the 2048-row figure;
// smaller rungs are scaled estimates (2% over linear) until a GPU run
// reports the real arena.
ModelMemoryProfile bonsai() {
  ModelMemoryProfile result{
      "Ternary-Bonsai-2-27B", kv::kMaximumLogicalTokens, 0,
      kv::Layout{16, 4, 256},
      {7'251'165'184, 1'266'040'832, 0, 350'224'384, 785'580'032,
       262'662'272, 256 * kMiB, 512 * kMiB}};
  result.footprint.cachedStateBytes = 196'083'712;
  return result;
}

uint64_t bonsaiPrefillBytes(uint32_t rows) {
  return 785'580'032ULL * rows / 2048 * 102 / 100;
}

DeviceCapabilities device16() {
  DeviceCapabilities result = device(11'453'251'584);
  result.physicalMemoryBytes = 17'179'869'184;
  return result;
}

#if defined(SPLASH_MACOS15_BUILD)
// One lane of the shared decode arena (model_execution_plan_test pins it).
uint64_t bonsaiDecodeBytes(uint32_t lanes) {
  return lanes == 1 ? 66'042'400ULL : 262'662'272ULL;
}

DeviceCapabilities device16NonSparse() {
  DeviceCapabilities result = device16();
  result.appleGpuFamily = 7;
  result.macosMajor = 15;
  result.macosMinor = 7;
  result.supportsPlacementSparse = false;
  return result;
}
#endif

void requireSameBreakdown(const EngineMemoryBreakdown &a,
                          const EngineMemoryBreakdown &b, const char *message) {
  require(a.hardBudgetBytes == b.hardBudgetBytes &&
              a.sharedPrefillBytes == b.sharedPrefillBytes &&
              a.sharedDecodeBytes == b.sharedDecodeBytes &&
              a.prefillRows == b.prefillRows &&
              a.maximumBatchWidth == b.maximumBatchWidth &&
              a.fixedRuntimeBytes == b.fixedRuntimeBytes &&
              a.dynamicBudgetBytes == b.dynamicBudgetBytes &&
              a.kvVirtualPages == b.kvVirtualPages,
          message);
}

void testAdaptivePrefillKeepsFittingPlansUnchanged() {
  const auto neverCalled = [](uint32_t) -> uint64_t {
    throw std::runtime_error("a plan that fits was resized");
  };
  // 32 GB M1 Max (~24 GiB automatic budget, 22 GiB cap) with Qwen Q4/Q5.
  DeviceCapabilities max = device(26'800'000'000ULL);
  max.physicalMemoryBytes = 32 * kGiB;
  for (const uint64_t target : {15'000'000'000ULL, 18'000'000'000ULL}) {
    const ModelMemoryProfile profile =
        test::modelMemoryProfile(target, 1'266'040'832, 0);
    for (const uint64_t cap : {uint64_t{0}, 22 * kGiB}) {
      const auto direct = requireEngineMemoryPlan(max, profile, cap);
      const auto adaptive =
          evaluateAdaptiveMemoryPlan(max, profile, cap, neverCalled);
      require(adaptive.plan && adaptive.plan->breakdown().prefillRows ==
                                   model::ExecutionLimits::prefillTokenBudget,
              "a fitting plan changed its prefill rows");
      requireSameBreakdown(adaptive.plan->breakdown(), direct.breakdown(),
                           "a fitting plan differs from the direct plan");
    }
  }
}

void testAdaptivePrefillFitsBonsaiOn16Gb() {
  const auto direct = evaluateEngineMemoryPlan(device16(), bonsai());
  require(!direct.plan &&
              direct.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              direct.status.breakdown.hardBudgetBytes == 10'379'509'760ULL &&
              direct.status.breakdown.deficitBytes == 477'784'192ULL,
          "Bonsai no longer misses the 16 GB budget by 477,784,192 bytes");
  const auto adaptive = evaluateAdaptiveMemoryPlan(
      device16(), bonsai(), 0, bonsaiPrefillBytes);
  require(adaptive.plan.has_value(), "Bonsai with its draft does not fit 16 GB");
  const auto &budget = adaptive.plan->breakdown();
  require(budget.prefillRows == 128 &&
              budget.sharedPrefillBytes == bonsaiPrefillBytes(128) &&
              budget.sharedDecodeBytes == 262'662'272ULL &&
              budget.minimumRequiredBytes <= budget.hardBudgetBytes,
          "16 GB Bonsai did not take the smallest prefill rung");
  // Sparse mappings stay whole 128-page batches (64 KiB scale tiles), so the
  // 369 pages that fit floor to 256 (8,192 tokens less the scratch rows).
  require(budget.kvVirtualPages == 256 &&
              adaptive.plan->maximumContextTokens() == 8185,
          "16 GB sparse Bonsai KV pool changed");
}

#if defined(SPLASH_MACOS15_BUILD)
// macOS 15 has no placement sparse: the pool is plain buffers, committed whole.
// The 16 GB Bonsai plan (rungs after 2048 are estimates: 2% over linear):
//   hard budget 10,379,509,760 = fixed 9,635,255,383 + dynamic 744,254,377
//   fixed = weights 8,517,206,016 + prefill 50,080,727 (128 rows) + decode
//           262,662,272 + pipelines 256 MiB + overhead 512 MiB
//   dynamic = one state cell 350,224,384 + one cached state 196,083,712
//             + 184 KV pages x 1,064,960 B
//   184 pages = 5,888 tokens, 5,881 after the 7 speculative scratch rows.
//   Without the cached-state reserve: 369 pages, 11,801 tokens.
void testBonsaiOn16GbMacos15() {
  const DeviceCapabilities dense = device16NonSparse();
  ModelMemoryProfile requested = bonsai();
  requested.requestedContextTokens = 16384;
  for (const ModelMemoryProfile &profile : {bonsai(), requested}) {
    const auto plan = evaluateAdaptiveMemoryPlan(dense, profile, 0,
                                                 bonsaiPrefillBytes);
    require(plan.plan.has_value(), "16 GB macOS 15 Bonsai does not fit");
    const auto &budget = plan.plan->breakdown();
    require(budget.prefillRows == 128 &&
                budget.sharedPrefillBytes == 50'080'727ULL &&
                budget.fixedRuntimeBytes == 9'635'255'383ULL &&
                budget.dynamicBudgetBytes == 744'254'377ULL &&
                budget.kvVirtualPages == 184 &&
                plan.plan->maximumContextTokens() == 5'881,
            "16 GB macOS 15 Bonsai plan changed");
  }
  // Each larger rung fits only fewer pages: 512 rows hold 228, 256 rows 322.
  for (const auto &[rows, pages] : {std::pair{512u, 228u}, std::pair{256u, 322u}}) {
    ModelMemoryProfile rung = bonsai();
    rung.footprint.prefillRows = rows;
    rung.footprint.sharedPrefillBytes = bonsaiPrefillBytes(rows);
    rung.exactKvPages = true;
    const auto result = evaluateEngineMemoryPlan(dense, rung);
    require(result.plan && result.plan->breakdown().kvVirtualPages == pages,
            "a Bonsai prefill rung holds a different KV page count");
  }
  // 16K int8 KV needs 513 pages; 144 pages (153,354,240 B) are missing.
  // Decode buffers of one lane (-188 MiB) alone would cover it; INT8 draft
  // codebooks (-125,153,280 B) alone would not (487 pages, 15,577 tokens).
  const auto holds = [&](uint64_t decodeLess, uint64_t draftLess) {
    ModelMemoryProfile profile = bonsai();
    profile.footprint.cachedStateBytes = 0;
    profile.footprint.sharedDecodeBytes -= decodeLess;
    profile.footprint.draftWeightsBytes -= draftLess;
    return evaluateAdaptiveMemoryPlan(dense, profile, 0, bonsaiPrefillBytes)
        .plan->maximumContextTokens();
  };
  require(holds(188 * kMiB, 0) >= 16384 && holds(0, 125'153'280ULL) == 15'577,
          "the remaining 16K levers changed");
}

// With one decode lane the 16 GB macOS 15 ladder keeps one snapshot slot and
// ~11.8K tokens (16K would leave no room for cached states). Rows
// outrank lanes, so each chunk size tries 4 lanes before 1, and the first rung
// that reaches the goal wins.
void testBonsaiOn16GbReaches16kWithOneLane() {
  const DeviceCapabilities dense = device16NonSparse();
  const auto plan = evaluateAdaptiveMemoryPlan(
      dense, bonsai(), 0, bonsaiPrefillBytes, bonsaiDecodeBytes);
  require(plan.plan.has_value(), "16 GB macOS 15 Bonsai does not fit");
  const auto &budget = plan.plan->breakdown();
  std::cout << "16 GB Bonsai macOS 15: " << budget.prefillRows << " rows, "
            << budget.maximumBatchWidth << " lane(s), " << budget.kvVirtualPages
            << " KV pages, " << plan.plan->maximumContextTokens()
            << " tokens\n";
  require(budget.prefillRows == 128 && budget.maximumBatchWidth == 1 &&
              budget.sharedDecodeBytes == bonsaiDecodeBytes(1) &&
              budget.kvVirtualPages == 369 &&
              plan.plan->maximumContextTokens() == 11'801,
          "16 GB macOS 15 Bonsai did not keep a snapshot slot with one lane");
  // Without the lane rungs the plan is the 4-lane one of the test above.
  const auto fourLanes =
      evaluateAdaptiveMemoryPlan(dense, bonsai(), 0, bonsaiPrefillBytes);
  require(fourLanes.plan &&
              fourLanes.plan->breakdown().maximumBatchWidth ==
                  model::ExecutionLimits::maximumBatchWidth &&
              fourLanes.plan->maximumContextTokens() == 5'881,
          "the lane rung changed the plan that does not ask for it");
}

// A plan that fits keeps all lanes and the model's prefill chunk.
void testFittingPlanKeepsAllLanes() {
  DeviceCapabilities max = device(26'800'000'000ULL);
  max.physicalMemoryBytes = 32 * kGiB;
  const auto neverCalled = [](uint32_t) -> uint64_t {
    throw std::runtime_error("a plan that fits was resized");
  };
  const auto plan = evaluateAdaptiveMemoryPlan(
      max, test::modelMemoryProfile(15'000'000'000ULL, 1'266'040'832, 0), 0,
      neverCalled, neverCalled);
  require(plan.plan &&
              plan.plan->breakdown().prefillRows ==
                  model::ExecutionLimits::prefillTokenBudget &&
              plan.plan->breakdown().maximumBatchWidth ==
                  model::ExecutionLimits::maximumBatchWidth,
          "a fitting plan lost lanes or prefill rows");
}

// The default plan of a 32 GB macOS 15 Mac keeps its 128-page KV rounding.
void testDefaultPlanKeepsKvRounding() {
  DeviceCapabilities max = device(26'800'000'000ULL);
  max.physicalMemoryBytes = 32 * kGiB;
  max.appleGpuFamily = 7;
  max.supportsPlacementSparse = false;
  const auto neverCalled = [](uint32_t) -> uint64_t {
    throw std::runtime_error("a plan that fits was resized");
  };
  ModelMemoryProfile profile = test::modelMemoryProfile(15'000'000'000ULL,
                                                        1'266'040'832, 0);
  profile.requestedContextTokens = 16384;
  const auto plan = evaluateAdaptiveMemoryPlan(max, profile, 0, neverCalled);
  require(plan.plan && plan.plan->breakdown().kvVirtualPages == 640 &&
              plan.plan->breakdown().prefillRows ==
                  model::ExecutionLimits::prefillTokenBudget,
          "the default macOS 15 plan changed its KV pool rounding");
}
#endif

// A 32 GB Mac capped at the 16 GB budget plans the same arenas.
void testAdaptivePrefillFollowsTheCapNotPhysicalMemory() {
  DeviceCapabilities max = device(26'800'000'000ULL);
  max.physicalMemoryBytes = 32 * kGiB;
  const auto capped = evaluateAdaptiveMemoryPlan(
      max, bonsai(), 10'379'509'760ULL, bonsaiPrefillBytes);
  const auto small = evaluateAdaptiveMemoryPlan(
      device16(), bonsai(), 0, bonsaiPrefillBytes);
  require(capped.plan && small.plan,
          "capped 32 GB or 16 GB Bonsai plan does not fit");
  requireSameBreakdown(capped.plan->breakdown(), small.plan->breakdown(),
                       "the capped 32 GB plan differs from the 16 GB plan");
}

void testAdaptivePrefillNamesTheRungsWhenNothingFits() {
  const auto result = evaluateAdaptiveMemoryPlan(
      device16(), bonsai(), 9 * kGiB, bonsaiPrefillBytes);
  require(!result.plan &&
              result.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              result.status.message.find("2048, 1024, 512, 256, 128 rows") !=
                  std::string::npos &&
              result.status.breakdown.prefillRows == 128,
          "an unfittable plan did not report the prefill rungs it tried");
}

} // namespace

#if !defined(SPLASH_MACOS15_BUILD)
void testDeviceValidationNamesTheMacosFloor() {
  require(!device().validationError(),
          "the reference device reported a validation error");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "macOS 26.3 was not refused with the macOS reason");
  // The operating system explains a missing placement-sparse query, so its
  // reason takes precedence over the feature's own.
  older.supportsPlacementSparse = false;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "an older macOS did not take precedence over the sparse reason");
  require(older.macosVersion() == "26.3.0",
          "the macOS version string is not major.minor.patch");
  DeviceCapabilities unknown = device();
  unknown.macosMajor = 0;
  unknown.macosMinor = 0;
  require(unknown.validationError().value_or("") == "macos_26_4_required",
          "an unknown macOS version was accepted");
  DeviceCapabilities newer = device();
  newer.macosMajor = 27;
  newer.macosMinor = 0;
  require(!newer.validationError(), "a newer macOS major was refused");
}

void testDeviceValidationMessageNamesWhatTheMacHas() {
  require(!device().validationMessage(),
          "the reference device has a validation message");
  const std::string needs =
      "Splash needs Apple GPU family 7 or newer (M1 or later) on macOS 26.4 "
      "or newer, with placement-sparse buffers; this Mac has ";
  DeviceCapabilities a13 = device();
  a13.deviceName = "Apple A13";
  a13.appleGpuFamily = 6;
  a13.macosPatch = 1;
  require(a13.validationMessage().value_or("") ==
              needs + "Apple A13 (Apple GPU family 6) on macOS 26.4.1, "
                      "with placement-sparse buffers "
                      "(apple_gpu_family_7_required)",
          "a family-6 GPU was not named against the family required");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  older.supportsPlacementSparse = false;
  require(older.validationMessage().value_or("") ==
              needs + "test (Apple GPU family 9) on macOS 26.3.0, where "
                      "placement-sparse support cannot be queried "
                      "(macos_26_4_required)",
          "an older macOS was not named against the macOS required");
  DeviceCapabilities dense = device();
  dense.supportsPlacementSparse = false;
  require(dense.validationMessage().value_or("") ==
              needs + "test (Apple GPU family 9) on macOS 26.4.0, without "
                      "placement-sparse buffers (placement_sparse_required)",
          "missing placement-sparse buffers were not named");
}
#endif

int main() {
  try {
    testUnifiedElasticBudget();
    testBf16BudgetAndStatus();
    testUserCeilingAndFailure();
    testDiskTierKvStagingIsBudgeted();
    testHardBudgetBoundaries();
    testContextTokensWithin();
    testModelProvidedKvGeometry();
    testAdaptivePrefillKeepsFittingPlansUnchanged();
    testAdaptivePrefillFitsBonsaiOn16Gb();
    testAdaptivePrefillFollowsTheCapNotPhysicalMemory();
#if defined(SPLASH_MACOS15_BUILD)
    testBonsaiOn16GbMacos15();
    testBonsaiOn16GbReaches16kWithOneLane();
    testFittingPlanKeepsAllLanes();
    testDefaultPlanKeepsKvRounding();
#endif
    testAdaptivePrefillNamesTheRungsWhenNothingFits();
#if !defined(SPLASH_MACOS15_BUILD)
    testDeviceValidationNamesTheMacosFloor();
    testDeviceValidationMessageNamesWhatTheMacHas();
#endif
    std::cout << "elastic memory plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
