#include "TestChecks.hpp"
#include "engine/MemoryPlan.hpp"
#include "TestModel.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

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
  return result;
}

ModelMemoryProfile model() {
  return test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
}

// The pages the budget holds for one request's KV beside one lane's state.
uint64_t budgetPages(const EngineMemoryBreakdown &budget) {
  return (budget.dynamicBudgetBytes - budget.laneStateBytes) / budget.kvPageBytes;
}

void testUnifiedElasticBudget() {
  EngineMemoryPlan plan = test::requireMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.maximumBatchWidth == 4 &&
              budget.kvExtentPages ==
                  model().targetKvLayout.extentPagesFor(budgetPages(budget)),
          "execution geometry did not reach memory planning");
  require(budget.fixedRuntimeBytes == model().fixedRuntimeBytes(),
          "active state was incorrectly precharged as fixed memory");
  require(budget.dynamicBudgetBytes ==
              budget.hardBudgetBytes - budget.fixedRuntimeBytes,
          "state and KV do not share one dynamic budget");
  require(budget.minimumDynamicBytes ==
                  budget.laneStateBytes +
                      kvRunwayPages(model().targetKvLayout.minimumExtentPages()) *
                          budget.kvPageBytes &&
              budget.minimumRequiredBytes ==
                  budget.fixedRuntimeBytes + budget.minimumDynamicBytes,
          "minimum B1 plus the KV runway is incorrect");
  require(budget.kvCapacityPages % 128 == 0 && budget.kvCapacityPages >= 128 &&
              budget.kvCapacityPages == budgetPages(budget) - budgetPages(budget) % 128 &&
              budget.kvCapacityBytes ==
                  uint64_t{budget.kvCapacityPages} * budget.kvPageBytes &&
              budget.kvCapacityTokens == uint64_t{budget.kvCapacityPages} * 32,
          "one request's KV capacity is not the budget's whole extents");
  require(plan.maximumContextTokens() ==
              std::min<uint64_t>(model().maximumContextTokens,
                                 budget.kvCapacityTokens -
                                     model::ExecutionLimits::speculativeScratchTokens),
          "advertised context exceeds elastic KV capacity");
  const std::string json = plan.toStatusJson();
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":" +
                        std::to_string(budget.kvExtentPages)) !=
                  std::string::npos,
          "elastic state/KV budget is missing from memory status");
  require(json.find("\"working_set_margin_bytes\":" +
                    std::to_string(budget.workingSetMarginBytes)) !=
                  std::string::npos &&
              json.find("\"headroom_bytes\"") == std::string::npos,
          "memory status did not name the working-set margin");
}

void testBf16BudgetAndStatus() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  const auto bf16 = test::requireMemoryPlan(device(), profile);
  const auto int8 = test::requireMemoryPlan(device(), model());
  const auto &budget = bf16.breakdown();
  require(budget.kvPageBytes == profile.targetKvLayout.bytesPerModelPage() &&
              budget.kvPageBytes > int8.breakdown().kvPageBytes &&
              budget.kvExtentPages ==
                  profile.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvCapacityPages % budget.kvExtentPages == 0,
          "BF16 planning did not use its payload size and its pool's extent size");
  require(budget.kvCapacityBytes <= budget.dynamicBudgetBytes &&
              budget.kvCapacityPages < int8.breakdown().kvCapacityPages,
          "BF16 KV capacity exceeded the shared budget");
  const auto json = bf16.toStatusJson();
  require(json.find("\"kv_format\":\"bf16\"") != std::string::npos &&
              json.find("\"kv_scale_value_bytes\"") == std::string::npos &&
              json.find("\"kv_quantization_bits\"") == std::string::npos,
          "BF16 memory status restated its format");
  const uint64_t minimum = budget.minimumRequiredBytes;
  require(!evaluateEngineMemoryPlan(device(), profile, minimum - 1).plan,
          "BF16 startup admitted less than its minimum footprint");
}

// The minimum holds the KV runway, the whole smallest extents that hold the
// pages startup warmup runs on: two of the 27B's 32-page BF16 extents.
void testMinimumHoldsTheWarmupRunway() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  require(profile.targetKvLayout.minimumExtentPages() == 32,
          "the BF16 fixture's smallest extent changed");
  const EngineMemoryPlan plan = test::requireMemoryPlan(device(), profile);
  const auto &budget = plan.breakdown();
  require(budget.minimumDynamicBytes ==
              budget.laneStateBytes + 64 * budget.kvPageBytes,
          "the minimum does not hold the warmup runway");
  const auto refused =
      evaluateEngineMemoryPlan(device(), profile, budget.minimumRequiredBytes - 1);
  require(!refused.plan && refused.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget short of the warmup runway was accepted");
}

// The pre-load fit check and the plan share one minimum: the fixed bytes,
// one lane's state and the KV runway.
void testMinimumRequiredBytesIsThePlans() {
  for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
    ModelMemoryProfile profile = model();
    profile.targetKvLayout.format = format;
    const EngineMemoryPlan plan = test::requireMemoryPlan(device(), profile);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    require(minimumRequiredBytes(budget.fixedRuntimeBytes, budget.laneStateBytes,
                                 profile.targetKvLayout) == budget.minimumRequiredBytes,
            "the minimum differs from the plan's");
  }
  require(!minimumRequiredBytes(std::numeric_limits<uint64_t>::max(), 1,
                                model().targetKvLayout),
          "an overflowing minimum was not refused");
}

void testUserCeilingAndFailure() {
  EngineMemoryPlan automatic = test::requireMemoryPlan(device(), model());
  const uint64_t ceiling =
      automatic.breakdown().minimumRequiredBytes + 64 * kMiB;
  EngineMemoryPlan limited =
      test::requireMemoryPlan(device(), model(), ceiling);
  require(limited.breakdown().hardBudgetBytes == ceiling,
          "explicit memory ceiling was ignored");
  require(test::requireMemoryPlan(device(), model(), 16 * kGiB)
                  .breakdown().hardBudgetBytes ==
              automatic.breakdown().hardBudgetBytes,
          "explicit memory ceiling overrode the safe working set");

  auto failed = evaluateEngineMemoryPlan(
      device(), model(), automatic.breakdown().minimumRequiredBytes - 1);
  require(!failed.plan &&
              failed.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "budget smaller than B1 plus one extent was accepted");
}

// A state's write to the disk tier stages through one state-sized buffer the
// plan sets aside beside the weights, so one lone request still reaches the
// advertised context.
void testDiskTierStateStagingIsBudgeted() {
  const EngineMemoryPlan without = test::requireMemoryPlan(device(), model());
  ModelMemoryProfile tiered = model();
  // One Qwen3.8-27B state (DEVELOPMENT.md, Disk cache).
  const uint64_t staging = 187 * kMiB;
  tiered.footprint.stateStagingBytes = staging;
  const EngineMemoryPlan with = test::requireMemoryPlan(device(), tiered);
  const auto &budget = with.breakdown();
  require(without.breakdown().fixedRuntimeBytes + staging +
                  budget.laneStateBytes + budget.kvCapacityBytes <=
              budget.hardBudgetBytes,
          "the advertised context cannot be allocated beside the state staging buffer");
  require(with.maximumContextTokens() < without.maximumContextTokens(),
          "a budget-limited context did not shrink by the state staging buffer");
  require(budget.stateStagingBytes == staging &&
              budget.fixedRuntimeBytes ==
                  without.breakdown().fixedRuntimeBytes + staging,
          "state staging was not planned as fixed runtime memory");
  // A budget that fits everything but the staging buffer is refused by the
  // plan, not by a warmup allocation.
  const auto tight = evaluateEngineMemoryPlan(
      device(), tiered, without.breakdown().minimumRequiredBytes);
  require(!tight.plan &&
              tight.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget without room for the state staging buffer was accepted");
  const std::string field = "\"state_staging_bytes\":" + std::to_string(staging);
  require(with.toStatusJson().find(field + ",\"fixed_runtime_bytes\"") !=
                  std::string::npos &&
              with.toStatusJson().find(field + "}}") != std::string::npos &&
              budget.describe().find("disk tier state staging: " +
                                     std::to_string(staging)) != std::string::npos,
          "state staging is missing from the memory plan status");
  require(without.breakdown().stateStagingBytes == 0 &&
              without.toStatusJson().find("\"state_staging_bytes\":0,") !=
                  std::string::npos,
          "a plan without the disk tier reported state staging");
}

// The pipeline and runtime reserves are the model constants rather than part
// of a model's plan, and a model's plan without one of its arenas is refused.
void testReservesAreTheModelConstants() {
  const EngineMemoryPlan plan = test::requireMemoryPlan(device(), model());
  require(plan.breakdown().pipelineReserveBytes == model::kPipelineReserveBytes &&
              plan.breakdown().runtimeOverheadReserveBytes ==
                  model::kRuntimeOverheadReserveBytes,
          "the plan's reserves are not the model constants");
  ModelMemoryProfile withoutDecode = model();
  withoutDecode.footprint.runtime.sharedDecodePlannedAllocatedBytes = 0;
  const EngineMemoryPlanResult refused =
      evaluateEngineMemoryPlan(device(), withoutDecode, 0);
  require(!refused.plan &&
              refused.status.code == BudgetErrorCode::InvalidModelSpec &&
              refused.status.message == "shared_decode_bytes_required",
          "a model plan without its decode arena was accepted");
}

void testHardBudgetBoundaries() {
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 0) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(0, 0) == 0 &&
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
  const EngineMemoryPlan plan = test::requireMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  const uint64_t ceiling = budget.minimumRequiredBytes + 64 * kMiB;
  const EngineMemoryPlan limited = test::requireMemoryPlan(device(), model(), ceiling);
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
  EngineMemoryPlan plan = test::requireMemoryPlan(device(), compact);
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.kvPageBytes == 332'800 &&
              budget.kvExtentPages ==
                  compact.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvExtentPages % compact.targetKvLayout.extentAlignmentPages() == 0,
          "memory plan ignored model-provided Q8 geometry");
  require(plan.toStatusJson().find("\"attention_layers\":10") !=
              std::string::npos &&
              plan.toStatusJson().find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

// The pool's extents leave the fewest of the budget's pages unused, the size
// nearest the 128 MiB target on a tie, and a budget below the smallest
// extent holds no pool.
void testExtentSizeFollowsThePool() {
  ModelMemoryProfile compact = model();
  compact.targetKvLayout = {10, 2, 256};
  const EngineMemoryBreakdown reference =
      test::requireMemoryPlan(device(), compact).breakdown();
  const auto planFor = [&](uint64_t pages) {
    return evaluateEngineMemoryPlan(
        device(), compact,
        reference.fixedRuntimeBytes + reference.laneStateBytes +
            pages * reference.kvPageBytes);
  };
#if defined(SPLASH_MACOS15_BUILD)
  constexpr std::array<std::array<uint64_t, 2>, 4> cases{
      {{10'240, 320}, {10'496, 256}, {10'751, 288}, {511, 480}}};
  constexpr uint64_t smallest = 224;
#else
  constexpr std::array<std::array<uint64_t, 2>, 4> cases{
      {{10'240, 512}, {10'496, 256}, {10'751, 256}, {511, 256}}};
  constexpr uint64_t smallest = 256;
#endif
  for (const auto [pages, extent] : cases) {
    const auto result = planFor(pages);
    require(result.plan && result.plan->breakdown().kvExtentPages == extent &&
                result.plan->breakdown().kvCapacityPages == pages - pages % extent &&
                result.plan->breakdown().kvExtentBytes == extent * 332'800,
            "the extent size does not leave the fewest pool pages over");
  }
  const auto tooSmall = planFor(smallest - 1);
  require(!tooSmall.plan && tooSmall.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              tooSmall.status.breakdown.minimumDynamicBytes ==
                  reference.laneStateBytes + smallest * 332'800,
          "a budget below the smallest extent was accepted");
}

// Ternary-Bonsai-2-27B with its DFlash draft: prepared target 7,251,165,184 B,
// target + draft 8,517,206,016 B. The prefill arena is the 2048-row figure;
// smaller rungs are scaled estimates (2% over linear) until a GPU run
// reports the real arena.
ModelMemoryProfile bonsai() {
  ModelMemoryProfile result = test::modelMemoryProfile(
      7'251'165'184ULL, 1'266'040'832ULL, 0);
  result.name = "Ternary-Bonsai-2-27B";
  result.footprint.runtime = {350'224'384ULL, 785'580'032ULL, 262'662'272ULL};
  result.footprint.cachedStateBytes = 196'083'712ULL;
  return result;
}

uint64_t bonsaiPrefillBytes(uint32_t rows) {
  return 785'580'032ULL * rows / 2048 * 102 / 100;
}

// One lane of the shared decode arena is a quarter of the replay tensors.
uint64_t bonsaiDecodeBytes(uint32_t lanes) {
  return lanes == 1 ? 66'042'400ULL : 262'662'272ULL;
}

DeviceCapabilities device16() {
  DeviceCapabilities result = device(11'453'251'584ULL);
  result.physicalMemoryBytes = 17'179'869'184ULL;
  return result;
}

void requireSameBreakdown(const EngineMemoryBreakdown &a,
                          const EngineMemoryBreakdown &b, const char *message) {
  require(a.hardBudgetBytes == b.hardBudgetBytes &&
              a.sharedPrefillBytes == b.sharedPrefillBytes &&
              a.sharedDecodeBytes == b.sharedDecodeBytes &&
              a.prefillRows == b.prefillRows &&
              a.maximumBatchWidth == b.maximumBatchWidth &&
              a.fixedRuntimeBytes == b.fixedRuntimeBytes &&
              a.dynamicBudgetBytes == b.dynamicBudgetBytes &&
              a.kvCapacityPages == b.kvCapacityPages,
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
      const auto direct = test::requireMemoryPlan(max, profile, cap);
      const auto adaptive =
          evaluateAdaptiveMemoryPlan(max, profile, cap, neverCalled, neverCalled);
      require(adaptive.plan && adaptive.plan->breakdown().prefillRows ==
                                   model::ExecutionLimits::prefillTokenBudget &&
                  adaptive.plan->breakdown().maximumBatchWidth ==
                      model::ExecutionLimits::maximumBatchWidth,
              "a fitting plan changed its prefill rows or lanes");
      requireSameBreakdown(adaptive.plan->breakdown(), direct.breakdown(),
                           "a fitting plan differs from the direct plan");
    }
  }
}

// 16 GB: the model's own arenas leave no room for a lane's state and the KV
// runway, so the ladder shrinks the prefill chunk, then the lanes, and keeps a
// prefix-cache snapshot out of one request's KV capacity.
void testAdaptiveLadderFitsBonsaiOn16Gb() {
  const auto direct = evaluateEngineMemoryPlan(device16(), bonsai(), 0);
  require(!direct.plan &&
              direct.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              direct.status.breakdown.hardBudgetBytes == 10'379'509'760ULL,
          "Bonsai no longer misses the 16 GB budget");
  const auto adaptive = evaluateAdaptiveMemoryPlan(
      device16(), bonsai(), 0, bonsaiPrefillBytes, bonsaiDecodeBytes);
  require(adaptive.plan.has_value(), "Bonsai with its draft does not fit 16 GB");
  const auto &budget = adaptive.plan->breakdown();
  require(budget.prefillRows < model::ExecutionLimits::prefillTokenBudget &&
              budget.sharedPrefillBytes == bonsaiPrefillBytes(budget.prefillRows) &&
              budget.minimumRequiredBytes <= budget.hardBudgetBytes &&
              budget.kvCapacityPages >= budget.kvExtentPages &&
              adaptive.plan->maximumContextTokens() > 0,
          "16 GB Bonsai did not take a smaller prefill rung");
  // The snapshot stays out of the request's KV capacity.
  const uint64_t cached = bonsai().footprint.cachedStateBytes;
  require(budget.dynamicBudgetBytes >=
              budget.laneStateBytes + budget.kvCapacityBytes + cached,
          "the plan keeps no room for a prefix-cache snapshot");
  // Rows outrank lanes: the first rung that reaches the goal wins, else the
  // last rung that fits; without the lane rungs the plan keeps all lanes.
  const auto fourLanes =
      evaluateAdaptiveMemoryPlan(device16(), bonsai(), 0, bonsaiPrefillBytes);
  require(fourLanes.plan &&
              fourLanes.plan->breakdown().maximumBatchWidth ==
                  model::ExecutionLimits::maximumBatchWidth,
          "the lane rung changed the plan that does not ask for it");
  require(budget.maximumBatchWidth <= fourLanes.plan->breakdown().maximumBatchWidth &&
              adaptive.plan->maximumContextTokens() >=
                  fourLanes.plan->maximumContextTokens(),
          "dropping lanes did not buy context");
}

// A 32 GB Mac capped at the 16 GB budget plans the same arenas.
void testAdaptiveFollowsTheCapNotPhysicalMemory() {
  DeviceCapabilities max = device(26'800'000'000ULL);
  max.physicalMemoryBytes = 32 * kGiB;
  const auto capped = evaluateAdaptiveMemoryPlan(
      max, bonsai(), 10'379'509'760ULL, bonsaiPrefillBytes, bonsaiDecodeBytes);
  const auto small = evaluateAdaptiveMemoryPlan(
      device16(), bonsai(), 0, bonsaiPrefillBytes, bonsaiDecodeBytes);
  require(capped.plan && small.plan,
          "capped 32 GB or 16 GB Bonsai plan does not fit");
  requireSameBreakdown(capped.plan->breakdown(), small.plan->breakdown(),
                       "the capped 32 GB plan differs from the 16 GB plan");
}

void testAdaptiveNamesTheRungsWhenNothingFits() {
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

#if defined(SPLASH_MACOS15_BUILD)
// The macOS 15 build serves Apple7+ on macOS 15 and refuses older ones.
void testMacos15DeviceFloor() {
  DeviceCapabilities m1 = device();
  m1.appleGpuFamily = 7;
  m1.macosMajor = 15;
  m1.macosMinor = 7;
  require(!m1.validationError(), "an Apple7 Mac on macOS 15.7 was refused");
  DeviceCapabilities old = m1;
  old.macosMajor = 14;
  require(old.validationError().value_or("") == "macos_15_required",
          "macOS 14 was not refused with the macOS 15 reason");
  DeviceCapabilities a13 = m1;
  a13.appleGpuFamily = 6;
  require(a13.validationError().value_or("") == "apple_gpu_family_7_required",
          "a family-6 GPU was not refused with the family-7 reason");
  require(DeviceCapabilities::kMinimumAppleGpuFamily == 7 &&
              m1.validationMessage().value_or("").empty(),
          "the macOS 15 device floor changed");
}
#else
void testDeviceValidationNamesTheMacosFloor() {
  require(!device().validationError(),
          "the reference device reported a validation error");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "macOS 26.3 was not refused with the macOS reason");
  // The operating system is checked before the device.
  older.appleGpuFamily = 8;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "an older macOS did not take precedence over the device's reason");
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
      "Splash needs Apple GPU family 9 or newer (M3 or later) on macOS 26.4 "
      "or newer; this Mac has ";
  DeviceCapabilities m2 = device();
  m2.deviceName = "Apple M2 Max";
  m2.appleGpuFamily = 8;
  m2.macosPatch = 1;
  require(m2.validationMessage().value_or("") ==
              needs + "Apple M2 Max (Apple GPU family 8) on macOS 26.4.1 "
                      "(apple_gpu_family_9_required)",
          "a family-8 GPU was not named against the family required");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationMessage().value_or("") ==
              needs + "test (Apple GPU family 9) on macOS 26.3.0 "
                      "(macos_26_4_required)",
          "an older macOS was not named against the macOS required");
}

#endif

int main() {
  try {
    testUnifiedElasticBudget();
    testBf16BudgetAndStatus();
    testMinimumHoldsTheWarmupRunway();
    testMinimumRequiredBytesIsThePlans();
    testUserCeilingAndFailure();
    testDiskTierStateStagingIsBudgeted();
    testReservesAreTheModelConstants();
    testHardBudgetBoundaries();
    testContextTokensWithin();
    testModelProvidedKvGeometry();
    testExtentSizeFollowsThePool();
    testAdaptivePrefillKeepsFittingPlansUnchanged();
    testAdaptiveLadderFitsBonsaiOn16Gb();
    testAdaptiveFollowsTheCapNotPhysicalMemory();
    testAdaptiveNamesTheRungsWhenNothingFits();
#if defined(SPLASH_MACOS15_BUILD)
    testMacos15DeviceFloor();
#else
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
