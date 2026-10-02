#include "engine/MemoryPlan.hpp"
#include "engine/Checked.hpp"
#include "engine/Json.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

std::string bytesAndMiB(uint64_t bytes) {
  std::ostringstream out;
  out << bytes << " bytes (" << std::fixed << std::setprecision(2)
      << static_cast<long double>(bytes) / kMiB << " MiB)";
  return out.str();
}

BudgetValidationStatus failure(BudgetErrorCode code, std::string message,
                               EngineMemoryBreakdown breakdown) {
  return {false, code, std::move(message), std::move(breakdown)};
}

} // namespace

std::string_view budgetErrorCodeName(BudgetErrorCode code) {
  switch (code) {
  case BudgetErrorCode::None:
    return "none";
  case BudgetErrorCode::InvalidDeviceCapabilities:
    return "invalid_device_capabilities";
  case BudgetErrorCode::InvalidModelSpec:
    return "invalid_model_spec";
  case BudgetErrorCode::WorkingSetTooSmall:
    return "working_set_too_small";
  case BudgetErrorCode::ArithmeticOverflow:
    return "arithmetic_overflow";
  case BudgetErrorCode::KvPoolDoesNotFit:
    return "kv_pool_does_not_fit";
  }
  return "unknown";
}

std::string deviceStatusJson(const DeviceCapabilities &device) {
  std::ostringstream out;
  out << '{' << "\"device_name\":" << json::quote(device.deviceName) << ','
      << "\"macos_version\":" << json::quote(device.macosVersion()) << ','
      << "\"apple_gpu_family\":" << device.appleGpuFamily << ','
      << "\"gpu_core_count\":" << device.gpuCoreCount << ','
      << "\"physical_memory_bytes\":" << device.physicalMemoryBytes << ','
      << "\"recommended_max_working_set_bytes\":"
      << device.recommendedMaxWorkingSetBytes << ','
      << "\"max_buffer_length_bytes\":" << device.maxBufferLengthBytes << ','
      << "\"max_threadgroup_memory_bytes\":"
      << device.maxThreadgroupMemoryBytes << ','
      << "\"max_threadgroup_width\":" << device.maxThreadgroupWidth << ','
      << "\"has_unified_memory\":"
      << (device.hasUnifiedMemory ? "true" : "false") << ','
      << "\"supports_placement_sparse\":"
      << (device.supportsPlacementSparse ? "true" : "false") << '}';
  return out.str();
}

std::optional<std::string> ModelMemoryProfile::validationError() const {
  if (name.empty()) return "model_name_required";
  if (!maximumContextTokens ||
      maximumContextTokens > kv::kMaximumLogicalTokens) {
    return "invalid_model_context_length";
  }
  if (!targetKvLayout.valid()) return "invalid_target_kv_layout";
  if (!footprint.targetWeightsBytes) return "target_weight_bytes_required";
  if (!footprint.draftWeightsBytes) return "draft_weight_bytes_required";
  if (!footprint.activeStateCellBytes) {
    return "active_state_cell_bytes_required";
  }
  if (!footprint.sharedPrefillBytes) return "shared_prefill_bytes_required";
  if (!footprint.sharedDecodeBytes) return "shared_decode_bytes_required";
  if (!footprint.pipelineReserveBytes) return "pipeline_reserve_required";
  if (!footprint.runtimeOverheadReserveBytes) {
    return "runtime_overhead_reserve_required";
  }
  try {
    static_cast<void>(fixedRuntimeBytes());
  } catch (const std::overflow_error &) {
    return "fixed_runtime_cost_overflow";
  }
  return std::nullopt;
}

uint64_t ModelMemoryProfile::fixedRuntimeBytes() const {
  uint64_t result = 0;
  for (uint64_t value : {
           footprint.targetWeightsBytes, footprint.draftWeightsBytes,
           footprint.visionWeightsBytes, footprint.sharedPrefillBytes,
           footprint.sharedDecodeBytes, footprint.pipelineReserveBytes,
           footprint.runtimeOverheadReserveBytes, footprint.kvStagingBytes}) {
    if (!checkedAdd(result, value, result)) {
      throw std::overflow_error("fixed runtime cost overflow");
    }
  }
  return result;
}

std::string modelStatusJson(const ModelMemoryProfile &model) {
  std::ostringstream out;
  out << '{' << "\"model_name\":" << json::quote(model.name) << ','
      << "\"maximum_context_tokens\":" << model.maximumContextTokens << ','
      << "\"attention_layers\":" << model.targetKvLayout.attentionLayers << ','
      << "\"kv_heads\":" << model.targetKvLayout.kvHeads << ','
      << "\"head_dimension\":" << model.targetKvLayout.headDimension << ','
      << "\"kv_page_tokens\":" << kv::kPageTokens << ','
      << "\"kv_quantization_bits\":"
      << (model.targetKvLayout.format == kv::Format::Int8 ? 8 : 16) << ','
      << "\"kv_format\":" << json::quote(kv::formatName(model.targetKvLayout.format)) << ','
      << "\"kv_elements_per_scale\":"
      << model.targetKvLayout.elementsPerScale() << ','
      << "\"kv_scale_value_bytes\":"
      << (model.targetKvLayout.format == kv::Format::Int8 ? sizeof(float) : 0) << ','
      << "\"kv_page_bytes\":" << model.targetKvLayout.bytesPerModelPage();
  // Keep the legacy field for existing INT8 status consumers.
  if (model.targetKvLayout.format == kv::Format::Int8)
    out << ",\"q8_page_bytes\":" << model.targetKvLayout.bytesPerModelPage();
  out
      << ','
      << "\"memory\":{" << "\"target_weights_bytes\":"
      << model.footprint.targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << model.footprint.draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << model.footprint.visionWeightsBytes << ','
      << "\"active_state_cell_bytes\":"
      << model.footprint.activeStateCellBytes << ','
      << "\"shared_prefill_bytes\":" << model.footprint.sharedPrefillBytes << ','
      << "\"shared_decode_bytes\":" << model.footprint.sharedDecodeBytes << ','
      << "\"pipeline_reserve_bytes\":" << model.footprint.pipelineReserveBytes
      << ',' << "\"runtime_overhead_reserve_bytes\":"
      << model.footprint.runtimeOverheadReserveBytes << ','
      << "\"kv_staging_bytes\":" << model.footprint.kvStagingBytes << "}}";
  return out.str();
}

std::string EngineMemoryBreakdown::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"physical_memory_bytes\":" << physicalMemoryBytes << ','
      << "\"recommended_working_set_bytes\":" << recommendedWorkingSetBytes
      << ','
      << "\"configured_memory_limit_bytes\":" << configuredMemoryLimitBytes
      << ',' << "\"headroom_bytes\":" << headroomBytes << ','
      << "\"hard_budget_bytes\":" << hardBudgetBytes << ','
      << "\"target_weights_bytes\":" << targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << visionWeightsBytes << ','
      << "\"maximum_batch_width\":" << maximumBatchWidth << ','
      << "\"active_state_cell_bytes\":" << activeStateCellBytes << ','
      << "\"shared_prefill_bytes\":" << sharedPrefillBytes << ','
      << "\"shared_decode_bytes\":" << sharedDecodeBytes << ','
      << "\"pipeline_reserve_bytes\":" << pipelineReserveBytes << ','
      << "\"runtime_overhead_reserve_bytes\":" << runtimeOverheadReserveBytes
      << ',' << "\"kv_staging_bytes\":" << kvStagingBytes << ','
      << "\"fixed_runtime_bytes\":" << fixedRuntimeBytes << ','
      << "\"dynamic_budget_bytes\":" << dynamicBudgetBytes << ','
      << "\"kv_page_tokens\":" << kvPageTokens << ','
      << "\"kv_page_bytes\":" << kvPageBytes << ','
      << "\"kv_sparse_mapping_batch_pages\":" << kvSparseMappingBatchPages
      << ','
      << "\"kv_extent_pages\":" << kvExtentPages << ','
      << "\"kv_extent_bytes\":" << kvExtentBytes << ','
      << "\"maximum_kv_pages\":" << maximumKvPages << ','
      << "\"kv_virtual_pages\":" << kvVirtualPages << ','
      << "\"kv_virtual_bytes\":" << kvVirtualBytes << ','
      << "\"kv_virtual_tokens\":" << kvVirtualTokens << ','
      << "\"minimum_dynamic_bytes\":" << minimumDynamicBytes << ','
      << "\"minimum_required_bytes\":" << minimumRequiredBytes << ','
      << "\"deficit_bytes\":" << deficitBytes << '}';
  return out.str();
}

std::string EngineMemoryBreakdown::describe() const {
  std::ostringstream out;
  out << "physical memory: " << bytesAndMiB(physicalMemoryBytes) << '\n'
      << "recommended working set: " << bytesAndMiB(recommendedWorkingSetBytes)
      << '\n'
      << "configured memory limit: "
      << (configuredMemoryLimitBytes ? bytesAndMiB(configuredMemoryLimitBytes)
                                     : "automatic")
      << '\n'
      << "working-set margin (max of 1 GiB or 2%): "
      << bytesAndMiB(headroomBytes) << '\n'
      << "hard budget: " << bytesAndMiB(hardBudgetBytes) << '\n'
      << "target weights: " << bytesAndMiB(targetWeightsBytes) << '\n'
      << "draft weights: " << bytesAndMiB(draftWeightsBytes) << '\n'
      << "vision weights: " << bytesAndMiB(visionWeightsBytes) << '\n'
      << "maximum DFlash batch width: " << maximumBatchWidth << '\n'
      << "active state cell: " << bytesAndMiB(activeStateCellBytes) << '\n'
      << "shared prefill: " << bytesAndMiB(sharedPrefillBytes) << '\n'
      << "shared decode: " << bytesAndMiB(sharedDecodeBytes) << '\n'
      << "pipeline reserve: " << bytesAndMiB(pipelineReserveBytes) << '\n'
      << "allocator/runtime reserve: "
      << bytesAndMiB(runtimeOverheadReserveBytes) << '\n'
      << "disk tier KV staging: " << bytesAndMiB(kvStagingBytes) << '\n'
      << "fixed runtime: " << bytesAndMiB(fixedRuntimeBytes) << '\n'
      << "elastic state/KV budget: " << bytesAndMiB(dynamicBudgetBytes) << '\n'
      << "KV page: " << kvPageTokens << " tokens, "
      << bytesAndMiB(kvPageBytes) << '\n'
      << "KV physical extent: " << kvExtentPages << " pages, "
      << bytesAndMiB(kvExtentBytes) << '\n'
      << "KV virtual address space: " << kvVirtualPages << " pages / "
      << kvVirtualTokens << " tokens (geometry maximum " << maximumKvPages
      << ")\n"
      << "minimum dynamic runtime: " << bytesAndMiB(minimumDynamicBytes) << '\n'
      << "minimum required: " << bytesAndMiB(minimumRequiredBytes) << '\n'
      << "deficit: " << bytesAndMiB(deficitBytes);
  return out.str();
}

std::string BudgetValidationStatus::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"schema_version\":2,"
      << "\"valid\":" << (valid ? "true" : "false") << ',' << "\"error_code\":";
  if (valid) {
    out << "null";
  } else {
    out << json::quote(budgetErrorCodeName(code));
  }
  out << ',' << "\"message\":" << json::quote(message) << ','
      << "\"budget\":" << breakdown.toStatusJson() << '}';
  return out.str();
}

std::string BudgetValidationStatus::describe() const {
  std::ostringstream out;
  out << (valid ? "engine memory plan valid"
                : "engine memory budget validation failed")
      << " [" << budgetErrorCodeName(code) << "]: " << message << '\n'
      << breakdown.describe();
  return out.str();
}

EngineMemoryPlan::EngineMemoryPlan(DeviceCapabilities device,
                                   ModelMemoryProfile model,
                                   EngineMemoryBreakdown breakdown)
    : device_(std::move(device)), model_(std::move(model)),
      breakdown_(std::move(breakdown)) {}

uint32_t EngineMemoryPlan::maximumContextTokens() const noexcept {
  const uint64_t physicalCapacity = breakdown_.kvVirtualTokens;
  const uint64_t logicalCapacity =
      physicalCapacity > model::ExecutionLimits::speculativeScratchTokens
          ? physicalCapacity - model::ExecutionLimits::speculativeScratchTokens
          : 0;
  return static_cast<uint32_t>(
      std::min<uint64_t>(model_.maximumContextTokens, logicalCapacity));
}

uint32_t EngineMemoryPlan::contextTokensWithin(uint64_t memoryBytes) const {
  if (!memoryBytes)
    return 0;
  const uint64_t configured = breakdown_.configuredMemoryLimitBytes;
  const EngineMemoryPlanResult within = evaluateEngineMemoryPlan(
      device_, model_, configured ? std::min(configured, memoryBytes) : memoryBytes);
  return within.plan ? within.plan->maximumContextTokens() : 0;
}

std::string EngineMemoryPlan::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"valid\":true,"
      << "\"maximum_context_tokens\":" << maximumContextTokens() << ','
      << "\"device\":" << deviceStatusJson(device_) << ','
      << "\"model\":" << modelStatusJson(model_) << ','
      << "\"budget\":" << breakdown_.toStatusJson() << '}';
  return out.str();
}

EngineMemoryPlanResult
evaluateEngineMemoryPlan(const DeviceCapabilities &device,
                         const ModelMemoryProfile &model,
                         uint64_t maximumMemoryBytes) {
  EngineMemoryBreakdown breakdown;
  breakdown.physicalMemoryBytes = device.physicalMemoryBytes;
  breakdown.recommendedWorkingSetBytes = device.recommendedMaxWorkingSetBytes;
  breakdown.configuredMemoryLimitBytes = maximumMemoryBytes;
  breakdown.targetWeightsBytes = model.footprint.targetWeightsBytes;
  breakdown.draftWeightsBytes = model.footprint.draftWeightsBytes;
  breakdown.visionWeightsBytes = model.footprint.visionWeightsBytes;
  breakdown.activeStateCellBytes = model.footprint.activeStateCellBytes;
  breakdown.sharedPrefillBytes = model.footprint.sharedPrefillBytes;
  breakdown.sharedDecodeBytes = model.footprint.sharedDecodeBytes;
  breakdown.pipelineReserveBytes = model.footprint.pipelineReserveBytes;
  breakdown.runtimeOverheadReserveBytes =
      model.footprint.runtimeOverheadReserveBytes;
  breakdown.prefillRows = model.footprint.prefillRows;
  breakdown.maximumBatchWidth = model.footprint.decodeLanes;
  breakdown.kvStagingBytes = model.footprint.kvStagingBytes;
  breakdown.kvPageTokens = kv::kPageTokens;
  breakdown.kvSparseMappingBatchPages =
      model.targetKvLayout.sparseMappingBatchPages();
  breakdown.kvExtentPages = model.targetKvLayout.backingExtentPages();

  if (auto error = device.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidDeviceCapabilities,
                                  *error, std::move(breakdown))};
  }
  if (auto error = model.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidModelSpec, *error,
                                  std::move(breakdown))};
  }

  breakdown.headroomBytes = EngineMemoryPolicy::workingSetMarginBytes(
      breakdown.recommendedWorkingSetBytes);
  if (breakdown.recommendedWorkingSetBytes <= breakdown.headroomBytes) {
    return {std::nullopt,
            failure(BudgetErrorCode::WorkingSetTooSmall,
                    "recommended working set does not exceed required headroom",
                    std::move(breakdown))};
  }
  breakdown.hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
      breakdown.recommendedWorkingSetBytes, maximumMemoryBytes);

  breakdown.kvPageBytes = model.targetKvLayout.bytesPerModelPage();
  breakdown.fixedRuntimeBytes = model.fixedRuntimeBytes();
  breakdown.dynamicBudgetBytes =
      breakdown.hardBudgetBytes > breakdown.fixedRuntimeBytes
          ? breakdown.hardBudgetBytes - breakdown.fixedRuntimeBytes
          : 0;
  if (!checkedMultiply(breakdown.kvPageBytes, breakdown.kvExtentPages,
                       breakdown.kvExtentBytes) ||
      !checkedAdd(breakdown.activeStateCellBytes, breakdown.kvExtentBytes,
                  breakdown.minimumDynamicBytes) ||
      !checkedAdd(breakdown.fixedRuntimeBytes, breakdown.minimumDynamicBytes,
                  breakdown.minimumRequiredBytes)) {
    return {std::nullopt,
            failure(BudgetErrorCode::ArithmeticOverflow,
                    "minimum elastic runtime footprint overflows uint64",
                    std::move(breakdown))};
  }
  const uint64_t largestBufferBytesPerPage =
      std::max(model.targetKvLayout.dataBytesPerLayerPage(),
               model.targetKvLayout.scaleBytesPerLayerPage());
  uint64_t geometryMaximum =
      device.maxBufferLengthBytes / largestBufferBytesPerPage;
  geometryMaximum -= geometryMaximum % breakdown.kvSparseMappingBatchPages;
  if (geometryMaximum > std::numeric_limits<uint32_t>::max()) {
    geometryMaximum = std::numeric_limits<uint32_t>::max();
    geometryMaximum -= geometryMaximum % breakdown.kvSparseMappingBatchPages;
  }
  breakdown.maximumKvPages = static_cast<uint32_t>(geometryMaximum);
  uint64_t reservedBytes = 0;
  if (!checkedAdd(breakdown.activeStateCellBytes, model.snapshotReserveBytes,
                  reservedBytes))
    reservedBytes = std::numeric_limits<uint64_t>::max();
  const uint64_t availableForOneRequestKv =
      breakdown.dynamicBudgetBytes > reservedBytes
          ? breakdown.dynamicBudgetBytes - reservedBytes
          : 0;
  uint64_t clampedKvPages =
      std::min<uint64_t>(availableForOneRequestKv / breakdown.kvPageBytes,
                         breakdown.maximumKvPages);
  // Non-sparse pools are plain buffers, so the 64-KiB sparse tile alignment
  // behind the mapping batch is policy, not a requirement; the low-memory
  // profile sizes them to whole pages.
  const uint32_t kvPageUnit = !device.supportsPlacementSparse && model.exactKvPages
                                  ? 1
                                  : breakdown.kvSparseMappingBatchPages;
  clampedKvPages -= clampedKvPages % kvPageUnit;
  if (!device.supportsPlacementSparse && model.requestedContextTokens) {
    // Non-sparse backing is committed in full at load, not grown on demand:
    // shrink the pool to what --max-context plus speculative scratch needs,
    // rounded up to whole sparse-mapping batches, so the remainder of the
    // dynamic budget stays free for prefix-cache state snapshots instead of
    // being consumed by KV pages the configured context will never use.
    const uint64_t requestedTokens = uint64_t{model.requestedContextTokens} +
                                     model::ExecutionLimits::speculativeScratchTokens;
    uint64_t requestedPages =
        (requestedTokens + breakdown.kvPageTokens - 1) / breakdown.kvPageTokens;
    requestedPages += kvPageUnit - 1;
    requestedPages -= requestedPages % kvPageUnit;
    requestedPages = std::max<uint64_t>(requestedPages, breakdown.kvExtentPages);
    clampedKvPages = std::min(clampedKvPages, requestedPages);
  }
  breakdown.kvVirtualPages = static_cast<uint32_t>(clampedKvPages);

  if (!checkedMultiply(breakdown.kvPageBytes, breakdown.kvVirtualPages,
                       breakdown.kvVirtualBytes) ||
      !checkedMultiply(breakdown.kvPageTokens, breakdown.kvVirtualPages,
                       breakdown.kvVirtualTokens)) {
    return {std::nullopt, failure(BudgetErrorCode::ArithmeticOverflow,
                                  "virtual KV capacity overflows uint64",
                                  std::move(breakdown))};
  }
  if (breakdown.maximumKvPages < breakdown.kvExtentPages ||
      breakdown.kvVirtualPages < breakdown.kvExtentPages ||
      breakdown.dynamicBudgetBytes < breakdown.minimumDynamicBytes) {
    if (breakdown.minimumRequiredBytes > breakdown.hardBudgetBytes) {
      breakdown.deficitBytes =
          breakdown.minimumRequiredBytes - breakdown.hardBudgetBytes;
    }
    return {
        std::nullopt,
        failure(
            BudgetErrorCode::KvPoolDoesNotFit,
            "hard budget or Metal geometry cannot fit one active state cell "
            "and one KV physical extent",
            std::move(breakdown))};
  }

  BudgetValidationStatus status{true, BudgetErrorCode::None,
                                "memory plan fits hard budget", breakdown};
  EngineMemoryPlan plan(device, model, breakdown);
  return {std::move(plan), std::move(status)};
}

static EngineMemoryPlanResult adaptiveLadder(
    const DeviceCapabilities &device, ModelMemoryProfile model,
    uint64_t maximumMemoryBytes,
    const std::function<uint64_t(uint32_t)> &prefillBytesForRows,
    const std::function<uint64_t(uint32_t)> &decodeBytesForLanes,
    EngineMemoryPlanResult result) {
  const uint32_t goal =
      std::min(model.requestedContextTokens ? model.requestedContextTokens
                                            : kAdaptiveContextGoalTokens,
               model.maximumContextTokens);
  model.exactKvPages = true;
  const ModelMemoryFootprint full = model.footprint;
  std::vector<uint32_t> rungs{full.prefillRows};
  for (const uint32_t rows : kReducedPrefillRows) {
    if (rows < full.prefillRows)
      rungs.push_back(rows);
  }
  std::vector<uint32_t> laneRungs{full.decodeLanes};
  if (decodeBytesForLanes && full.decodeLanes > 1)
    laneRungs.push_back(1);
  std::string tried = std::to_string(full.prefillRows);
  std::optional<EngineMemoryPlanResult> fitted;
  for (const uint32_t rows : rungs) {
    bool rowFits = false;
    for (const uint32_t lanes : laneRungs) {
      if (rows == full.prefillRows && lanes == full.decodeLanes)
        continue;
      model.footprint = full;
      if (rows != full.prefillRows) {
        model.footprint.prefillRows = rows;
        model.footprint.sharedPrefillBytes = prefillBytesForRows(rows);
      }
      if (lanes != full.decodeLanes) {
        model.footprint.decodeLanes = lanes;
        model.footprint.sharedDecodeBytes = decodeBytesForLanes(lanes);
      }
      EngineMemoryPlanResult attempt =
          evaluateEngineMemoryPlan(device, model, maximumMemoryBytes);
      if (attempt.plan) {
        rowFits = true;
        fitted = std::move(attempt);
        if (fitted->plan->maximumContextTokens() >= goal)
          return std::move(*fitted);
        continue;
      }
      if (attempt.status.code != BudgetErrorCode::KvPoolDoesNotFit)
        return attempt;
      result = std::move(attempt);
    }
    if (!rowFits && rows != full.prefillRows)
      tried += ", " + std::to_string(rows);
  }
  if (fitted)
    return std::move(*fitted);
  result.status.message +=
      "; packed prefill chunks of " + tried + " rows were tried";
  return result;
}

EngineMemoryPlanResult evaluateAdaptiveMemoryPlan(
    const DeviceCapabilities &device, ModelMemoryProfile model,
    uint64_t maximumMemoryBytes,
    const std::function<uint64_t(uint32_t)> &prefillBytesForRows,
    const std::function<uint64_t(uint32_t)> &decodeBytesForLanes) {
  EngineMemoryPlanResult result =
      evaluateEngineMemoryPlan(device, model, maximumMemoryBytes);
  if (result.plan || result.status.code != BudgetErrorCode::KvPoolDoesNotFit)
    return result;
  // Sparse pools grow KV on demand, so only dense pools lose context to snapshots.
  if (model.footprint.cachedStateBytes && !device.supportsPlacementSparse) {
    ModelMemoryProfile reserved = model;
    // 1 MiB covers backend allocations the plan does not count.
    reserved.snapshotReserveBytes = model.footprint.cachedStateBytes + kMiB;
    EngineMemoryPlanResult withSnapshot =
        adaptiveLadder(device, std::move(reserved), maximumMemoryBytes,
                       prefillBytesForRows, decodeBytesForLanes, result);
    if (withSnapshot.plan)
      return withSnapshot;
  }
  return adaptiveLadder(device, std::move(model), maximumMemoryBytes,
                        prefillBytesForRows, decodeBytesForLanes,
                        std::move(result));
}

EngineMemoryPlan requireEngineMemoryPlan(const DeviceCapabilities &device,
                                         const ModelMemoryProfile &model,
                                         uint64_t maximumMemoryBytes) {
  EngineMemoryPlanResult result =
      evaluateEngineMemoryPlan(device, model, maximumMemoryBytes);
  if (!result.plan)
    throw std::runtime_error(result.status.describe());
  return std::move(*result.plan);
}

} // namespace splash::engine
