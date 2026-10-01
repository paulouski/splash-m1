#include "DFlashDraft.hpp"
#include "DraftCheckpoint.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetHiddenSize || !layout.selectorRank ||
      !layout.kvHeads) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.selectorRank != 256) {
    throw WeightStoreError(
        "draft selector kernels are compiled for rank 256");
  }
  validateQ4Layout(layout.dynamicSize, layout.hiddenSize);
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

} // namespace

DFlashDraftRing::DFlashDraftRing(
    metal::MetalBackend &backend, std::shared_ptr<StateAllocationTracker> tracker,
    DraftStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)), layers_(layout.layers) {
  if (!tracker_)
    throw std::invalid_argument("draft state allocation tracker is empty");
  if (!layout.valid() ||
      layout.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("draft state layout is invalid");
  }
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const metal::MetalBuffer base = backend.allocateBuffer(
      layout.ringBytes(), metal::BufferStorage::Shared, label);
  uint64_t cursor = 0;
  for (DFlashDraftRingLayer &layer : layers_) {
    layer.keys = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
    layer.values = backend.view(base, cursor, layout.tensorBytes());
    cursor += layout.tensorBytes();
  }
  if (cursor != layout.ringBytes())
    throw std::logic_error("draft ring accounting mismatch");
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.ringBytes())
    throw std::logic_error("draft ring allocation is below declared bytes");
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraftRing::~DFlashDraftRing() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

DFlashDraft::DFlashDraft(const DFlashDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(backend, weights.layout.vocabularySize,
                ExecutionLimits::draftQueryRows) {
  requireLayout(weights_.layout);
  if (weights_.layers.size() != weights_.layout.layers ||
      !weights_.layout.stateLayout().valid()) {
    throw std::invalid_argument("draft weights do not match state geometry");
  }

  const ops::DraftAttentionShape expectedContextShape{
      5120, 1280, 6144, 4096, 32, 8, 128};
  if (backend_.capabilities().appleGpuFamily != 7 || weights_.layout.layers != 5 ||
      weights_.layout.attentionShape() != expectedContextShape)
    return;

  const uint32_t contextSize = weights_.layout.attentionShape().contextSize();
  const uint32_t skippedRows = weights_.layout.qkvSize - contextSize;
  for (const DFlashDraftLayerWeights &layer : weights_.layers) {
    const ops::Projection &qkv = layer.qkvProjection;
    if (qkv.layout() != ops::WeightLayout::Affine64 || qkv.bits != 4 ||
        qkv.outputSize != weights_.layout.qkvSize ||
        qkv.inputSize != weights_.layout.hiddenSize || skippedRows % 256)
      return;
  }

  std::vector<ops::Projection> contextKv;
  contextKv.reserve(weights_.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights_.layers) {
    const ops::Projection &qkv = layer.qkvProjection;
    const uint64_t skippedElements = uint64_t{skippedRows} * qkv.inputSize;
    const ops::AffineWeights &weights = qkv.affine();
    const auto tail = [&](const metal::MetalBuffer &plane, uint64_t offset) {
      return backend_.view(plane, offset, plane.sizeBytes() - offset);
    };
    ops::Projection slice{
        contextSize, qkv.inputSize,
        ops::AffineWeights{tail(weights.weights, skippedElements / 2),
                           tail(weights.scales, skippedElements / 32),
                           tail(weights.biases, skippedElements / 32)}};
    slice.bits = qkv.bits;
    slice.destination = qkv.destination;
    contextKv.push_back(std::move(slice));
  }

  const ops::Linear &linear = operators_.linear();
  for (uint32_t layer = 0; layer < weights_.layout.layers; ++layer) {
    for (uint32_t lanes = 1; lanes <= ExecutionLimits::maximumBatchWidth;
         ++lanes) {
      if (!linear.decodeBatchSlicePlan(
              weights_.layers[layer].qkvProjection, contextKv[layer], lanes))
        return;
    }
  }
  contextKv_ = std::move(contextKv);
}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, DFlashSelectionBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, uint32_t proposalTokens) const {
  selector_.addDraftSelector(
      graph,
      {std::move(buffers.logits), std::move(buffers.partialIds),
       std::move(buffers.partialValues), std::move(buffers.candidates),
       std::move(buffers.unary), std::move(buffers.selectorHidden),
       weights_.predecessorCodebook, weights_.successorCodebook,
       std::move(buffers.uniforms), std::move(buffers.proposedTokens),
       std::move(buffers.proposalProbabilities)},
      anchors, policies, proposalTokens);
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  const bool kvOnly = !contextKv_.empty();
  const uint32_t contextWidth = kvOnly
      ? layout.attentionShape().contextSize() : layout.qkvSize;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden, buffers.projectionSums,
                                     weights_.contextProjection, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                 buffers.projected, buffers.projectionSums, rows);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &qkv = kvOnly
        ? contextKv_[layer] : weights_.layers[layer].qkvProjection;
    operators_.linear().addPrefill(graph, buffers.hidden,
                      qkv, buffers.qkv, buffers.projectionSums, rows);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t qkvOffset = uint64_t{span.compactRow} * contextWidth *
                                 sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.qkv, qkvOffset,
                        uint64_t{span.rows} * contextWidth *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          layout.stateLayout().tokens, span.startPosition,
          layout.attentionShape(), kvOnly);
    }
  }
}

void DFlashDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths, uint32_t lanes,
    ops::LinearDispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      cacheLengths.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, buffers.linearScratch,
        operators_.linear().decodePlan(weights.attentionDynamic, lanes).input());
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.attentionDynamic,
                       buffers.dynamic, lanes, stats, buffers.linearScratch,
                       attentionNormalized);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addDecodeBatch(graph, buffers.convolved,
                       weights.qkvProjection, buffers.proposalQkv, lanes,
                       stats, buffers.linearScratch);
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, layout.stateLayout().tokens, attentionPlan);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    operators_.linear().addDecodeBatch(graph,
                       buffers.proposalQkv, weights.outputProjection,
                       buffers.projected, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, buffers.linearScratch,
        operators_.linear().decodePlan(weights.mlpDynamic, lanes).input());
    operators_.linear().addDecodeBatch(graph,
                       buffers.normalized, weights.mlpDynamic, buffers.dynamic,
                       lanes, stats, buffers.linearScratch, mlpNormalized);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    operators_.linear().addGateUpBatch(graph, buffers.convolved, weights.gateProjection,
                       weights.upProjection, buffers.gateScratch,
                       buffers.intermediate, lanes, stats, buffers.linearScratch);
    operators_.linear().addDecodeBatch(graph,
                       buffers.intermediate, weights.downProjection,
                       buffers.projected, lanes, stats, buffers.linearScratch);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, buffers.linearScratch,
      operators_.linear().decodePlan(vocabularyProjection, lanes).input());
  const ops::PreparedInput afterHead = operators_.linear().addDecodeBatch(
      graph, buffers.finalHidden, vocabularyProjection, buffers.logits, lanes, stats,
      buffers.linearScratch, finalHidden);
  operators_.linear().addDecodeBatch(graph,
                     buffers.finalHidden, weights_.selectorProjection,
                     buffers.selectorHidden, lanes, stats, buffers.linearScratch,
                     afterHead);
}

void DFlashDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions, uint32_t lanes,
    ops::LinearDispatchStats &stats) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      startPositions.size() != ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const bool kvOnly = !contextKv_.empty();
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  operators_.linear().addDecodeBatch(graph,
                     buffers.capturedTargetHidden, weights_.contextProjection,
                     buffers.projected, lanes, stats, buffers.linearScratch);
  // Every layer's qkv projection reads the same normalized rows.
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      buffers.linearScratch, operators_.linear().decodePlan(weights_.layers[0].qkvProjection, lanes).input());

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &qkv = kvOnly
        ? contextKv_[layer] : weights_.layers[layer].qkvProjection;
    if (kvOnly) {
      hidden = operators_.linear().addDecodeBatchSlice(
          graph, buffers.hidden, weights_.layers[layer].qkvProjection, qkv,
          buffers.qkv, lanes, stats, buffers.linearScratch, hidden);
    } else {
      hidden = operators_.linear().addDecodeBatch(
          graph, buffers.hidden, qkv, buffers.qkv, lanes, stats,
          buffers.linearScratch, hidden);
    }
    ops::DraftAttention::addContextCommit(
        graph, buffers.qkv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.stateLayout().tokens, layout.attentionShape(),
        lanes, kvOnly);
  }
}

namespace {

// Reads a draft's files in their section order: each layer, then model.bin.
template <class Files>
DFlashDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                             const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t convolutionBytes = checkedWeightMultiply(
      checkedWeightMultiply(4, layout.hiddenSize,
                            "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedWeightMultiply(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readAffineProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readAffineProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readAffineProjection(
        file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
    layer.gateProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readAffineProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readAffineProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.selectorProjection = readAffineProjection(
        file, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedWeightMultiply(
        checkedWeightMultiply(layout.vocabularySize, layout.selectorRank,
                              "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackedDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return WeightFile(backend, directory / filename, "draft/" + filename,
                    kDFlashLayerMagic, index, 0);
}

WeightFile PackedDraftFiles::model() const {
  return WeightFile(backend, directory / "model.bin", "draft/model.bin",
                    kDFlashLayerMagic, layout.layers, 1);
}

DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend,
                                          const DraftFiles &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DraftCheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackedDraftFiles>(files), layout);
}

} // namespace splash::model
