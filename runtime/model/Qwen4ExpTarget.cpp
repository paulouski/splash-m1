#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/Gguf.h"
#include "metal/abi/Qwen4.h"
#include "model/PleStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/Qwen4.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kLaneRows = ExecutionLimits::targetVerifyRows;
constexpr uint32_t kIndexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;

template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

// Rows from `begin` of a row-major buffer of `width` values of T to its end: a projection's tile may run past
// the rows it is given.
template <class T>
metal::MetalBuffer rowsFrom(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                            uint32_t width) {
  const uint64_t offset = uint64_t{begin} * width * sizeof(T);
  return backend.view(buffer, offset, buffer.sizeBytes() - offset);
}

// The QSA indexer's fp32 projections of `rows` rows: one float segment each
// (Qwen4Exp.cpp readIndexer), at any row count.
void addIndexerProjection(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                          const ops::Projection &projection, metal::MetalBuffer output, uint32_t rows) {
  const auto &segments = projection.blocks().segments;
  if (segments.size() != 1 || !segments.front().isFloat())
    throw std::invalid_argument("the QSA indexer projections are single float segments");
  ops::addGgufFloat(graph, std::move(input), segments.front(), std::move(output), rows, projection.outputSize, 0,
                    ops::FloatOutput::Float32, linear.ggufFloatTile(rows, projection.outputSize));
}

// The staged PLE rows as a table the embedding gather reads: one native
// IQ4_NL row per slot.
ops::EmbeddingWeights stagedTable(const metal::MetalBuffer &staging, uint32_t rowValues) {
  return {static_cast<uint32_t>(staging.sizeBytes() / pleRowBytes), rowValues,
          ops::NativeRows(staging, GGUF_FMT_IQ4NL)};
}

// The QSA blocks a step's rows complete: those whose last token is one of
// `rows` rows from `first`.
uint32_t qsaBlockCount(uint32_t first, uint32_t rows) {
  return (first + rows) / QWEN4_QSA_BLOCK - first / QWEN4_QSA_BLOCK;
}

} // namespace

// The state a prefill command's layers share.
struct Qwen4ExpTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const SplashKvLayer> kvLayers;
  std::vector<ops::PrefillAttentionPlan> attention{};
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct Qwen4ExpTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const SplashKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

Qwen4ExpTarget::Qwen4ExpTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry,
                               metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : weights_(weights), geometry_(geometry), backend_(backend), operators_(operators) {
  if (!geometry_.qwen4 || weights.layers.size() != geometry_.layers)
    throw std::invalid_argument("Qwen3.8-Flash-Next weights do not match execution geometry");
}

ops::HyperShape Qwen4ExpTarget::hyperShape() const noexcept {
  return {geometry_.hiddenSize, geometry_.qwen4.hyperConnections, geometry_.qwen4.hyperRank};
}

ops::PleShape Qwen4ExpTarget::pleShape() const noexcept {
  return {geometry_.hiddenSize, geometry_.qwen4.hyperConnections, geometry_.qwen4.pleConvolutionTaps,
          geometry_.qwen4.pleNgram};
}

metal::MetalBuffer Qwen4ExpTarget::addPrefill(metal::CommandGraph &graph, const QwenTargetPrefillBuffers &b,
                                              std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
                                              std::span<const SplashKvLayer> kvLayers) const {
  if (sequences.empty() || sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget || kvLayers.size() != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  for (const QwenTargetPrefillSequence &sequence : sequences)
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers || !sequence.pleHistoryIn ||
        !sequence.pleHistoryOut)
      throw std::invalid_argument("Qwen prefill state layer mismatch");
  PrefillStep step{graph, b, sequences, rows, kvLayers};
  for (const QwenTargetPrefillSequence &sequence : sequences)
    step.attention.push_back(
        operators_.prefillAttention(sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
  step.moe = operators_.moePrefill(geometry_.moeShape(), rows);

  const Qwen4Buffers &h = b.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape();
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output) {
    linear.addPrefill(graph, std::move(input), projection, std::move(output), b.projectionSums, rows,
                      b.linearScratch);
  };
  // The mix of the streams into `output` and, when it injects, the injection
  // weights into h.weights.
  const auto mix = [&](const Qwen4HyperWeights &hyper, metal::MetalBuffer output, metal::MetalBuffer streams) {
    ops::Qwen4::addNorm(graph, std::move(streams), hyper.norm, hyper.inject, h.normalized, h.weights, hs, rows);
    project(h.normalized, hyper.down, h.low);
    ops::Qwen4::addLow(graph, h.low, hs, rows);
    project(h.low, hyper.up, h.gate);
    ops::Qwen4::addMix(graph, h.normalized, h.gate, std::move(output), hs, rows);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<float>(backend_, buffer, begin, count, width);
  };
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<uint16_t>(backend_, buffer, begin, count, width);
  };
  // A full-attention block's mixer over the prefill rows: the QSA indexer's
  // keys, the attention's KV into the pages and, when `attend`, the attention
  // and its output projection into h.branch.
  const auto addAttention = [&](const Qwen4ExpLayerWeights &layer, const QwenAttentionWeights &mixer,
                                uint32_t attentionLayer, bool attend) {
    const Qwen4IndexerWeights &indexer = *layer.indexer;
    const uint32_t topBlocks = geometry_.qwen4.indexerTopBlocks;
    project(b.normalized, mixer.inputProjection, b.fullPacked);
    addIndexerProjection(graph, linear, b.normalized, indexer.query, h.indexerQuery, rows);
    addIndexerProjection(graph, linear, b.normalized, indexer.key, h.indexerKey, rows);
    for (size_t s = 0; s < sequences.size(); ++s) {
      const QwenTargetPrefillSequence &sequence = sequences[s];
      const uint32_t first = static_cast<uint32_t>(sequence.chunk.committed_tokens);
      const ops::QsaRows qsa{sequence.rows, first, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
      const metal::MetalBuffer queries = f32Rows(h.qsaQueries, sequence.rowBegin, sequence.rows, kIndexWidth);
      ops::Qwen4::addQsaProject(
          graph, f32Rows(h.indexerQuery, sequence.rowBegin, sequence.rows, kIndexWidth),
          f32Rows(h.indexerKey, sequence.rowBegin, sequence.rows, QWEN4_QSA_DIMENSION), indexer.queryNorm,
          f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
          f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs), sequence.pageTable,
          h.qsaIndex[attentionLayer], queries, qsa);
      const uint64_t blocksBegin = uint64_t{sequence.rowBegin} * sizeof(Qwen4QsaBlock);
      ops::Qwen4::addQsaPool(graph, backend_.view(h.qsaBlocks, blocksBegin, h.qsaBlocks.sizeBytes() - blocksBegin),
                             qsaBlockCount(first, sequence.rows), indexer.keyNorm, h.inverseFrequencies,
                             sequence.pageTable, h.qsaIndex[attentionLayer]);
      const bool sparse = ops::Qwen4::qsaSparse(first, sequence.rows, topBlocks);
      if (attend && sparse)
        ops::Qwen4::addQsaSelect(graph, queries, h.qsaIndex[attentionLayer], sequence.pageTable, h.qsaScores,
                                 h.qsaScoreRows,
                                 rowsOf<uint32_t>(backend_, h.qsaMask, sequence.rowBegin, sequence.rows,
                                                  h.qsaMaskWords),
                                 qsa);
      const uint64_t headBytes =
          uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
      const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
      const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
      const metal::MetalBuffer attentionQueries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
      const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
      const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
      const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
      ops::PagedAttention::addPrefillProjection(
          graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
          mixer.queryNorm, mixer.keyNorm,
          f32Rows(b.ropeCos, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs),
          f32Rows(b.ropeSin, sequence.rowBegin, sequence.rows, geometry_.rotaryPairs), attentionQueries, keys,
          values, sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
      ops::PagedAttention::addPrefillStore(graph, kvLayers[attentionLayer], keys, values, sequence.pageTable,
                                           sequence.chunk, geometry_.kvLayout);
      if (!attend) continue;
      ops::PagedAttention::addPrefill(
          graph, kvLayers[attentionLayer], attentionQueries, attentionRows, b.attentionPartials,
          b.attentionStatistics, sequence.pageTable, sequence.chunk, step.attention[s],
          ops::QsaMask{h.qsaMask, sparse ? h.qsaMaskWords : 0, sequence.rowBegin});
      ops::PagedAttention::addPrefillGate(
          graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
          attentionRows, u16Rows(b.attentionHidden, sequence.rowBegin, sequence.rows, geometry_.attentionWidth),
          sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
    }
    if (attend) project(b.attentionHidden, mixer.outputProjection, h.branch);
  };
  ops::Qwen4::addInit(graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights_.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape();
      ops::Embedding::add(graph, h.pleSlots, stagedTable(h.pleStaging, geometry_.qwen4.pleHeadDimension),
                          h.pleRows, rows * geometry_.qwen4.pleHeads());
      project(h.pleRows, ple.key, h.gate);
      project(h.pleRows, ple.value, h.pleValue);
      ops::Qwen4::addPleGate(graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm, ple.convolutionNorm,
                             h.pleGated, h.pleConvolution, ps, rows);
      for (const QwenTargetPrefillSequence &sequence : sequences)
        ops::Qwen4::addPleConvolution(graph, u16Rows(h.pleConvolution, sequence.rowBegin, sequence.rows, ps.width()),
                                      sequence.pleHistoryIn, sequence.pleHistoryOut, ple.convolution,
                                      f32Rows(h.pleGated, sequence.rowBegin, sequence.rows, ps.width()),
                                      f32Rows(h.streams, sequence.rowBegin, sequence.rows, ps.width()), ps,
                                      sequence.rows);
    }
    mix(layer.mixerHyper, b.normalized, h.streams);
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            project(b.normalized, mixer.inputProjection, b.gdnPacked);
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
                return u16Rows(buffer, sequence.rowBegin, sequence.rows, width);
              };
              const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
                return f32Rows(buffer, sequence.rowBegin, sequence.rows, width);
              };
              ops::GDN::addPrefill(
                  graph,
                  {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights,
                   sequence.convolutionIn[gdnLayer], sequence.convolutionOut[gdnLayer],
                   u16(b.gdnQueries, geometry_.gdnKeyWidth()), u16(b.gdnKeys, geometry_.gdnKeyWidth()),
                   u16(b.gdnValues, geometry_.attentionWidth), mixer.decay, mixer.timeBias,
                   f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnLayer], sequence.recurrentOut[gdnLayer],
                   u16(b.recurrent, geometry_.attentionWidth), mixer.mixerNorm,
                   u16(b.gdnHidden, geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows, mixer.outputHeadOrder);
            }
            project(b.gdnHidden, mixer.outputProjection, h.branch);
          } else {
            addAttention(layer, mixer, step.attentionLayer++, true);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
    mix(layer.ffnHyper, b.normalized, h.streams);
    ops::MoE::add(graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
  }
  if (step.gdnLayer != geometry_.stateLayout.layers ||
      step.attentionLayer + geometry_.qwen4.mtpLayers != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
  // The final mix, which the head reads as it is.
  mix(weights_.finalHyper, b.hidden[1], h.streams);
  if (weights_.mtp) {
    // The MTP head's KV at the rows: row r pairs the trunk's streams of the
    // position before (row 0: the sequence's carried streams) with the row's
    // token; each sequence carries its last row's streams.
    const Qwen4MtpWeights &mtp = *weights_.mtp;
    const uint32_t hidden = geometry_.hiddenSize, width = geometry_.qwen4.streamWidth();
    for (const QwenTargetPrefillSequence &sequence : sequences) {
      ops::Qwen4::addMtpInput(graph, u16Rows(b.hidden[0], sequence.rowBegin, sequence.rows, hidden),
                              mtpCarried(sequence.pleHistoryIn),
                              f32Rows(h.streams, sequence.rowBegin, sequence.rows, width), mtp.embedNorm,
                              mtp.hiddenNorm, h.mtpInput, hs, sequence.rows, sequence.rows, sequence.rowBegin, rows);
      ops::Qwen4::addMtpCarry(graph, f32Rows(h.streams, sequence.rowBegin, sequence.rows, width),
                              mtpCarried(sequence.pleHistoryOut), {}, hs, sequence.rows);
    }
    for (uint32_t s = 0; s < hs.streams; ++s)
      project(rowsFrom<uint16_t>(backend_, h.mtpInput, s * rows, 2 * hidden), mtp.embedHidden,
              rowsFrom<uint16_t>(backend_, h.gate, s * rows, hidden));
    ops::Qwen4::addMtpWiden(graph, h.gate, h.mtpStreams, hs, rows);
    mix(mtp.block.mixerHyper, b.normalized, h.mtpStreams);
    addAttention(mtp.block, std::get<QwenAttentionWeights>(mtp.block.mixer), step.attentionLayer, false);
  }
  return b.hidden[1];
}

void Qwen4ExpTarget::addVerify(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &b,
                               std::span<const SplashKvLayer> kvLayers,
                               std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers || b.gdnPacked.size() != geometry_.stateLayout.layers ||
      b.gdnMixed.size() != geometry_.stateLayout.layers || b.gdnDecay.size() != geometry_.stateLayout.layers ||
      b.gdnBeta.size() != geometry_.stateLayout.layers || b.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      b.chunkValues.size() != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("invalid Qwen verify batch");
  VerifyStep step = verifyStep(graph, b, kvLayers, chunks, lanes);

  const Qwen4Buffers &h = b.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape();
  const uint32_t rows = step.rows;
  ops::Qwen4::addInit(graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights_.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape();
      ops::Embedding::add(graph, h.pleSlots, stagedTable(h.pleStaging, geometry_.qwen4.pleHeadDimension), h.pleRows,
                          rows * geometry_.qwen4.pleHeads());
      addVerifyProjection(step, h.pleRows, ple.key, h.gate);
      addVerifyProjection(step, h.pleRows, ple.value, h.pleValue);
      ops::Qwen4::addPleGate(graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm, ple.convolutionNorm,
                             h.pleGated, h.pleConvolution, ps, rows);
      // Each lane's eight rows after its history; the commit writes the
      // history after its retained rows.
      for (uint32_t lane = 0; lane < lanes; ++lane)
        ops::Qwen4::addPleConvolution(
            graph, rowsOf<uint16_t>(backend_, h.pleConvolution, lane * kLaneRows, kLaneRows, ps.width()),
            b.pleHistories[lane], {}, ple.convolution,
            rowsOf<float>(backend_, h.pleGated, lane * kLaneRows, kLaneRows, ps.width()),
            rowsOf<float>(backend_, h.streams, lane * kLaneRows, kLaneRows, ps.width()), ps, kLaneRows);
    }
    addVerifyMix(step, layer.mixerHyper, h.streams, b.normalized);
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            addVerifyProjection(step, b.normalized, mixer.inputProjection, b.gdnPacked[gdnLayer]);
            const ops::PreparedInput hidden = ops::GDN::addDecode(
                graph,
                {b.gdnPacked[gdnLayer], mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates,
                 b.gdnMixed[gdnLayer], mixer.decay, mixer.timeBias, b.gdnDecay[gdnLayer], b.gdnBeta[gdnLayer],
                 mixer.mixerNorm, b.gdnHidden, b.linearScratch},
                geometry_.gdnShape(), lanes, gdnLayer,
                {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()},
                mixer.outputHeadOrder, linear.decodePlan(mixer.outputProjection, lanes).input());
            addVerifyProjection(step, b.gdnHidden, mixer.outputProjection, h.branch, hidden);
          } else {
            addVerifyAttention(step, layer, mixer, step.attentionLayer++);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
    addVerifyMix(step, layer.ffnHyper, h.streams, b.normalized);
    ops::MoE::add(graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
  }
  if (step.gdnLayer != geometry_.stateLayout.layers ||
      step.attentionLayer + geometry_.qwen4.mtpLayers != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
  addVerifyMix(step, weights_.finalHyper, h.streams, b.finalHidden);
  addVerifyProjection(step, b.finalHidden, weights_.logitsProjection, b.logits);
  // The MTP head's KV at the step's rows: row r pairs the trunk's streams of
  // the position before with the row's token. The head reads them as its
  // context at the next steps; the rows past each lane's accepted ones are
  // overwritten before any later step reads them.
  if (weights_.mtp) addMtpVerify(step, b.hidden[0], h.streams, kLaneRows, false);
}

Qwen4ExpTarget::VerifyStep Qwen4ExpTarget::verifyStep(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &b,
                                                      std::span<const SplashKvLayer> kvLayers,
                                                      std::span<const kv::ChunkedPrefillParams> chunks,
                                                      uint32_t lanes) const {
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, b, kvLayers, chunks, lanes, lanes * kLaneRows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  return step;
}

// A decode projection of the step's lanes through its own plan; `prepared`
// describes the scratch table its producer wrote for that plan's input.
void Qwen4ExpTarget::addVerifyProjection(VerifyStep &step, metal::MetalBuffer input,
                                         const ops::Projection &projection, metal::MetalBuffer output,
                                         ops::PreparedInput prepared) const {
  const ops::Linear &linear = operators_.linear();
  linear.add(step.graph,
             {.input = std::move(input), .output = std::move(output), .scratch = step.buffers.linearScratch,
              .prepared = std::move(prepared)},
             projection, linear.decodePlan(projection, step.lanes));
}

void Qwen4ExpTarget::addVerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer streams,
                                  metal::MetalBuffer output) const {
  const Qwen4Buffers &h = step.buffers.qwen4;
  const ops::HyperShape hs = hyperShape();
  ops::Qwen4::addNorm(step.graph, std::move(streams), hyper.norm, hyper.inject, h.normalized, h.weights, hs,
                      step.rows);
  addVerifyProjection(step, h.normalized, hyper.down, h.low);
  ops::Qwen4::addLow(step.graph, h.low, hs, step.rows);
  addVerifyProjection(step, h.low, hyper.up, h.gate);
  ops::Qwen4::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows);
}

// A full-attention block's mixer over the step's rows (the QSA indexer, the
// attention and its output projection) into h.branch.
void Qwen4ExpTarget::addVerifyAttention(VerifyStep &step, const Qwen4ExpLayerWeights &layer,
                                        const QwenAttentionWeights &mixer, uint32_t attentionLayer) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Buffers &h = b.qwen4;
  const ops::Linear &linear = operators_.linear();
  const Qwen4IndexerWeights &indexer = *layer.indexer;
  const uint32_t topBlocks = geometry_.qwen4.indexerTopBlocks;
  const uint32_t rows = step.rows;
  addVerifyProjection(step, b.normalized, mixer.inputProjection, b.fullPacked);
  addIndexerProjection(step.graph, linear, b.normalized, indexer.query, h.indexerQuery, rows);
  addIndexerProjection(step.graph, linear, b.normalized, indexer.key, h.indexerKey, rows);
  bool sparse = false;
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    const uint32_t first = step.chunks[lane].committed_tokens;
    sparse = sparse || ops::Qwen4::qsaSparse(first, kLaneRows, topBlocks);
    const ops::QsaRows qsa{kLaneRows, first, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
    ops::Qwen4::addQsaProject(
        step.graph, rowsOf<float>(backend_, h.indexerQuery, lane * kLaneRows, kLaneRows, kIndexWidth),
        rowsOf<float>(backend_, h.indexerKey, lane * kLaneRows, kLaneRows, QWEN4_QSA_DIMENSION), indexer.queryNorm,
        rowsOf<float>(backend_, b.ropeCos, lane * kLaneRows, kLaneRows, geometry_.rotaryPairs),
        rowsOf<float>(backend_, b.ropeSin, lane * kLaneRows, kLaneRows, geometry_.rotaryPairs), b.pageTables[lane],
        h.qsaIndex[attentionLayer], rowsOf<float>(backend_, h.qsaQueries, lane * kLaneRows, kLaneRows, kIndexWidth),
        qsa);
    const uint64_t blocksBegin = lane * h.qsaBlockLaneBytes;
    ops::Qwen4::addQsaPool(step.graph, backend_.view(h.qsaBlocks, blocksBegin, h.qsaBlockLaneBytes),
                           qsaBlockCount(first, kLaneRows), indexer.keyNorm, h.inverseFrequencies, b.pageTables[lane],
                           h.qsaIndex[attentionLayer]);
  }
  if (sparse)
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      ops::Qwen4::addQsaSelect(
          step.graph, rowsOf<float>(backend_, h.qsaQueries, lane * kLaneRows, kLaneRows, kIndexWidth),
          h.qsaIndex[attentionLayer], b.pageTables[lane], h.qsaScores, h.qsaScoreRows,
          rowsOf<uint32_t>(backend_, h.qsaMask, lane * kLaneRows, kLaneRows, h.qsaMaskWords),
          {kLaneRows, step.chunks[lane].committed_tokens, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32});
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[attentionLayer],
                                           b.chunkValues[attentionLayer], geometry_.attentionQueryHeads,
                                           geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(
      step.graph, step.kvLayers[attentionLayer],
      {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer], b.fullQueries, b.attentionPartials,
       b.attentionStatistics, b.fullAttention, b.pageTables},
      step.chunks, step.attention, ops::QsaMask{h.qsaMask, sparse ? h.qsaMaskWords : 0, 0});
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads, geometry_.kvLayout,
      step.lanes, b.linearScratch, linear.decodePlan(mixer.outputProjection, step.lanes).input());
  addVerifyProjection(step, b.attentionHidden, mixer.outputProjection, h.branch, hidden);
}

// A state cell's carried MTP streams within its auxiliary state, after the
// PLE history.
metal::MetalBuffer Qwen4ExpTarget::mtpCarried(const metal::MetalBuffer &auxiliary) const {
  const uint64_t historyBytes =
      uint64_t{geometry_.qwen4.pleHistory()} * geometry_.qwen4.streamWidth() * sizeof(uint16_t);
  return backend_.view(auxiliary, historyBytes, uint64_t{geometry_.qwen4.streamWidth()} * sizeof(float));
}

// The MTP head over each lane's rows: its inputs (`embedding` rows paired with
// the carried streams and `history`, row r the streams of row r - 1), its
// attention block (writing its KV) and, when `full`, its FFN, its head mix and
// logits. Rows from `limit` on repeat row limit - 1.
void Qwen4ExpTarget::addMtpVerify(VerifyStep &step, metal::MetalBuffer embedding, metal::MetalBuffer history,
                                  uint32_t limit, bool full) const {
  const Qwen4MtpWeights &mtp = *weights_.mtp;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Buffers &h = b.qwen4;
  const ops::HyperShape hs = hyperShape();
  const uint32_t rows = step.rows, hidden = geometry_.hiddenSize, width = geometry_.qwen4.streamWidth();
  for (uint32_t lane = 0; lane < step.lanes; ++lane)
    ops::Qwen4::addMtpInput(step.graph, rowsOf<uint16_t>(backend_, embedding, lane * kLaneRows, kLaneRows, hidden),
                            mtpCarried(b.pleHistories[lane]),
                            rowsOf<float>(backend_, history, lane * kLaneRows, kLaneRows, width), mtp.embedNorm,
                            mtp.hiddenNorm, h.mtpInput, hs, kLaneRows, limit, lane * kLaneRows, rows);
  // The eh projection of each stream's rows, then the fp32 streams.
  for (uint32_t s = 0; s < hs.streams; ++s)
    addVerifyProjection(step, rowsFrom<uint16_t>(backend_, h.mtpInput, s * rows, 2 * hidden), mtp.embedHidden,
                        rowsFrom<uint16_t>(backend_, h.gate, s * rows, hidden));
  ops::Qwen4::addMtpWiden(step.graph, h.gate, h.mtpStreams, hs, rows);
  const uint32_t attentionLayer = geometry_.kvLayout.attentionLayers - geometry_.qwen4.mtpLayers;
  addVerifyMix(step, mtp.block.mixerHyper, h.mtpStreams, b.normalized);
  addVerifyAttention(step, mtp.block, std::get<QwenAttentionWeights>(mtp.block.mixer), attentionLayer);
  if (!full) return;
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  addVerifyMix(step, mtp.block.ffnHyper, h.mtpStreams, b.normalized);
  ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, mtp.block.ffn, *step.moe);
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  addVerifyMix(step, mtp.head, h.mtpStreams, b.finalHidden);
  addVerifyProjection(step, b.finalHidden, weights_.logitsProjection, b.logits);
}

void Qwen4ExpTarget::addMtpDraft(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &b,
                                 std::span<const SplashKvLayer> kvLayers,
                                 std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
                                 uint32_t steps) const {
  if (!weights_.mtp || !lanes || lanes > ExecutionLimits::maximumBatchWidth || !steps ||
      steps > b.proposalsPerLane || b.proposalsPerLane + 1 > kLaneRows || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      b.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      b.chunkValues.size() != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("invalid MTP draft batch");
  VerifyStep step = verifyStep(graph, b, kvLayers, chunks, lanes);
  const Qwen4Buffers &h = b.qwen4;
  // Step j reads rows 0..j-1 (the anchor and the proposals so far, each with
  // the streams of the row before) and drafts proposal j - 1 from row j - 1.
  // The rows before it recompute what the steps before wrote; the later rows
  // repeat it, so they route to its experts.
  for (uint32_t j = 1; j <= steps; ++j) {
    ops::Qwen4::addMtpTokens(graph, b.anchors, b.proposals, h.mtpTokens, lanes, kLaneRows, b.proposalsPerLane, j);
    ops::Embedding::add(graph, h.mtpTokens, weights_.tokenEmbedding, b.hidden[1], step.rows);
    addMtpVerify(step, b.hidden[1], h.mtpStreams, j, true);
    ops::Qwen4::addMtpArgmax(graph, b.logits, b.proposals, b.candidates, lanes, kLaneRows, geometry_.vocabularySize,
                             b.proposalsPerLane, b.candidatesPerProposal, j - 1, j == steps);
  }
}

void Qwen4ExpTarget::addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer logits,
                             uint32_t lanes, ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  linear.add(graph, {.input = std::move(hidden), .output = std::move(logits), .scratch = scratch},
             weights_.logitsProjection, linear.decodePlan(weights_.logitsProjection, lanes));
}

void Qwen4ExpTarget::addStateCommit(metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers,
                                    uint32_t lanes) const {
  const ops::PleShape shape = pleShape();
  const uint64_t laneBytes = uint64_t{kLaneRows} * shape.width() * sizeof(uint16_t);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const metal::MetalBuffer retained =
        backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t));
    ops::Qwen4::addPleCommit(graph, backend_.view(buffers.pleConvolution, lane * laneBytes, laneBytes),
                             buffers.pleHistoriesIn[lane], buffers.pleHistoriesOut[lane], retained, shape);
    // The streams of the last retained row, which the next cycle's first row pairs with its anchor.
    if (weights_.mtp)
      ops::Qwen4::addMtpCarry(graph,
                              rowsOf<float>(backend_, buffers.streams, lane * kLaneRows, kLaneRows,
                                            geometry_.qwen4.streamWidth()),
                              mtpCarried(buffers.pleHistoriesOut[lane]), retained, hyperShape(), kLaneRows);
  }
}

} // namespace splash::model
