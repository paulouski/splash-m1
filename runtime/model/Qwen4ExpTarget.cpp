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
  const auto mix = [&](const Qwen4HyperWeights &hyper, metal::MetalBuffer output) {
    ops::Qwen4::addNorm(graph, h.streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs, rows);
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
    mix(layer.mixerHyper, b.normalized);
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
            const uint32_t attentionLayer = step.attentionLayer++;
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
              if (sparse)
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
              ops::PagedAttention::addPrefill(
                  graph, kvLayers[attentionLayer], attentionQueries, attentionRows, b.attentionPartials,
                  b.attentionStatistics, sequence.pageTable, sequence.chunk, step.attention[s],
                  ops::QsaMask{h.qsaMask, sparse ? h.qsaMaskWords : 0, sequence.rowBegin});
              ops::PagedAttention::addPrefillGate(
                  graph, u16Rows(b.fullPacked, sequence.rowBegin, sequence.rows, geometry_.packedFullWidth),
                  attentionRows, u16Rows(b.attentionHidden, sequence.rowBegin, sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
            }
            project(b.attentionHidden, mixer.outputProjection, h.branch);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
    mix(layer.ffnHyper, b.normalized);
    ops::MoE::add(graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
  }
  if (step.gdnLayer != geometry_.stateLayout.layers || step.attentionLayer != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
  // The final mix, which the head reads as it is.
  mix(weights_.finalHyper, b.hidden[1]);
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
  const uint32_t rows = lanes * kLaneRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, b, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);

  const Qwen4Buffers &h = b.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape();
  // A decode projection of `lanes` lanes through its own plan; `prepared`
  // describes the scratch table its producer wrote for that plan's input.
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output,
                           ops::PreparedInput prepared = {}) {
    linear.add(graph,
               {.input = std::move(input), .output = std::move(output), .scratch = b.linearScratch,
                .prepared = std::move(prepared)},
               projection, linear.decodePlan(projection, lanes));
  };
  const auto mix = [&](const Qwen4HyperWeights &hyper, metal::MetalBuffer output) {
    ops::Qwen4::addNorm(graph, h.streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs, rows);
    project(h.normalized, hyper.down, h.low);
    ops::Qwen4::addLow(graph, h.low, hs, rows);
    project(h.low, hyper.up, h.gate);
    ops::Qwen4::addMix(graph, h.normalized, h.gate, std::move(output), hs, rows);
  };
  ops::Qwen4::addInit(graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights_.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape();
      ops::Embedding::add(graph, h.pleSlots, stagedTable(h.pleStaging, geometry_.qwen4.pleHeadDimension), h.pleRows,
                          rows * geometry_.qwen4.pleHeads());
      project(h.pleRows, ple.key, h.gate);
      project(h.pleRows, ple.value, h.pleValue);
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
    mix(layer.mixerHyper, b.normalized);
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            project(b.normalized, mixer.inputProjection, b.gdnPacked[gdnLayer]);
            const ops::PreparedInput hidden = ops::GDN::addDecode(
                graph,
                {b.gdnPacked[gdnLayer], mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates,
                 b.gdnMixed[gdnLayer], mixer.decay, mixer.timeBias, b.gdnDecay[gdnLayer], b.gdnBeta[gdnLayer],
                 mixer.mixerNorm, b.gdnHidden, b.linearScratch},
                geometry_.gdnShape(), lanes, gdnLayer,
                {geometry_.stateLayout.convolutionLayerBytes(), geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()},
                mixer.outputHeadOrder, linear.decodePlan(mixer.outputProjection, lanes).input());
            project(b.gdnHidden, mixer.outputProjection, h.branch, hidden);
          } else {
            const uint32_t attentionLayer = step.attentionLayer++;
            const Qwen4IndexerWeights &indexer = *layer.indexer;
            const uint32_t topBlocks = geometry_.qwen4.indexerTopBlocks;
            project(b.normalized, mixer.inputProjection, b.fullPacked);
            addIndexerProjection(graph, linear, b.normalized, indexer.query, h.indexerQuery, rows);
            addIndexerProjection(graph, linear, b.normalized, indexer.key, h.indexerKey, rows);
            bool sparse = false;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              const uint32_t first = chunks[lane].committed_tokens;
              sparse = sparse || ops::Qwen4::qsaSparse(first, kLaneRows, topBlocks);
              const ops::QsaRows qsa{kLaneRows, first, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32};
              ops::Qwen4::addQsaProject(
                  graph, rowsOf<float>(backend_, h.indexerQuery, lane * kLaneRows, kLaneRows, kIndexWidth),
                  rowsOf<float>(backend_, h.indexerKey, lane * kLaneRows, kLaneRows, QWEN4_QSA_DIMENSION),
                  indexer.queryNorm,
                  rowsOf<float>(backend_, b.ropeCos, lane * kLaneRows, kLaneRows, geometry_.rotaryPairs),
                  rowsOf<float>(backend_, b.ropeSin, lane * kLaneRows, kLaneRows, geometry_.rotaryPairs),
                  b.pageTables[lane], h.qsaIndex[attentionLayer],
                  rowsOf<float>(backend_, h.qsaQueries, lane * kLaneRows, kLaneRows, kIndexWidth), qsa);
              const uint64_t blocksBegin = lane * h.qsaBlockLaneBytes;
              ops::Qwen4::addQsaPool(graph,
                                     backend_.view(h.qsaBlocks, blocksBegin, h.qsaBlockLaneBytes),
                                     qsaBlockCount(first, kLaneRows), indexer.keyNorm, h.inverseFrequencies,
                                     b.pageTables[lane], h.qsaIndex[attentionLayer]);
            }
            if (sparse)
              for (uint32_t lane = 0; lane < lanes; ++lane)
                ops::Qwen4::addQsaSelect(
                    graph, rowsOf<float>(backend_, h.qsaQueries, lane * kLaneRows, kLaneRows, kIndexWidth),
                    h.qsaIndex[attentionLayer], b.pageTables[lane], h.qsaScores, h.qsaScoreRows,
                    rowsOf<uint32_t>(backend_, h.qsaMask, lane * kLaneRows, kLaneRows, h.qsaMaskWords),
                    {kLaneRows, chunks[lane].committed_tokens, topBlocks, h.qsaMaskWords, h.qsaMaskWords * 32});
            ops::PagedAttention::addVerifyProjection(graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                                     b.ropeSin, b.fullQueries, b.chunkKeys[attentionLayer],
                                                     b.chunkValues[attentionLayer], geometry_.attentionQueryHeads,
                                                     geometry_.kvLayout, lanes);
            ops::PagedAttention::addVerify(
                graph, kvLayers[attentionLayer],
                {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer], b.fullQueries, b.attentionPartials,
                 b.attentionStatistics, b.fullAttention, b.pageTables},
                chunks, step.attention, ops::QsaMask{h.qsaMask, sparse ? h.qsaMaskWords : 0, 0});
            const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
                graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
                geometry_.kvLayout, lanes, b.linearScratch,
                linear.decodePlan(mixer.outputProjection, lanes).input());
            project(b.attentionHidden, mixer.outputProjection, h.branch, hidden);
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
    mix(layer.ffnHyper, b.normalized);
    ops::MoE::add(graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe);
    ops::Qwen4::addCombine(graph, h.streams, h.branch, h.weights, hs, rows);
  }
  if (step.gdnLayer != geometry_.stateLayout.layers || step.attentionLayer != geometry_.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
  mix(weights_.finalHyper, b.finalHidden);
  project(b.finalHidden, weights_.logitsProjection, b.logits);
}

void Qwen4ExpTarget::addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer logits,
                             uint32_t lanes, ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  linear.add(graph, {.input = std::move(hidden), .output = std::move(logits), .scratch = scratch},
             weights_.logitsProjection, linear.decodePlan(weights_.logitsProjection, lanes));
}

void Qwen4ExpTarget::addPleCommit(metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers,
                                  uint32_t lanes) const {
  const ops::PleShape shape = pleShape();
  const uint64_t laneBytes = uint64_t{kLaneRows} * shape.width() * sizeof(uint16_t);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    ops::Qwen4::addPleCommit(graph, backend_.view(buffers.pleConvolution, lane * laneBytes, laneBytes),
                             buffers.pleHistoriesIn[lane], buffers.pleHistoriesOut[lane],
                             backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)),
                             shape);
}

} // namespace splash::model
