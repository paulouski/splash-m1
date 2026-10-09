#pragma once

// Qwen3.8-Flash-Next (llama.cpp's qwen4exp, transformers' qwen4_exp): the
// Qwen3.5 hybrid of GDN and gated full attention with a sparse MoE FFN, and
// three additions.
//  - Hyper-connections replace every norm: the residual is hyperConnections
//    parallel streams of hiddenSize values. Before each mixer and each FFN a
//    mix (Qwen4HyperWeights) normalizes every stream, gates it by a low-rank
//    projection of all of them and averages the streams into the block's
//    input; the block's output is added back to every stream scaled by its
//    own injection weight. A last mix without injection feeds the head; the
//    model has no output norm.
//  - QSA sparse attention: the full-attention layers attend, past the first
//    indexerTopBlocks blocks of indexerBlockTokens tokens, to the
//    indexerTopBlocks blocks an indexer scores highest and the current block.
//  - A PLE n-gram embedding feeds one GDN layer: the 2- and 3-gram hashes of
//    every token pick rows of a table of hundreds of millions of rows, which
//    stays in its GGUF shard (model/PleStore.hpp reads it in place) and gate
//    a value added to every stream through a dilated convolution.
// Only GGUF sources (unsloth's) are read.

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "model/PleStore.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/Vision.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace splash::model {

struct Qwen4ExpLayout final : QwenHybridLayout<1> {
  static constexpr QwenFfnKind ffnKind = QwenFfnKind::SparseMoe;
  // Never packaged or read from MLX: only its GGUF loads (ModelFactory.cpp).
  static constexpr bool ggufOnly = true;
  // Never packaged; the names keep PackedTargetFiles well-formed.
  static constexpr std::string_view layerMagic = "MDQ40001";
  static constexpr std::string_view headMagic = "MDQ40002";

  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  // Residual streams, and the rank of each mix's low-rank gate.
  uint32_t hyperConnections = 4;
  uint32_t hyperRank = 320;
  // The QSA indexer's query heads and their dimension, and the attention's
  // selection: topBlocks blocks of blockTokens tokens.
  uint32_t indexerHeads = 4;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerTopBlocks = 512;
  uint32_t indexerBlockTokens = 4;
  // The one layer with a PLE n-gram embedding: n-grams of up to pleNgram
  // tokens, pleHeadsPerNgram hash heads per n-gram size from 2 up, rows of
  // pleHeadDimension values, a convolution of pleConvolutionTaps taps, and the
  // end-of-sequence token that cuts an n-gram.
  uint32_t pleLayer = 1;
  uint32_t pleNgram = 3;
  uint32_t pleHeadsPerNgram = 8;
  uint32_t pleHeadDimension = 160;
  uint32_t pleConvolutionTaps = 4;
  uint32_t pleEosToken = 248044;
  // The MTP head's full-attention blocks (0 or 1); its KV is the last
  // attention layer of the KV pages.
  uint32_t mtpLayers = 0;

  constexpr Qwen4ExpLayout()
      : QwenHybridLayout{.maximumContextTokens = 262'144,
                         .layers = 48,
                         .hiddenSize = 2560,
                         .vocabularySize = 248320,
                         // qkv 10240 | z 6144 | beta, alpha 96, to whole 64-column tiles.
                         .packedGdnWidth = 16512,
                         .packedFullWidth = 13312,
                         .convolutionDimension = 10240,
                         .gdnKeyHeads = 16,
                         .gdnValueHeads = 48,
                         .gdnHeadDimension = 128,
                         .attentionWidth = 6144,
                         .attentionQueryHeads = 24,
                         .attentionKvHeads = 2,
                         .attentionHeadDimension = 256,
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248077,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {47}} {}

  // A residual row: every stream of a position.
  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept { return hyperConnections * hiddenSize; }
  // The PLE rows a position gathers: pleHeadsPerNgram heads for each n-gram
  // size from 2 to pleNgram, and their values.
  [[nodiscard]] constexpr uint32_t pleHeads() const noexcept { return (pleNgram - 1) * pleHeadsPerNgram; }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept { return pleHeads() * pleHeadDimension; }
  // Rows of the PLE convolution's history: (taps - 1) taps pleNgram rows apart.
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept { return (pleConvolutionTaps - 1) * pleNgram; }
  // Every page also holds the QSA indexer's keys; every state cell the PLE
  // convolution's history (bf16 rows of every stream) and, with the MTP head,
  // the streams of the position before the next one it reads (fp32).
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    kv::Layout layout = QwenHybridLayout<1>::kvLayout();
    layout.attentionLayers += mtpLayers;
    layout.indexDimension = indexerHeadDimension;
    return layout;
  }
  [[nodiscard]] constexpr uint64_t pleHistoryBytes() const noexcept {
    return uint64_t{pleHistory()} * streamWidth() * sizeof(uint16_t);
  }
  [[nodiscard]] constexpr uint64_t mtpCarriedBytes() const noexcept {
    return mtpLayers ? uint64_t{streamWidth()} * sizeof(float) : 0;
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    GdnStateLayout layout = QwenHybridLayout<1>::gdnStateLayout();
    layout.auxiliaryBytes = pleHistoryBytes() + mtpCarriedBytes();
    return layout;
  }
  // Tokens the indexer selects.
  [[nodiscard]] constexpr uint32_t indexerTokens() const noexcept { return indexerTopBlocks * indexerBlockTokens; }
  bool operator==(const Qwen4ExpLayout &) const = default;
};

// The vision tower of Qwen3.8-Flash-Next, projecting into its hidden width.
inline constexpr ops::VisionLayout kQwen4ExpVisionLayout{.outputHiddenSize = Qwen4ExpLayout{}.hiddenSize};

// One hyper-connection mix: the streams' norm (streamWidth F32 multipliers),
// the low-rank gate's down [rank, streamWidth] and up [streamWidth, rank]
// projections, and, for a mix that injects its block's output back, the
// injection weights [hyperConnections][streamWidth] F32.
struct Qwen4HyperWeights final {
  ops::NormWeights norm;
  ops::Projection down;
  ops::Projection up;
  metal::MetalBuffer inject;
};

// The QSA indexer of a full-attention layer: its query [heads * dim, hidden]
// and key [dim, hidden] projections (F32, widened from the GGUF's BF16) and
// their norms.
struct Qwen4IndexerWeights final {
  ops::Projection query;
  ops::Projection key;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
};

// The PLE layer's projections of the gathered n-gram rows, its norms (each
// streamWidth F32 multipliers) and its dilated convolution's taps
// ([streamWidth][taps] F32).
struct Qwen4PleWeights final {
  ops::Projection key;
  ops::Projection value;
  ops::NormWeights keyNorm;
  ops::NormWeights queryNorm;
  ops::NormWeights convolutionNorm;
  metal::MetalBuffer convolution;
};

struct Qwen4ExpLayerWeights final {
  Qwen4HyperWeights mixerHyper;
  QwenMixerWeights mixer;
  std::optional<Qwen4IndexerWeights> indexer;
  Qwen4HyperWeights ffnHyper;
  ops::MoeWeights ffn;
  std::optional<Qwen4PleWeights> ple;
};

// The MTP head: a position's input is the trunk's residual streams at the
// position before (hiddenNorm per stream) and the position's token embedding
// (embedNorm), each stream through embedHidden ([e ; h_s] -> hidden); then one
// full-attention block of the trunk's form, whose output streams feed the next
// draft step, and the head's mix, whose output the trunk's vocabulary
// projection reads.
struct Qwen4MtpWeights final {
  Qwen4ExpLayerWeights block;
  ops::Projection embedHidden;
  ops::NormWeights embedNorm;
  ops::NormWeights hiddenNorm;
  Qwen4HyperWeights head;
};

// The PLE table, which stays in its GGUF shard: where row 0 is and the hash
// that picks a token's rows (PleSource and PleHasher read them).
struct Qwen4PleTable final {
  std::filesystem::path path;
  uint64_t dataOffset = 0;
  uint64_t rows = 0;
  PleHashParams hash;
};

struct Qwen4ExpWeights final : QwenTargetWeights<Qwen4ExpLayout, Qwen4ExpLayerWeights> {
  // The last mix, which feeds the head (finalNorm is unused).
  Qwen4HyperWeights finalHyper;
  Qwen4PleTable pleTable;
  std::optional<Qwen4MtpWeights> mtp;
};

class GgufMtpLoader;
// Loads the target and, with `mtp`, its MTP head (layout.mtpLayers).
[[nodiscard]] Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend, Qwen4ExpLayout layout,
                                                  const QwenTargetFiles<Qwen4ExpLayout> &files,
                                                  GgufMtpLoader *mtp = nullptr);

} // namespace splash::model
