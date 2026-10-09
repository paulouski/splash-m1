#pragma once

// What Qwen3.8-Flash-Next's graphs (Qwen4ExpTarget.hpp) bind beyond the Qwen
// hybrid's: its dimensions as the geometry carries them, and the scratch of a
// prefill or verify step. QwenTarget.hpp's buffer structs hold the scratch,
// empty for the other families.

#include "metal/MetalBackend.hpp"
#include "ops/Qwen4.hpp"

#include <cstdint>
#include <span>

namespace splash::model {

struct Qwen4Geometry final {
  // Residual streams; zero for every other family.
  uint32_t hyperConnections = 0;
  uint32_t hyperRank = 0;
  uint32_t pleLayer = 0;
  uint32_t pleNgram = 0;
  uint32_t pleHeadsPerNgram = 0;
  uint32_t pleHeadDimension = 0;
  uint32_t pleConvolutionTaps = 0;
  uint32_t indexerTopBlocks = 0;
  uint32_t hiddenSize = 0;
  uint32_t maximumContextTokens = 0;
  // The MTP head's full-attention blocks (0 or 1), the last of the KV layers.
  uint32_t mtpLayers = 0;

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return hyperConnections != 0; }
  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept { return hyperConnections * hiddenSize; }
  [[nodiscard]] constexpr uint32_t pleHeads() const noexcept { return (pleNgram - 1) * pleHeadsPerNgram; }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept { return pleHeads() * pleHeadDimension; }
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept { return (pleConvolutionTaps - 1) * pleNgram; }
  // Words of a row's QSA block bitmap: one bit per block of four tokens of
  // the context, the tail block included.
  [[nodiscard]] constexpr uint32_t qsaMaskWords() const noexcept { return hyperConnections ? (maximumContextTokens / 4 + 2 + 31) / 32 : 0; }
};

// A step's scratch of `rows` rows (prefill or verify): the fp32 residual
// streams, a mix's normalized streams, low-rank and gate rows and injection
// weights, the zero rows the MoE adds its output to, the PLE rows (the
// runtime's family hook stages the table rows a step gathers, and each row's
// slot indices into them), projections, gated values and convolution inputs,
// and the QSA indexer's scratch.
struct Qwen4Buffers final {
  metal::MetalBuffer streams;
  metal::MetalBuffer normalized;
  metal::MetalBuffer low;
  metal::MetalBuffer gate;
  metal::MetalBuffer weights;
  metal::MetalBuffer branch;
  metal::MetalBuffer zeros;
  // The staged PLE rows (IQ4_NL rows of pleHeadDimension values), the slot of
  // each (row, head) among them, and the rows gathered through them.
  metal::MetalBuffer pleStaging;
  metal::MetalBuffer pleSlots;
  metal::MetalBuffer pleRows;
  metal::MetalBuffer pleValue;
  metal::MetalBuffer pleGated;
  metal::MetalBuffer pleConvolution;
  // QSA: the indexer's projections (fp32), its prepared queries, the score
  // scratch of qsaScoreRows rows, the step's bitmaps of qsaMaskWords words per
  // row, the RoPE inverse frequencies, the block records the family hook
  // wrote (Qwen4QsaBlock: a prefill sequence's from record rowBegin on, a
  // decode lane's from byte lane * qsaBlockLaneBytes) and where each
  // full-attention layer's index tensors sit in the KV extents.
  metal::MetalBuffer indexerQuery;
  metal::MetalBuffer indexerKey;
  metal::MetalBuffer qsaQueries;
  metal::MetalBuffer qsaScores;
  metal::MetalBuffer qsaMask;
  metal::MetalBuffer qsaBlocks;
  metal::MetalBuffer inverseFrequencies;
  std::span<const ops::QsaIndex> qsaIndex;
  uint64_t qsaBlockLaneBytes = 0;
  uint32_t qsaScoreRows = 0;
  uint32_t qsaMaskWords = 0;
  // The MTP head: its fp32 streams, the bf16 inputs of its eh projection
  // (rows of 2 * hidden per stream) and a draft step's tokens.
  metal::MetalBuffer mtpStreams;
  metal::MetalBuffer mtpInput;
  metal::MetalBuffer mtpTokens;
};

} // namespace splash::model
