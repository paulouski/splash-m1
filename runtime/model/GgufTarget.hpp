#pragma once

// Source adapter for a Qwen GGUF. Preparation writes immutable cached files;
// serving uses the same read-only WeightFile mappings as packaged weights.

#include <deque>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedFiles.hpp"
#include "model/PrismMlx.hpp"
#include "model/QwenHybridLayout.hpp"

namespace splash::model {

// The .gguf files of a target directory: its one GGUF, or every part of a
// split one (NAME-0000K-of-0000N.gguf, K = 1..N) in split order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the metadata of the GGUF, whose files are `paths`.
  GgufTargetLoader(metal::MetalBackend &backend, const std::vector<std::filesystem::path> &paths,
                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion);
  GgufTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
      : GgufTargetLoader(backend, std::vector{path}, geometry, std::move(admitConversion)) {}
  // The same target read from a Prism ML Hadamard MLX checkpoint, presented as its rotated PQ2_0 GGUF.
  GgufTargetLoader(metal::MetalBackend &backend, const PrismMlxDirectory &directory,
                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion);
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  // Every image's cache identity and size, layers first, for the model's
  // disk check before the first image is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept { return images_.weights(); }
  // Writes every missing image and maps none.
  void prepare();

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // A qwen4exp target's PLE table, read in place from its shard and not
  // prepared; empty for any other.
  [[nodiscard]] const std::optional<gguf::PleTable> &ple() const noexcept { return ple_; }
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // Plans every image of file; a Prism MLX source keys them by its identity.
  void plan(const GgufFile &file, const gguf::TargetGeometry &geometry, std::string_view prismIdentity);

  void checkUnchanged() const;

  metal::MetalBackend &backend_;
  std::deque<WeightSource> files_;
  std::vector<const WeightSource *> sources_; // files_, which the images name by index
  std::optional<GgufRotation> rotation_;
  std::optional<gguf::PleTable> ple_;
  PreparedImages<gguf::Image> images_; // layers, head, embedding
};

// A qwen4exp MTP head's GGUF (mtp/ in the model root): one image, planned,
// prepared and read as a target's (gguf::planMtpImage).
class GgufMtpLoader final {
public:
  GgufMtpLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                const gguf::TargetGeometry &geometry, PreparationCheck admitConversion);
  GgufMtpLoader(const GgufMtpLoader &) = delete;
  GgufMtpLoader &operator=(const GgufMtpLoader &) = delete;

  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept { return images_.weights(); }
  void prepare();
  [[nodiscard]] WeightFile open();

private:
  metal::MetalBackend &backend_;
  WeightSource source_;
  PreparedImages<gguf::Image> images_;
};

// The GGUF geometry of a Qwen layout with its family's dense or sparse MoE
// FFN.
template <class Layout>
[[nodiscard]] gguf::TargetGeometry ggufTargetGeometry(const Layout &layout) {
  gguf::TargetGeometry geometry;
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.gdnKeyHeads = layout.gdnKeyHeads;
  geometry.gdnValueHeads = layout.gdnValueHeads;
  geometry.gdnHeadDimension = layout.gdnHeadDimension;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.rotaryPairs = layout.rotaryPairs;
  geometry.rotaryTheta = layout.rotaryTheta;
  geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
  if constexpr (requires { layout.hyperConnections; }) { // Qwen3.8-Flash-Next
    geometry.hyperConnections = layout.hyperConnections;
    geometry.hyperRank = layout.hyperRank;
    geometry.indexerHeads = layout.indexerHeads;
    geometry.indexerHeadDimension = layout.indexerHeadDimension;
    geometry.indexerBlockTokens = layout.indexerBlockTokens;
    geometry.indexerTopBlocks = layout.indexerTopBlocks;
    geometry.pleLayer = layout.pleLayer;
    geometry.pleNgram = layout.pleNgram;
    geometry.pleHeadsPerNgram = layout.pleHeadsPerNgram;
    geometry.pleHeadDimension = layout.pleHeadDimension;
    geometry.pleConvolutionTaps = layout.pleConvolutionTaps;
  }
  if constexpr (Layout::ffnKind == QwenFfnKind::SparseMoe) {
    geometry.experts = layout.experts;
    geometry.expertsPerToken = layout.expertsPerToken;
    geometry.expertIntermediateSize = layout.expertIntermediateSize;
  } else {
    geometry.intermediateSize = layout.intermediateSize;
  }
  return geometry;
}

} // namespace splash::model
