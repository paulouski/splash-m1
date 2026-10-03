#pragma once

#include "model/AffinePreparation.hpp"
#include "model/PreparedFiles.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <memory>
#include <span>
#include <vector>

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_6MoeLayout;
class SafetensorsCheckpoint;

// Native MLX affine source -> the existing packed target ABI. Both this adapter
// and the block-quantized adapter publish through PreparedWeights and serve
// through WeightFile; neither changes inference kernels. The checkpoint is
// planned once; each image is prepared when it is opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_8Layout &layout, PreparationCheck admitConversion);
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_6MoeLayout &layout, PreparationCheck admitConversion);
  ~AffineTargetLoader();
  // Every image's cache identity and size, layers first, for the model's
  // disk check before the first image is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept;
  // Writes every missing image and maps none.
  void prepare();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Every planned image of a layout, its sections at their offsets: the layers,
// the head, the embedding. Sized at the default (all-4-bit) storage: a
// checkpoint with a per-tensor Q5 override needs more than this once prepared
// (Q5 adds 8 B per 64 values); callers before a checkpoint is open accept
// that underestimate, callers with one use the overloads below.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout);
// source's real per-tensor bits.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout,
                                                            const SafetensorsCheckpoint &source);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout,
                                                            const SafetensorsCheckpoint &source);
// The planned image of target layer `layer`: its sections at their offsets,
// at the default (all-4-bit) storage.
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_8Layout &layout, uint32_t layer);
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_6MoeLayout &layout, uint32_t layer);
// source's real per-tensor bits.
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_8Layout &layout, uint32_t layer,
                                             const SafetensorsCheckpoint &source);
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_6MoeLayout &layout, uint32_t layer,
                                             const SafetensorsCheckpoint &source);

} // namespace splash::model
