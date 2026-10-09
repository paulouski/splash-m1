#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedWeights.hpp"

#include <span>

namespace splash::model {

// The files of a GGUF, one or a split GGUF's parts in order (GgufFile).
using GgufSources = std::span<const WeightSource *const>;

// The identity of a planned image of source: the preparation identity, the
// whole plan and the bytes and type of every tensor it reads.
[[nodiscard]] PreparedWeight ggufImageWeight(GgufSources sources, const gguf::Image &image);

// Writes a planned image into its preallocated, zeroed destination file: the
// header and descriptors, the copied rows and the planes the GPU repacks.
// Staging stays within kWeightPreparationStagingBytes whatever the tensor,
// layer or expert count; admit runs before each chunk.
void writeGgufImage(metal::MetalBackend &backend, GgufSources sources, int destination,
                    const gguf::Image &image, const PreparationCheck &admit);

// The same for a GGUF of one file.
[[nodiscard]] inline PreparedWeight ggufImageWeight(const WeightSource &source, const gguf::Image &image) {
  const WeightSource *sources[] = {&source};
  return ggufImageWeight(sources, image);
}
inline void writeGgufImage(metal::MetalBackend &backend, const WeightSource &source, int destination,
                           const gguf::Image &image, const PreparationCheck &admit) {
  const WeightSource *sources[] = {&source};
  writeGgufImage(backend, sources, destination, image, admit);
}

} // namespace splash::model
