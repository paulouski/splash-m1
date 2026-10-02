#pragma once

// A Prism ML Hadamard MLX checkpoint (prism_hadamard_qwen35, schema 2: affine
// 2-bit, group 128) presented as the rotated PQ2_0 GGUF it was folded from, so
// the GGUF planner, repack and rotation load it unchanged. Bytes are
// synthesized on read: a PQ2_0 block is the group's F16 scale then the 32
// bytes of its 128 two-bit codes, which are the checkpoint's packed words
// as stored; norms and GDN tensors keep their F32 values. The checkpoint's
// GDN value heads are already grouped.

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace splash::model {

// The target directory of a Prism MLX checkpoint, for GgufTargetLoader.
struct PrismMlxDirectory {
  std::filesystem::path path;
};

// Whether directory's config.json names a Prism Hadamard checkpoint.
[[nodiscard]] bool isPrismMlx(const std::filesystem::path &directory);
// Throws unless directory holds the one layout this loader reads: schema 2,
// affine 2-bit/group-128, and a hadamard.json.
void requirePrismMlxConfig(const std::filesystem::path &directory);
// The one weight file, which binds to the GGUF view.
[[nodiscard]] std::filesystem::path prismMlxFile(const std::filesystem::path &directory);

struct PrismMlxView {
  GgufMemoryHeader header;
  // Digest of what the synthesis reads besides the tensor data: the
  // safetensors header and hadamard.json.
  std::string identity;
};

// Validates the checkpoint in directory against geometry (metadata, every
// tensor's dtype and shape, the signs, no unexpected tensor) and makes
// source's readData synthesize the GGUF tensors the returned header names.
[[nodiscard]] PrismMlxView bindPrismMlx(WeightSource &source, const std::filesystem::path &directory,
                                        const gguf::TargetGeometry &geometry);

// The cache key of an image of a bound source, from the GGUF planner's key.
[[nodiscard]] std::string prismMlxKey(std::string_view plannedKey, std::string_view identity);

} // namespace splash::model
