#pragma once

// The layout of a prepared GGUF target image: the planner writes it, the
// weight store reads it. After the 16-byte file header (model/WeightLayout.hpp)
// come 16 KiB-aligned sections. A quantized tensor is a descriptor section,
// then its plane0, optional plane1 and meta sections in its format's layout
// (metal/abi/QuantFormat.h). A tensor copied as stored (the token
// embedding's native rows; the F32 MoE router, shared-expert gate and
// alpha/beta) is a descriptor and its rows; a norm, the convolution and the
// GDN head vectors are their rows alone, F32 or narrowed to the bf16 values
// they equal.
// Editing this file re-prepares every GGUF model.

#include "metal/abi/QuantFormat.h"

#include <bit>
#include <cstdint>
#include <string_view>

namespace splash::model {

inline constexpr std::string_view kGgufImageMagic = "MDGG0001";

// Columns of a superblock: a staging unit wide enough for every format.
inline constexpr uint64_t kGgufBlockColumns = 256;

// The columns a quantized tensor's width in `format` is a multiple of: one
// 64-input span for formats of blocks of up to 64 elements, else the 256-input
// superblock (the register decode tiles decode four spans of a format of
// several groups per meta unit together, and PQ2_0's 128-input blocks too).
[[nodiscard]] inline constexpr uint64_t ggufColumnUnit(const QuantFormat &format) {
  return format.block_elements > 64 ? kGgufBlockColumns : 64;
}

// The rows an image holds for a tensor of `rows`: whole plane tiles, the rows
// past the tensor's zero. The descriptor keeps the tensor's own rows.
[[nodiscard]] inline constexpr uint64_t ggufStorageRows(uint64_t rows) {
  return (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
}

// A tensor's descriptor section, little-endian.
struct GgufTensorDescriptor {
  uint32_t type;       // ggml type
  uint32_t outputSize; // rows
  uint32_t inputSize;  // columns
  uint32_t p0, p1;     // plane0 and plane1 bytes per 32 columns of a row
  uint32_t metaBytes;  // meta bytes per metaGroups 32-column groups of a row
  uint32_t metaGroups;
  uint32_t reserved;
  uint64_t plane0Bytes, plane1Bytes, metaTotalBytes;
  uint64_t unused;
};
static_assert(sizeof(GgufTensorDescriptor) == 64 && std::endian::native == std::endian::little,
              "a GGUF tensor descriptor is 64 little-endian bytes");

struct GgufPlaneBytes {
  uint64_t plane0, plane1, meta;
};

// The planes of a [rows, columns] tensor in format; rows are the image's
// (ggufStorageRows).
[[nodiscard]] inline constexpr GgufPlaneBytes ggufPlaneBytes(const QuantFormat &format, uint64_t rows,
                                                             uint64_t columns) {
  const uint64_t groups = columns / 32;
  return {rows * groups * format.plane0_bytes, rows * groups * format.plane1_bytes,
          rows * (groups / format.meta_groups) * format.meta_bytes};
}

// Bytes of `columns` values of a row in format's GGUF blocks.
[[nodiscard]] inline constexpr uint64_t ggufRowBytes(const QuantFormat &format, uint64_t columns) {
  return columns / format.block_elements * format.block_bytes;
}

} // namespace splash::model
