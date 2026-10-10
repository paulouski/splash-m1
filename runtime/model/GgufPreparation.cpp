// Editing this file re-prepares every GGUF model.
#include "WeightPreparationIdentity.hpp"
#include "model/GgufPreparation.hpp"

#include "metal/abi/GgufRepack.h"
#include "model/Bfloat16.hpp"
#include "model/GgufImageLayout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>

namespace splash::model {
namespace {

// A chunk's rows, columns and plane bytes are within the staging bound, so
// they fit the repack kernel's 32-bit parameters.
static_assert(kWeightPreparationStagingBytes <= std::numeric_limits<uint32_t>::max());

// The source row image row `row` of `rows` is read from.
uint64_t sourceRow(const gguf::TensorRows &rows, uint64_t row) {
  const gguf::RowOrder &order = rows.order;
  if (row < order.from) return row;
  if (!order.headRows || !order.keyHeads || !order.valueHeadsPerKey)
    throw GgufError("invalid weight row permutation");
  const uint64_t head = (row - order.from) / order.headRows;
  const uint64_t source = order.from +
      ((head % order.valueHeadsPerKey) * order.keyHeads + head / order.valueHeadsPerKey) * order.headRows +
      (row - order.from) % order.headRows;
  if (source >= rows.rows) throw GgufError("weight row permutation is out of bounds");
  return source;
}

void requireRange(uint64_t offset, uint64_t bytes, uint64_t available) {
  if (offset > available || bytes > available - offset)
    throw GgufError("prepared weight section is out of bounds");
}

// Image rows [first, first + count) of `rows`, bytes [column, column + span)
// of each, back to back. Consecutive source rows are read together.
void readRows(GgufSources sources, const gguf::TensorRows &rows, uint64_t first,
              uint64_t count, uint64_t column, uint64_t span, uint8_t *to) {
  if (first > rows.rows || count > rows.rows - first || column > rows.rowBytes || span > rows.rowBytes - column)
    throw GgufError("prepared weight rows are out of bounds");
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (span == rows.rowBytes)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    if (rows.file >= sources.size()) throw GgufError("prepared weight rows name a missing GGUF file");
    sources[rows.file]->readData(rows.offset + start * rows.rowBytes + column, {to + row * span, run * span});
    row += run;
  }
}

// F32 values, narrowed in place to the bf16 values they equal: the kernels
// read these tensors as bf16, and preparation never rounds a weight.
std::span<uint8_t> narrowToBfloat16(std::span<uint8_t> bytes, const std::string &name) {
  const uint64_t count = bytes.size() / 4;
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    std::memcpy(&value, bytes.data() + 4 * i, 4);
    const std::optional<uint16_t> narrowed = exactBfloat16(value);
    if (!narrowed) throw GgufError(name + " is not bf16-exact; it needs an F32 path");
    std::memcpy(bytes.data() + 2 * i, &*narrowed, 2);
  }
  return bytes.first(2 * count);
}

// BF16 values widened in place to the F32 values they equal: bytes holds
// the BF16 values in its first half.
std::span<uint8_t> widenToFloat32(std::span<uint8_t> bytes) {
  const uint64_t count = bytes.size() / 4;
  for (uint64_t i = count; i-- > 0;) {
    uint16_t value;
    std::memcpy(&value, bytes.data() + 2 * i, 2);
    const uint32_t widened = uint32_t{value} << 16;
    std::memcpy(bytes.data() + 4 * i, &widened, 4);
  }
  return bytes;
}

// Q8_0 blocks (half d, 32 int8 q) as the F32 values d * q they equal.
constexpr uint64_t kQ80BlockBytes = 34, kQ80FloatBytes = 32 * sizeof(float);
void dequantizeQ80(std::span<const uint8_t> blocks, std::span<uint8_t> to) {
  for (uint64_t b = 0; b < blocks.size() / kQ80BlockBytes; ++b) {
    uint16_t half;
    std::memcpy(&half, blocks.data() + b * kQ80BlockBytes, 2);
    const float d = static_cast<float>(std::bit_cast<_Float16>(half));
    for (uint32_t i = 0; i < 32; ++i) {
      const float value = d * float(static_cast<int8_t>(blocks[b * kQ80BlockBytes + 2 + i]));
      std::memcpy(to.data() + b * kQ80FloatBytes + 4 * i, &value, 4);
    }
  }
}

// Q4_K blocks (half d, half dmin, 12 scale bytes, 128 bytes of 4-bit q) as their 256 F32 values.
constexpr uint64_t kQ4KBlockBytes = 144, kQ4KFloatBytes = 256 * sizeof(float);
void dequantizeQ4K(std::span<const uint8_t> blocks, std::span<uint8_t> to) {
  const auto half = [](const uint8_t *at) {
    uint16_t bits;
    std::memcpy(&bits, at, 2);
    return static_cast<float>(std::bit_cast<_Float16>(bits));
  };
  for (uint64_t b = 0; b < blocks.size() / kQ4KBlockBytes; ++b) {
    const uint8_t *block = blocks.data() + b * kQ4KBlockBytes;
    const float d = half(block), dmin = half(block + 2);
    const uint8_t *scales = block + 4, *q = block + 16;
    float *out = reinterpret_cast<float *>(to.data() + b * kQ4KFloatBytes);
    for (uint32_t j = 0; j < 8; ++j) {
      // ggml's get_scale_min_k4.
      const uint32_t scale = j < 4 ? scales[j] & 63 : (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4);
      const uint32_t min = j < 4 ? scales[j + 4] & 63 : (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
      const uint8_t *codes = q + 32 * (j / 2);
      for (uint32_t l = 0; l < 32; ++l) {
        const uint32_t code = j % 2 ? codes[l] >> 4 : codes[l] & 15;
        out[32 * j + l] = d * float(scale) * float(code) - dmin * float(min);
      }
    }
  }
}

// A Q5_0 row's blocks (half d, 4 bytes of high bits, 16 bytes of low nibbles) as the Q8_0 blocks
// (half d, 32 int8 q) they equal exactly: q = the 5-bit code - 16.
void q50ToQ80(std::span<const uint8_t> from, uint8_t *to) {
  for (uint64_t b = 0; b < from.size() / gguf::kQ50BlockBytes; ++b) {
    const uint8_t *block = from.data() + b * gguf::kQ50BlockBytes;
    uint8_t *out = to + b * kQ80BlockBytes;
    uint32_t high;
    std::memcpy(&high, block + 2, 4);
    std::memcpy(out, block, 2);
    for (uint32_t j = 0; j < 16; ++j) {
      out[2 + j] = static_cast<uint8_t>(int8_t(((block[6 + j] & 15) | (((high >> j) & 1) << 4)) - 16));
      out[2 + j + 16] = static_cast<uint8_t>(int8_t(((block[6 + j] >> 4) | (((high >> (j + 16)) & 1) << 4)) - 16));
    }
  }
}

// Destination bytes of source bytes: halved when narrowed, doubled when
// widened, a block's F32 values when dequantized.
uint64_t copiedBytes(const gguf::Copy &copy, uint64_t sourceBytes) {
  switch (copy.conversion) {
  case gguf::Conversion::NarrowToBfloat16: return sourceBytes / 2;
  case gguf::Conversion::WidenToFloat32: return sourceBytes * 2;
  case gguf::Conversion::DequantizeQ80: return sourceBytes / kQ80BlockBytes * kQ80FloatBytes;
  case gguf::Conversion::DequantizeQ4K: return sourceBytes / kQ4KBlockBytes * kQ4KFloatBytes;
  case gguf::Conversion::None: break;
  }
  return sourceBytes;
}

uint64_t copyBytes(const gguf::Copy &copy) {
  return copiedBytes(copy, copy.source.rows * copy.source.rowBytes);
}

void writeCopy(GgufSources source, int destination, const gguf::Copy &copy,
               std::span<uint8_t> staging, const PreparationCheck &admit) {
  const gguf::TensorRows &rows = copy.source;
  // Whole rows per read where they fit; a wider row in pieces of whole values.
  // A widened read takes half the staging, which holds its F32 values; a
  // dequantized one a fifth, whole rows of whole blocks.
  const bool dequantize = copy.conversion == gguf::Conversion::DequantizeQ80 ||
                          copy.conversion == gguf::Conversion::DequantizeQ4K;
  const uint64_t room = copy.conversion == gguf::Conversion::WidenToFloat32       ? staging.size() / 2
                        : copy.conversion == gguf::Conversion::DequantizeQ80      ? staging.size() / 5
                        : copy.conversion == gguf::Conversion::DequantizeQ4K      ? staging.size() / 9
                                                                                  : staging.size();
  if (dequantize && rows.rowBytes > (room & ~uint64_t{3})) throw GgufError("weight row exceeds preparation bound");
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, room & ~uint64_t{3});
  const uint64_t batch = span == rows.rowBytes ? room / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      admit();
      const uint64_t width = std::min(span, rows.rowBytes - column);
      auto bytes = staging.first(count * width);
      readRows(source, rows, first, count, column, width, bytes.data());
      switch (copy.conversion) {
      case gguf::Conversion::NarrowToBfloat16: bytes = narrowToBfloat16(bytes, rows.name); break;
      case gguf::Conversion::WidenToFloat32: bytes = widenToFloat32(staging.first(2 * count * width)); break;
      case gguf::Conversion::DequantizeQ80: {
        const auto out = staging.subspan(bytes.size(), copiedBytes(copy, bytes.size()));
        dequantizeQ80(bytes, out);
        bytes = out;
        break;
      }
      case gguf::Conversion::DequantizeQ4K: {
        const auto out = staging.subspan(bytes.size(), copiedBytes(copy, bytes.size()));
        dequantizeQ4K(bytes, out);
        bytes = out;
        break;
      }
      case gguf::Conversion::None: break;
      }
      writeWeightBytes(destination, copy.destination + copiedBytes(copy, first * rows.rowBytes + column), bytes);
    }
  }
}

// The plane0, plane1 and meta bytes of a [rows, columns] tensor in format.
std::array<uint64_t, 3> planeBytes(const QuantFormat &format, uint64_t rows, uint64_t columns) {
  const GgufPlaneBytes bytes = ggufPlaneBytes(format, rows, columns);
  return {bytes.plane0, bytes.plane1, bytes.meta};
}

uint64_t total(const std::array<uint64_t, 3> &planes) { return planes[0] + planes[1] + planes[2]; }

// Where a repack's plane0, plane1 and meta start in the image.
std::array<uint64_t, 3> planeOffsets(const gguf::Repack &repack) {
  return {repack.plane0, repack.plane1, repack.meta};
}

// The rows and columns of one repack step and its staging: complete rows
// where they fit, so one read and one submission cover many plane tiles;
// very wide rows split within the same bound.
struct RepackChunk {
  uint64_t rows = 0, columns = 0, inputBytes = 0, outputBytes = 0;
};

RepackChunk repackChunk(const gguf::Repack &repack) {
  const QuantFormat &format = kQuantFormats[repack.format];
  // Input and output staging of one block of columns of one row.
  const uint64_t blockStaging =
      ggufRowBytes(format, kGgufBlockColumns) + total(planeBytes(format, 1, kGgufBlockColumns));
  RepackChunk chunk;
  chunk.columns = std::min<uint64_t>(repack.columns,
      kWeightPreparationStagingBytes / (QUANT_TILE_ROWS * blockStaging) * kGgufBlockColumns);
  if (!chunk.columns) throw GgufError("weight row exceeds preparation bound");
  const auto input = [&](uint64_t rows) { return rows * ggufRowBytes(format, chunk.columns); };
  const auto output = [&](uint64_t rows) { return total(planeBytes(format, rows, chunk.columns)); };
  chunk.rows = chunk.columns == repack.columns
      ? std::min<uint64_t>(repack.rows,
            kWeightPreparationStagingBytes / (input(QUANT_TILE_ROWS) + output(QUANT_TILE_ROWS)) * QUANT_TILE_ROWS)
      : QUANT_TILE_ROWS;
  chunk.inputBytes = input(chunk.rows);
  chunk.outputBytes = output(chunk.rows);
  return chunk;
}

// A repack's plan within an image of imageBytes: its format, tile-aligned
// shape, sources of its row width and planes inside the image.
void requireRepack(const gguf::Repack &repack, uint64_t imageBytes) {
  if (repack.format >= GGUF_FMT_COUNT || !repack.rows || repack.rows % QUANT_TILE_ROWS || !repack.columns ||
      repack.columns % ggufColumnUnit(kQuantFormats[repack.format]))
    throw GgufError("invalid prepared weight repack");
  const QuantFormat &format = kQuantFormats[repack.format];
  const uint64_t rowBytes = ggufRowBytes(format, repack.columns);
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &rows : repack.sources) {
    // Q5_0 rows are read as the Q8_0 rows they equal.
    const bool q50 = repack.format == GGUF_FMT_Q80 && rows.type == ggml::kQ5_0;
    if (rows.rowBytes != (q50 ? repack.columns / 32 * gguf::kQ50BlockBytes : rowBytes))
      throw GgufError("invalid prepared weight source size");
    sourceRows += rows.rows;
  }
  if (sourceRows > repack.rows) throw GgufError("invalid prepared weight source size");
  // A format without plane1 has an empty plane1 at offset 0.
  const auto offsets = planeOffsets(repack);
  const auto sizes = planeBytes(format, repack.rows, repack.columns);
  for (size_t plane = 0; plane < sizes.size(); ++plane) requireRange(offsets[plane], sizes[plane], imageBytes);
}

void writeRepack(metal::MetalBackend &backend, GgufSources source, int destination,
                 const gguf::Repack &repack, const RepackChunk &chunk, metal::MetalBuffer &input,
                 metal::MetalBuffer &output, const PreparationCheck &admit) {
  const QuantFormat &format = kQuantFormats[repack.format];
  const auto offsets = planeOffsets(repack);
  auto *host = static_cast<uint8_t *>(input.contents());
  const auto *prepared = static_cast<const uint8_t *>(output.contents());
  for (uint64_t firstRow = 0; firstRow < repack.rows; firstRow += chunk.rows) {
    const uint64_t rows = std::min(chunk.rows, repack.rows - firstRow);
    for (uint64_t firstColumn = 0; firstColumn < repack.columns; firstColumn += chunk.columns) {
      admit();
      const uint64_t columns = std::min(chunk.columns, repack.columns - firstColumn);
      const uint64_t chunkRowBytes = ggufRowBytes(format, columns);
      const uint64_t column = ggufRowBytes(format, firstColumn);
      // The sources' rows in image order, then zero rows.
      uint64_t start = 0;
      for (const gguf::TensorRows &tensor : repack.sources) {
        const uint64_t begin = std::max(firstRow, start), end = std::min(firstRow + rows, start + tensor.rows);
        if (begin < end && tensor.type == ggml::kQ5_0) {
          const uint64_t count = end - begin, q50Width = columns / 32 * gguf::kQ50BlockBytes;
          std::vector<uint8_t> q50(count * q50Width);
          readRows(source, tensor, begin - start, count, firstColumn / 32 * gguf::kQ50BlockBytes, q50Width, q50.data());
          for (uint64_t r = 0; r < count; ++r)
            q50ToQ80({q50.data() + r * q50Width, q50Width}, host + (begin - firstRow + r) * chunkRowBytes);
        } else if (begin < end) {
          readRows(source, tensor, begin - start, end - begin, column, chunkRowBytes,
                   host + (begin - firstRow) * chunkRowBytes);
        }
        start += tensor.rows;
      }
      if (start < firstRow + rows) {
        const uint64_t zero = std::max(start, firstRow);
        std::memset(host + (zero - firstRow) * chunkRowBytes, 0, (firstRow + rows - zero) * chunkRowBytes);
      }
      const uint64_t chunkGroups = columns / 32;
      const auto lengths = planeBytes(format, rows, columns);
      // A plane is [rows / QUANT_TILE_ROWS][units][QUANT_TILE_ROWS] tiles and
      // a chunk starts on a tile: after the planes of the rows above it and,
      // in its tile rows, of the columns before it.
      const auto above = planeBytes(format, firstRow, repack.columns);
      const auto before = planeBytes(format, QUANT_TILE_ROWS, firstColumn);
      GgufRepackParams params{};
      params.rows = static_cast<uint32_t>(rows);
      params.input_size = static_cast<uint32_t>(columns);
      params.fmt = repack.format;
      params.src_row_bytes = static_cast<uint32_t>(chunkRowBytes);
      params.dst_plane1 = static_cast<uint32_t>(lengths[0]);
      params.dst_meta = static_cast<uint32_t>(lengths[0] + lengths[1]);
      const metal::ComputeDispatch dispatch{"gguf_repack", {{0, input}, {1, output}},
          {{2, &params, sizeof(params)}}, {rows * chunkGroups / 256, 1, 1}, {256, 1, 1}};
      static_cast<void>(backend.submit(dispatch));
      uint64_t offset = 0;
      for (size_t plane = 0; plane < lengths.size(); ++plane) {
        writeWeightBytes(destination, offsets[plane] + above[plane] + before[plane], {prepared + offset, lengths[plane]});
        offset += lengths[plane];
      }
    }
  }
}

} // namespace

PreparedWeight ggufImageWeight(GgufSources sources, const gguf::Image &image) {
  // The envelope version covers identity serialization. Conversion code and
  // its storage ABI are fingerprinted at build time, independently of tuning.
  WeightIdentity identity("splash-gguf-preparation-v3 " SPLASH_GGUF_PREPARATION_ID);
  identity.record("image", image.layer, image.type, image.bytes);
  const auto input = [&](const gguf::TensorRows &rows) {
    const uint64_t shape[] = {rows.rows, rows.rowBytes};
    if (rows.file >= sources.size()) throw GgufError("prepared weight rows name a missing GGUF file");
    identity.input(*sources[rows.file], rows.offset, rows.rows * rows.rowBytes, ggmlTypeName(rows.type), shape);
    identity.record("order", rows.order.from, rows.order.headRows, rows.order.keyHeads,
                    rows.order.valueHeadsPerKey);
  };
  for (const gguf::Fill &fill : image.fills) identity.record("fill", fill.offset, weightDigest(fill.bytes));
  for (const gguf::Copy &copy : image.copies) {
    identity.record("copy", copy.destination, static_cast<uint32_t>(copy.conversion));
    input(copy.source);
  }
  for (const gguf::Repack &repack : image.repacks) {
    identity.record("repack", repack.format, repack.rows, repack.columns, repack.plane0, repack.plane1, repack.meta);
    for (const gguf::TensorRows &rows : repack.sources) input(rows);
  }
  return identity.weight(image.bytes, "target/" + image.name, sources.front()->path().string());
}

void writeGgufImage(metal::MetalBackend &backend, GgufSources source, int destination,
                    const gguf::Image &image, const PreparationCheck &admit) {
  for (const gguf::Fill &fill : image.fills) {
    requireRange(fill.offset, fill.bytes.size(), image.bytes);
    writeWeightBytes(destination, fill.offset, fill.bytes);
  }
  // Copies keep their source precision and never pass through a
  // quantization operation. One staging buffer serves every copy.
  uint64_t copyStaging = 0;
  for (const gguf::Copy &copy : image.copies) {
    if (!copy.source.rows || !copy.source.rowBytes ||
        (copy.conversion == gguf::Conversion::DequantizeQ80   ? copy.source.rowBytes % kQ80BlockBytes
         : copy.conversion == gguf::Conversion::DequantizeQ4K ? copy.source.rowBytes % kQ4KBlockBytes
                                                              : copy.conversion != gguf::Conversion::None &&
                                                                    copy.source.rowBytes % 4))
      throw GgufError("invalid prepared weight copy");
    requireRange(copy.destination, copyBytes(copy), image.bytes);
    // A widened copy stages its F32 values, a dequantized one its blocks and them.
    const uint64_t sourceBytes = copy.source.rows * copy.source.rowBytes;
    const bool dequantized = copy.conversion == gguf::Conversion::DequantizeQ80 ||
                             copy.conversion == gguf::Conversion::DequantizeQ4K;
    const uint64_t staged = copy.conversion == gguf::Conversion::WidenToFloat32 ? copyBytes(copy)
                            : dequantized                                       ? sourceBytes + copyBytes(copy)
                                                                                : sourceBytes;
    copyStaging = std::max(copyStaging, std::min(staged, kWeightPreparationStagingBytes));
  }
  {
    std::vector<uint8_t> staging(copyStaging);
    for (const gguf::Copy &copy : image.copies) writeCopy(source, destination, copy, staging, admit);
  }
  // One pair of staging buffers serves every repack of the image.
  std::vector<RepackChunk> chunks;
  RepackChunk largest;
  for (const gguf::Repack &repack : image.repacks) {
    requireRepack(repack, image.bytes);
    chunks.push_back(repackChunk(repack));
    largest.inputBytes = std::max(largest.inputBytes, chunks.back().inputBytes);
    largest.outputBytes = std::max(largest.outputBytes, chunks.back().outputBytes);
  }
  if (!image.repacks.empty()) {
    admit();
    auto input = backend.allocateBuffer(largest.inputBytes, metal::BufferStorage::Shared, "prepare/source");
    auto output = backend.allocateBuffer(largest.outputBytes, metal::BufferStorage::Shared, "prepare/planes");
    for (size_t i = 0; i < image.repacks.size(); ++i)
      writeRepack(backend, source, destination, image.repacks[i], chunks[i], input, output, admit);
  }
}

} // namespace splash::model
