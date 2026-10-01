// Editing this file re-prepares every affine model. DFlash2 drafts are affine models too.
#include "WeightPreparationIdentity.hpp"
#include "model/AffinePreparation.hpp"
#include "model/Q5Pack.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::model::affine {
namespace {

// Input and output staging: each half of the preparation bound.
constexpr uint64_t kChunkBytes = kWeightPreparationStagingBytes / 2;

// Bytes per row of a 64-column group: codes, scales, biases.
std::array<uint32_t, 3> groupBytes(const Section &section) { return {8 * section.bits, 2, 2}; }

// Bytes per row of a 64-column group of a BF16 weight.
constexpr uint32_t kBfloat16GroupBytes = kQ4GroupElements * kBFloat16Bytes;

// Groups of one field a 256-row tile step converts.
uint32_t chunkGroups(const Section &section, uint32_t unit) {
  return static_cast<uint32_t>(std::min<uint64_t>(section.columns / kQ4GroupElements, kChunkBytes / (kQ4StorageN * unit)));
}

float bfloat16Value(const uint8_t *bytes) {
  uint16_t bits;
  std::memcpy(&bits, bytes, sizeof bits);
  return std::bit_cast<float>(uint32_t(bits) << 16);
}

// The float nearest a finite value, ties to even (Apple clang's _Float16
// conversion, the platform's native half). Throws on overflow (the source
// magnitude does not fit half's finite range); half's normal range is wider
// than BF16's, so underflow to a half subnormal or zero is the only new
// rounding risk this conversion adds, and an all-zero group already
// dequantizes to zero.
uint16_t nearestFloat16(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  if (!std::isfinite(static_cast<float>(half)))
    throw std::runtime_error("BF16 scale or bias overflows half range");
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof bits);
  return bits;
}

// Canonical prepared scale/bias dtype is half: converts `count` consecutive
// little-endian BF16 values in place to half bits.
void convertBf16ToHalf(std::span<uint8_t> bytes, uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    const uint16_t half = nearestFloat16(bfloat16Value(bytes.data() + i * 2));
    std::memcpy(bytes.data() + i * 2, &half, sizeof half);
  }
}

float float16Value(const uint8_t *bytes) {
  _Float16 half;
  std::memcpy(&half, bytes, sizeof half);
  return static_cast<float>(half);
}

// The BF16 nearest a finite value, ties to even.
uint16_t nearestBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// Every Copy-kind tensor other than U32 codes is canonicalized to BF16: every
// kernel that reads one (norm_rms's weight, a GDN's conv1d/dt_bias) reads it
// in that type, never half or F32. Converts `count` consecutive little-endian
// F16 values in place to BF16 bits; a BF16 source is untouched.
void convertF16ToBf16(std::span<uint8_t> bytes, uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    const uint16_t bf16 = nearestBfloat16(float16Value(bytes.data() + i * 2));
    std::memcpy(bytes.data() + i * 2, &bf16, sizeof bf16);
  }
}

// Gathers into input rows [firstRow, firstRow + 256) of groups [firstGroup,
// firstGroup + count) of one field of one expert, unit bytes per row of a
// group: each part's rows as stored, rows past the parts zero. A step of
// whole rows reads a part's rows at once, as they are contiguous in the
// source. field 1/2 (scales/biases) are converted to half in place when the
// source part stores them as BF16; field 0's own dtype (U32 codes, or a
// Quantize section's raw BF16 weight input) is never scale/bias and is left
// as read.
void gatherRows(const Section &section, uint32_t expert, size_t field, uint32_t unit, uint32_t firstRow,
                uint32_t firstGroup, uint32_t count, std::span<uint8_t> input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t rowBytes = uint64_t(count) * unit;
  const uint64_t sourceRowBytes = uint64_t(groups) * unit;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + kQ4StorageN, partStart + part.rows);
    if (begin < end) {
      const uint64_t offset = (uint64_t(expert) * part.rows + begin - partStart) * sourceRowBytes +
                              uint64_t(firstGroup) * unit;
      uint8_t *to = input.data() + (begin - firstRow) * rowBytes;
      if (count == groups) {
        part.fields[field].tensor->read(offset, {to, (end - begin) * rowBytes});
      } else {
        for (uint32_t row = 0; row < end - begin; ++row)
          part.fields[field].tensor->read(offset + row * sourceRowBytes, {to + row * rowBytes, rowBytes});
      }
      if (field != 0 && part.fields[field].tensor->dtype == "BF16")
        convertBf16ToHalf({to, (end - begin) * rowBytes}, (end - begin) * rowBytes / 2);
    }
    partStart += part.rows;
  }
  if (firstRow + kQ4StorageN > partStart) {
    const uint32_t gathered = std::max(firstRow, partStart) - firstRow;
    std::fill(input.begin() + gathered * rowBytes, input.begin() + kQ4StorageN * rowBytes, 0);
  }
}

// Transposes a gathered tile of 256 rows of count groups into count groups of
// 256 rows, unit bytes each.
void tile(std::span<const uint8_t> input, std::span<uint8_t> output, uint32_t count, uint32_t unit) {
  const uint64_t rowBytes = uint64_t(count) * unit;
  for (uint32_t group = 0; group < count; ++group)
    for (uint32_t row = 0; row < kQ4StorageN; ++row)
      std::memcpy(output.data() + (uint64_t(group) * kQ4StorageN + row) * unit,
                  input.data() + row * rowBytes + uint64_t(group) * unit, unit);
}

// Reorders each field of each expert into [rows / 256][groups][256] tiles of
// its group bytes; rows past the parts are zero.
void writeProjection(int destination, const Section &section, std::vector<uint8_t> &input,
                     std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint64_t expertBytes = section.bytes / section.experts;
  for (uint32_t expert = 0; expert < section.experts; ++expert) {
    uint64_t fieldBase = section.offset + expert * expertBytes;
    for (size_t field = 0; field < unit.size(); ++field) {
      const uint32_t stepGroups = chunkGroups(section, unit[field]);
      for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
        for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
          admit();
          const uint32_t count = std::min(stepGroups, groups - firstGroup);
          gatherRows(section, expert, field, unit[field], firstRow, firstGroup, count, input);
          tile(input, output, count, unit[field]);
          const uint64_t offset = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN * unit[field];
          writeWeightBytes(destination, fieldBase + offset,
                           std::span(output).first(uint64_t(kQ4StorageN) * count * unit[field]));
        }
      }
      fieldBase += uint64_t(section.rows) * groups * unit[field];
    }
  }
}

// A group of 64 BF16 weights as MLX's affine quantization rounds it to 4 bits
// (mlx.core.quantize, its Metal kernel), in float: the range runs from the
// minimum to the maximum or 0, whichever is greater, the end of the range
// farther from zero is the bias, the scale is adjusted so that 0 falls on a
// code unless that code is 0, each code is round((w - bias) / scale), halves
// away from zero, clamped to [0, 15], and scale and bias are stored as half
// (the canonical prepared scale/bias dtype).
void quantizeGroup(const uint8_t *weights, uint8_t *codes, uint16_t &scale, uint16_t &bias) {
  std::array<float, kQ4GroupElements> w;
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = bfloat16Value(weights + i * kBFloat16Bytes);
    if (!std::isfinite(w[i])) throw std::runtime_error("non-finite weight in a BF16 projection");
  }
  const float minimum = *std::min_element(w.begin(), w.end());
  const float maximum = std::max(0.0F, *std::max_element(w.begin(), w.end()));
  const bool minimumEdge = std::fabs(minimum) > std::fabs(maximum);
  float step = std::max((maximum - minimum) / 15.0F, 1e-7F);
  if (!minimumEdge) step = -step;
  const float edge = minimumEdge ? minimum : maximum;
  const float q0 = std::round(edge / step);
  float offset = 0.0F;
  if (q0 != 0.0F) {
    step = edge / q0;
    offset = edge;
  }
  const auto code = [&](float value) {
    return static_cast<uint8_t>(std::clamp(std::round((value - offset) / step), 0.0F, 15.0F));
  };
  for (size_t i = 0; i < w.size(); i += 2) codes[i / 2] = static_cast<uint8_t>(code(w[i]) | code(w[i + 1]) << 4);
  scale = nearestFloat16(step);
  bias = nearestFloat16(offset);
}

// Quantizes each step of 256 rows and count groups of a BF16 projection and
// writes its codes, scales and biases as writeProjection writes them: each
// field in [rows / 256][groups][256] tiles of its group bytes.
void writeQuantized(int destination, const Section &section, std::vector<uint8_t> &input,
                    std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint32_t stepGroups = chunkGroups(section, kBfloat16GroupBytes);
  std::array<uint64_t, 3> fieldBase{section.offset};
  for (size_t field = 1; field < unit.size(); ++field)
    fieldBase[field] = fieldBase[field - 1] + uint64_t(section.rows) * groups * unit[field - 1];
  for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
    for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
      admit();
      const uint32_t count = std::min(stepGroups, groups - firstGroup);
      gatherRows(section, 0, 0, kBfloat16GroupBytes, firstRow, firstGroup, count, input);
      // The step's fields, one after another, each in [count][256] tile order.
      std::array<uint8_t *, 3> fields{output.data()};
      for (size_t field = 1; field < unit.size(); ++field)
        fields[field] = fields[field - 1] + uint64_t(count) * kQ4StorageN * unit[field - 1];
      for (uint32_t row = 0; row < kQ4StorageN; ++row) {
        for (uint32_t group = 0; group < count; ++group) {
          const uint64_t at = uint64_t(group) * kQ4StorageN + row;
          uint16_t scale, bias;
          quantizeGroup(input.data() + (uint64_t(row) * count + group) * kBfloat16GroupBytes,
                        fields[0] + at * unit[0], scale, bias);
          std::memcpy(fields[1] + at * unit[1], &scale, sizeof scale);
          std::memcpy(fields[2] + at * unit[2], &bias, sizeof bias);
        }
      }
      const uint64_t tile = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN;
      for (size_t field = 0; field < unit.size(); ++field)
        writeWeightBytes(destination, fieldBase[field] + tile * unit[field],
                         {fields[field], uint64_t(count) * kQ4StorageN * unit[field]});
    }
  }
}

// Gathers one 256-row tile's codes of a Q5 section into [group][256 rows]
// contiguous lo4/hi planes (Q5Pack.hpp), reading whichever part covers each
// row at that part's own bits: a Q4 part (nibble codes, 32 B/group) is
// promoted losslessly (hi = 0), a Q5 part (MLX's native 40 B/group bitstream)
// is split. Rows past the parts are zero (both buffers are reassigned).
void gatherQ5Codes(const Section &section, uint32_t firstRow, uint32_t groups, std::vector<uint8_t> &lo4Tile,
                   std::vector<uint8_t> &hiTile) {
  lo4Tile.assign(uint64_t(groups) * kQ4StorageN * q5::kLo4Bytes, 0);
  hiTile.assign(uint64_t(groups) * kQ4StorageN * q5::kHiBytes, 0);
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + kQ4StorageN, partStart + part.rows);
    if (begin < end) {
      const uint32_t sourceUnit = part.bits == 4 ? q5::kLo4Bytes : q5::kMlxGroupBytes;
      const uint64_t rowBytes = uint64_t(groups) * sourceUnit;
      std::vector<uint8_t> row(rowBytes);
      for (uint32_t r = begin; r < end; ++r) {
        part.fields[0].tensor->read(uint64_t(r - partStart) * rowBytes, row);
        const uint32_t rowInTile = r - firstRow;
        for (uint32_t g = 0; g < groups; ++g) {
          uint8_t *lo4Dest = lo4Tile.data() + (uint64_t(g) * kQ4StorageN + rowInTile) * q5::kLo4Bytes;
          uint8_t *hiDest = hiTile.data() + (uint64_t(g) * kQ4StorageN + rowInTile) * q5::kHiBytes;
          if (part.bits == 4) q5::promoteQ4Group(row.data() + uint64_t(g) * sourceUnit, lo4Dest, hiDest);
          else q5::splitMlxQ5Group(row.data() + uint64_t(g) * sourceUnit, lo4Dest, hiDest);
        }
      }
    }
    partStart += part.rows;
  }
}

// Writes a Q5 (5-bit affine) section: field order {lo4, scales, biases, hi},
// so the lo4 plane alone is byte-identical to a same-shape all-Q4 section's
// weight plane. Codes go through gatherQ5Codes; scales/biases are 2 bytes
// either way, so they reuse the ordinary per-field gatherRows/tile unchanged.
// ponytail: one gatherQ5Codes source read per (row, part) rather than a bulk
// contiguous read per part run (writeProjection's fast path) -- mixed-bits
// part boundaries rarely land on a 256-row tile boundary, and this is
// preparation-time-only cost; a fast path for the common single-bits-section
// case is the upgrade path if prep time becomes a problem.
void writeQ5Projection(int destination, const Section &section, std::vector<uint8_t> &input,
                      std::vector<uint8_t> &output, const PreparationCheck &admit) {
  if (section.experts != 1) throw std::runtime_error("Q5 expert projections are not supported");
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t lo4Bytes = uint64_t(section.rows) * groups * q5::kLo4Bytes;
  const uint64_t scaleBytes = uint64_t(section.rows) * groups * 2;
  const uint64_t lo4Base = section.offset;
  const uint64_t scaleBase = lo4Base + lo4Bytes;
  const uint64_t biasBase = scaleBase + scaleBytes;
  const uint64_t hiBase = biasBase + scaleBytes;
  std::vector<uint8_t> lo4Tile, hiTile;
  for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
    admit();
    gatherQ5Codes(section, firstRow, groups, lo4Tile, hiTile);
    const uint32_t tile = firstRow / kQ4StorageN;
    for (uint32_t g = 0; g < groups; ++g) {
      const uint64_t tileGroupBase = (uint64_t(tile) * groups + g) * kQ4StorageN;
      writeWeightBytes(destination, lo4Base + tileGroupBase * q5::kLo4Bytes,
                       std::span(lo4Tile).subspan(uint64_t(g) * kQ4StorageN * q5::kLo4Bytes,
                                                  uint64_t(kQ4StorageN) * q5::kLo4Bytes));
      writeWeightBytes(destination, hiBase + tileGroupBase * q5::kHiBytes,
                       std::span(hiTile).subspan(uint64_t(g) * kQ4StorageN * q5::kHiBytes,
                                                 uint64_t(kQ4StorageN) * q5::kHiBytes));
    }
  }
  const uint32_t stepGroups = chunkGroups(section, 2);
  for (uint32_t field = 1; field <= 2; ++field) {
    const uint64_t fieldBase = field == 1 ? scaleBase : biasBase;
    for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
      for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
        admit();
        const uint32_t count = std::min(stepGroups, groups - firstGroup);
        gatherRows(section, 0, field, 2, firstRow, firstGroup, count, input);
        tile(input, output, count, 2);
        const uint64_t offset = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN * 2;
        writeWeightBytes(destination, fieldBase + offset, std::span(output).first(uint64_t(kQ4StorageN) * count * 2));
      }
    }
  }
}

// float(-exp(double(A_log))) of a BF16 or F32 vector. The A_log it reads and
// the decay it writes are staged together, within the staging bound of every
// conversion step.
void writeDecay(int destination, const Section &section) {
  const SourceTensor &tensor = *section.input.tensor;
  if (tensor.bytes + section.bytes > kWeightPreparationStagingBytes)
    throw std::runtime_error("decay tensor exceeds the preparation staging bound");
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  std::vector<float> values(section.bytes / sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    float logarithm;
    if (tensor.dtype == "BF16") {
      logarithm = bfloat16Value(bytes.data() + i * kBFloat16Bytes);
    } else if (tensor.dtype == "F16") {
      logarithm = float16Value(bytes.data() + i * kBFloat16Bytes);
    } else {
      std::memcpy(&logarithm, bytes.data() + i * 4, 4);
    }
    values[i] = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
    if (!std::isfinite(values[i])) throw std::runtime_error("non-finite GDN decay");
  }
  writeWeightBytes(destination, section.offset, {reinterpret_cast<const uint8_t *>(values.data()), section.bytes});
}

// A scale or bias tensor (AffinePlan.hpp's halfCopy): a raw copy of an F16
// source, converted in place of a BF16 source, in chunks of `input`.
void writeHalfCopy(int destination, const Section &section, std::vector<uint8_t> &input,
                   const PreparationCheck &admit) {
  const SourceTensor &tensor = *section.input.tensor;
  const uint64_t chunk = std::min(section.bytes, kChunkBytes);
  for (uint64_t at = 0; at < section.bytes; at += chunk) {
    admit();
    const auto piece = std::span(input).first(std::min(chunk, section.bytes - at));
    tensor.read(at, piece);
    if (tensor.dtype == "BF16") convertBf16ToHalf(piece, piece.size() / 2);
    writeWeightBytes(destination, section.offset + at, piece);
  }
}

// AffinePlan.hpp's copy(): U32 packed codes (the token table's weight plane)
// are a raw copy, every other tensor -- a norm, a GDN's conv1d.weight or
// dt_bias -- is canonicalized to BF16 (every kernel that reads one reads it
// in that type). A BF16 source is a raw copy; an F16 source is converted.
void writeCopy(int destination, const Section &section, std::vector<uint8_t> &input,
              const PreparationCheck &admit) {
  const SourceTensor &tensor = *section.input.tensor;
  if (tensor.dtype == "U32") {
    tensor.copy(destination, section.offset, input, admit);
    return;
  }
  const uint64_t chunk = std::min(section.bytes, kChunkBytes);
  for (uint64_t at = 0; at < section.bytes; at += chunk) {
    admit();
    const auto piece = std::span(input).first(std::min(chunk, section.bytes - at));
    tensor.read(at, piece);
    if (tensor.dtype == "F16") convertF16ToBf16(piece, piece.size() / 2);
    writeWeightBytes(destination, section.offset + at, piece);
  }
}

} // namespace

// The key: this code's identity, the plan and the bytes, dtype and shape of
// every tensor read. config.json is only validated against the layout, whose
// dimensions and quantization the plan records; nothing else in it changes
// these bytes.
PreparedWeight affineImageWeight(const Image &image, std::string_view directory, const std::string &source) {
  WeightIdentity identity("splash-affine-preparation-v4 " SPLASH_AFFINE_PREPARATION_ID);
  identity.record("image", image.magic, image.layer, image.type, image.bytes);
  for (const Section &section : image.sections) {
    identity.record("section", int(section.kind), section.offset, section.bytes, section.rows, section.columns,
                    section.experts, section.bits);
    if (section.parts.empty()) section.input.tensor->identify(identity);
    for (const ProjectionPart &part : section.parts) {
      identity.record("part", part.rows);
      for (const Input &field : part.fields) field.tensor->identify(identity);
    }
  }
  return identity.weight(image.bytes, std::string(directory) + "/" + image.name, source);
}

void writeAffineImage(int destination, const Image &image, const PreparationCheck &admit) {
  // One input and one output buffer serve every section.
  uint64_t inputBytes = 0, outputBytes = 0;
  for (const Section &section : image.sections) {
    switch (section.kind) {
    case SectionKind::Projection:
      for (uint32_t unit : groupBytes(section)) {
        const uint64_t bytes = uint64_t(kQ4StorageN) * chunkGroups(section, unit) * unit;
        inputBytes = std::max(inputBytes, bytes);
        outputBytes = std::max(outputBytes, bytes);
      }
      break;
    case SectionKind::Quantize: {
      const uint64_t rows = uint64_t(kQ4StorageN) * chunkGroups(section, kBfloat16GroupBytes);
      const auto unit = groupBytes(section);
      inputBytes = std::max(inputBytes, rows * kBfloat16GroupBytes);
      outputBytes = std::max(outputBytes, rows * (unit[0] + unit[1] + unit[2]));
      break;
    }
    case SectionKind::Copy:
    case SectionKind::HalfCopy:
      inputBytes = std::max(inputBytes, std::min(section.bytes, kChunkBytes));
      break;
    case SectionKind::Decay:
      break;
    }
  }
  std::vector<uint8_t> input(inputBytes), output(outputBytes);
  writeWeightBytes(destination, 0, weightFileHeader(image.magic, image.layer, image.type));
  // readAffineProjection's bits, one entry per single-expert Projection or
  // Quantize section (both are read by it; a Quantize section is always
  // 4-bit), in file order: only the sections it actually reads (the mixer
  // and, for a dense FFN, gate/up/down/lm_head) advance the reader's cursor,
  // so a section only ever read by the Q8/expert readers (a MoE router, gate
  // or shared expert) leaves an unconsumed entry rather than desyncing it.
  std::vector<uint32_t> bitsTable;
  for (const Section &section : image.sections)
    if ((section.kind == SectionKind::Projection || section.kind == SectionKind::Quantize) &&
        section.experts == 1)
      bitsTable.push_back(section.bits);
  writeWeightBytes(destination, kAffineBitsTableOffset, affineBitsTableBytes(bitsTable));
  for (const Section &section : image.sections) {
    admit();
    switch (section.kind) {
    case SectionKind::Projection:
      if (section.bits == 5) writeQ5Projection(destination, section, input, output, admit);
      else writeProjection(destination, section, input, output, admit);
      break;
    case SectionKind::Quantize:
      writeQuantized(destination, section, input, output, admit);
      break;
    case SectionKind::Decay:
      writeDecay(destination, section);
      break;
    case SectionKind::Copy:
      writeCopy(destination, section, input, admit);
      break;
    case SectionKind::HalfCopy:
      writeHalfCopy(destination, section, input, admit);
      break;
    }
  }
}

} // namespace splash::model::affine
