#pragma once

// Layout constants of the weight files: preparation writes them, the weight
// store reads them. Preparation code takes them from this header, so the
// preparation identity does not follow the reader's API.
// Editing this file re-prepares every affine and vision model.

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

// An affine image's per-projection-section bits (4 or 5), one entry per
// Projection-kind section in image.sections order: a little-endian u32 count
// then one byte per entry, right after the 16-byte magic header, within the
// same 16 KiB header block (the first section always starts at
// kWeightFileAlignment, so this space was zero padding before). Preparation
// writes it (AffinePreparation.cpp); the reader parses it before it opens
// any projection, so a section's stored width is never assumed.
inline constexpr uint64_t kAffineBitsTableOffset = 16;

// The bytes of an affine bits table (offset kAffineBitsTableOffset), for
// writeWeightBytes.
[[nodiscard]] inline std::vector<uint8_t> affineBitsTableBytes(std::span<const uint32_t> bits) {
  const uint64_t tableBytes = 4 + bits.size();
  if (kAffineBitsTableOffset + tableBytes > kWeightFileAlignment)
    throw std::invalid_argument("affine bits table does not fit the header block");
  std::vector<uint8_t> bytes(tableBytes);
  const uint32_t count = static_cast<uint32_t>(bits.size());
  std::memcpy(bytes.data(), &count, 4);
  for (size_t i = 0; i < bits.size(); ++i) {
    if (bits[i] > 0xFF) throw std::invalid_argument("affine section bits does not fit a byte");
    bytes[4 + i] = static_cast<uint8_t>(bits[i]);
  }
  return bytes;
}

// The bits table of header, the first kWeightFileAlignment bytes of a mapped
// affine image.
[[nodiscard]] inline std::vector<uint32_t> parseAffineBitsTable(std::span<const uint8_t> header) {
  if (header.size() < kAffineBitsTableOffset + 4) throw std::runtime_error("affine bits table is truncated");
  uint32_t count;
  std::memcpy(&count, header.data() + kAffineBitsTableOffset, 4);
  if (kAffineBitsTableOffset + 4 + uint64_t(count) > header.size())
    throw std::runtime_error("affine bits table is truncated");
  std::vector<uint32_t> bits(count);
  for (uint32_t i = 0; i < count; ++i) bits[i] = header[kAffineBitsTableOffset + 4 + i];
  return bits;
}

// The packed vision tower.
inline constexpr std::string_view kVisionMagic = "MDFV0001";

// offset rounded up to the next section boundary.
[[nodiscard]] inline constexpr uint64_t alignWeightOffset(uint64_t offset) {
  return (offset + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1);
}

// The 16-byte header a weight file starts with: its eight-byte magic, then
// its layer and type, little-endian.
[[nodiscard]] inline std::array<uint8_t, 16> weightFileHeader(std::string_view magic, uint32_t layer,
                                                              uint32_t type) {
  if (magic.size() != 8) throw std::invalid_argument("a weight file magic is eight bytes");
  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), magic.data(), 8);
  std::memcpy(header.data() + 8, &layer, 4);
  std::memcpy(header.data() + 12, &type, 4);
  return header;
}

} // namespace splash::model
