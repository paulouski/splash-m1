#pragma once

// Editing this file re-prepares every affine model.

// Q5/group64 affine codes: MLX's native format is a sequential 5-bit
// little-endian bitstream (40 bytes/group, no byte alignment between
// values). Splash stores Q5 as two planes instead, so the Apple7/8 register
// kernels can extend their existing Q4 nibble read with one cheap extra bit
// rather than a byte-crossing 5-bit extraction:
//   lo4[32]: the low 4 bits of each of the 64 values, packed exactly as a Q4
//            tile already is (codes[i/2] = lo(i) | lo(i+1) << 4).
//   hi[8]:   the 5th bit of each value, 8 values/byte, same per-value order
//            as lo4's nibbles (hi[i/8] bit (i%8) = bit 4 of value i).
// A Q4 source promoted into a Q5 (mixed-bits) section is lossless: its lo4
// plane is its existing nibble codes unchanged, its hi plane is all zero.
// Design ported from qwen38-reforge/runs/swift-tiled-q5-20260928 (verified
// there against mx.quantized_matmul(bits=5) on real tensors).

#include <array>
#include <cstdint>
#include <cstring>

namespace splash::model::q5 {

inline constexpr uint32_t kGroupElements = 64;
inline constexpr uint32_t kMlxGroupBytes = 40;  // 64 * 5 bits
inline constexpr uint32_t kLo4Bytes = 32;       // 64 * 4 bits, Q4-tile format
inline constexpr uint32_t kHiBytes = 8;         // 64 * 1 bit

// Value `index` (0..63) of a group packed as MLX packs affine bits=5: a
// sequential bitstream, value `j` at bits [5j, 5j+4], byte array little-endian.
[[nodiscard]] inline uint32_t extractMlxQ5(const uint8_t *group40, uint32_t index) noexcept {
  const uint32_t bitOffset = index * 5;
  const uint32_t byteIndex = bitOffset / 8;
  const uint32_t shift = bitOffset % 8;
  uint32_t word = group40[byteIndex];
  if (byteIndex + 1 < kMlxGroupBytes) word |= uint32_t(group40[byteIndex + 1]) << 8;
  return (word >> shift) & 0x1Fu;
}

// The inverse of extractMlxQ5: ORs value (0..31) into its 5-bit slot of a
// zero-initialized 40-byte group. Independent of extractMlxQ5 (a from-scratch
// packer), so a pack/unpack round trip is not circular.
inline void packMlxQ5(uint8_t *group40, uint32_t index, uint32_t value) noexcept {
  const uint32_t bitOffset = index * 5;
  const uint32_t byteIndex = bitOffset / 8;
  const uint32_t shift = bitOffset % 8;
  const uint32_t bits = (value & 0x1Fu) << shift;
  group40[byteIndex] |= uint8_t(bits & 0xFFu);
  if (byteIndex + 1 < kMlxGroupBytes) group40[byteIndex + 1] |= uint8_t((bits >> 8) & 0xFFu);
}

// Splits a group of 64 5-bit MLX-native codes into Splash's lo4/hi planes.
inline void splitMlxQ5Group(const uint8_t *group40, uint8_t *lo4, uint8_t *hi) noexcept {
  std::memset(hi, 0, kHiBytes);
  for (uint32_t i = 0; i < kGroupElements; ++i) {
    const uint32_t v = extractMlxQ5(group40, i);
    const uint32_t lo = v & 0xF, hib = (v >> 4) & 1;
    if (i % 2 == 0) lo4[i / 2] = uint8_t(lo);
    else lo4[i / 2] = uint8_t(lo4[i / 2] | (lo << 4));
    hi[i / 8] = uint8_t(hi[i / 8] | (hib << (i % 8)));
  }
}

// A Q4 group's nibble codes, promoted losslessly into Splash's Q5 planes: the
// lo4 plane is copied unchanged, the hi plane is zero (an exact Q5 code with
// hi=0 equals the original Q4 code at the same scale and bias).
inline void promoteQ4Group(const uint8_t *q4Lo4, uint8_t *lo4, uint8_t *hi) noexcept {
  std::memcpy(lo4, q4Lo4, kLo4Bytes);
  std::memset(hi, 0, kHiBytes);
}

// The 5-bit code (0..31) value `index` decodes to from Splash's split planes.
[[nodiscard]] inline uint32_t valueFromSplit(const uint8_t *lo4, const uint8_t *hi, uint32_t index) noexcept {
  const uint32_t nibble = (index % 2 == 0) ? (lo4[index / 2] & 0xF) : (lo4[index / 2] >> 4);
  const uint32_t hiBit = (hi[index / 8] >> (index % 8)) & 1;
  return nibble | (hiBit << 4);
}

} // namespace splash::model::q5
