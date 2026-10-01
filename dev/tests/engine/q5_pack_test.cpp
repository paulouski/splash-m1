#include "TestChecks.hpp"
#include "model/Q5Pack.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace splash::model::q5;
using splash::test::require;

namespace {

// scale * code + bias, MLX's own affine dequant.
float dequant(uint16_t scaleBits, uint16_t biasBits, uint32_t code) {
  // Half -> float via the exact-half trick used by the production kernels:
  // 0x6400 | v is the exact half value 1024 + v for v in [0, 31].
  const uint32_t exactHalfBits = 0x6400u | code;
  const auto halfToFloat = [](uint16_t bits) {
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1F;
    const uint32_t mantissa = bits & 0x3FF;
    if (exponent == 0) return std::bit_cast<float>(sign);  // codes are never subnormal here
    const uint32_t bits32 = sign | ((exponent + 112) << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits32);
  };
  const float value1024 = halfToFloat(static_cast<uint16_t>(exactHalfBits));
  const float scale = halfToFloat(scaleBits);
  const float bias = halfToFloat(biasBits);
  return (value1024 - 1024.0f) * scale + bias;
}

} // namespace

int main() {
  try {
    // 1. Independent pack -> production unpack round trip, many seeds.
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> codeDist(0, 31);
    for (int seed = 0; seed < 32; ++seed) {
      std::array<uint32_t, kGroupElements> values{};
      for (auto &v : values) v = codeDist(rng);
      std::array<uint8_t, kMlxGroupBytes> group40{};
      for (uint32_t i = 0; i < kGroupElements; ++i) packMlxQ5(group40.data(), i, values[i]);
      for (uint32_t i = 0; i < kGroupElements; ++i)
        require(extractMlxQ5(group40.data(), i) == values[i], "extractMlxQ5 round trip mismatch");

      // 2. Split into lo4/hi and check every value decodes exactly.
      std::array<uint8_t, kLo4Bytes> lo4{};
      std::array<uint8_t, kHiBytes> hi{};
      splitMlxQ5Group(group40.data(), lo4.data(), hi.data());
      for (uint32_t i = 0; i < kGroupElements; ++i)
        require(valueFromSplit(lo4.data(), hi.data(), i) == values[i], "split-plane value mismatch");

      // 3. Dequant via the split planes equals dequant via the MLX code directly.
      const uint16_t scale = 0x3C00 + static_cast<uint16_t>(seed);  // arbitrary finite half
      const uint16_t bias = 0xB800 + static_cast<uint16_t>(seed * 3);
      for (uint32_t i = 0; i < kGroupElements; ++i) {
        const uint32_t splitCode = valueFromSplit(lo4.data(), hi.data(), i);
        const uint32_t mlxCode = extractMlxQ5(group40.data(), i);
        require(dequant(scale, bias, splitCode) == dequant(scale, bias, mlxCode),
                "split-plane dequant does not exactly equal MLX dequant");
      }
    }

    // 4. Q4 promotion: lo4 unchanged, hi all zero, decodes to the original nibble.
    for (int seed = 0; seed < 8; ++seed) {
      std::array<uint8_t, kLo4Bytes> q4Lo4{};
      for (auto &b : q4Lo4) b = static_cast<uint8_t>(rng());
      std::array<uint8_t, kLo4Bytes> lo4{};
      std::array<uint8_t, kHiBytes> hi{};
      hi.fill(0xFF);  // promoteQ4Group must zero this itself, not rely on a caller's memset.
      promoteQ4Group(q4Lo4.data(), lo4.data(), hi.data());
      require(lo4 == q4Lo4, "promoted Q4 lo4 plane changed");
      for (uint8_t b : hi) require(b == 0, "promoted Q4 hi plane is not zero");
      for (uint32_t i = 0; i < kGroupElements; ++i) {
        const uint32_t nibble = (i % 2 == 0) ? (q4Lo4[i / 2] & 0xF) : (q4Lo4[i / 2] >> 4);
        require(valueFromSplit(lo4.data(), hi.data(), i) == nibble, "promoted value does not match source nibble");
      }
    }

    // 5. A mixed Q4+Q5 section, reproducing gatherQ5Codes' [group][256 rows]
    // tile addressing (AffinePreparation.cpp): part A (rows 0..36, bits=4)
    // and part B (rows 37..99, bits=5) share a 256-row tile whose boundary
    // (row 37) is not 256-aligned, with padding rows 100..255 past both.
    {
      constexpr uint32_t kTileRows = 256, kGroups = 2, kPartABoundary = 37, kSectionRows = 100;
      std::array<uint32_t, kGroups * kTileRows * kGroupElements> expected{};  // 0 past the parts
      std::vector<uint8_t> lo4Tile(uint64_t(kGroups) * kTileRows * kLo4Bytes, 0xAA);  // poisoned, must be overwritten
      std::vector<uint8_t> hiTile(uint64_t(kGroups) * kTileRows * kHiBytes, 0xAA);
      for (uint32_t row = 0; row < kTileRows; ++row) {
        for (uint32_t g = 0; g < kGroups; ++g) {
          uint8_t *lo4Dest = lo4Tile.data() + (uint64_t(g) * kTileRows + row) * kLo4Bytes;
          uint8_t *hiDest = hiTile.data() + (uint64_t(g) * kTileRows + row) * kHiBytes;
          if (row < kPartABoundary) {  // part A, bits=4
            std::array<uint8_t, kLo4Bytes> q4Lo4{};
            for (auto &b : q4Lo4) b = static_cast<uint8_t>(rng());
            promoteQ4Group(q4Lo4.data(), lo4Dest, hiDest);
            for (uint32_t i = 0; i < kGroupElements; ++i)
              expected[(g * kTileRows + row) * kGroupElements + i] =
                  (i % 2 == 0) ? (q4Lo4[i / 2] & 0xF) : (q4Lo4[i / 2] >> 4);
          } else if (row < kSectionRows) {  // part B, bits=5
            std::array<uint8_t, kMlxGroupBytes> group40{};
            std::array<uint32_t, kGroupElements> values{};
            for (auto &v : values) v = codeDist(rng);
            for (uint32_t i = 0; i < kGroupElements; ++i) packMlxQ5(group40.data(), i, values[i]);
            splitMlxQ5Group(group40.data(), lo4Dest, hiDest);
            for (uint32_t i = 0; i < kGroupElements; ++i)
              expected[(g * kTileRows + row) * kGroupElements + i] = values[i];
          } else {  // padding: zero, gatherQ5Codes never writes it
            std::memset(lo4Dest, 0, kLo4Bytes);
            std::memset(hiDest, 0, kHiBytes);
          }
        }
      }
      for (uint32_t row = 0; row < kTileRows; ++row)
        for (uint32_t g = 0; g < kGroups; ++g) {
          const uint8_t *lo4Src = lo4Tile.data() + (uint64_t(g) * kTileRows + row) * kLo4Bytes;
          const uint8_t *hiSrc = hiTile.data() + (uint64_t(g) * kTileRows + row) * kHiBytes;
          for (uint32_t i = 0; i < kGroupElements; ++i)
            require(valueFromSplit(lo4Src, hiSrc, i) == expected[(g * kTileRows + row) * kGroupElements + i],
                    "mixed-section tile mismatch at row " + std::to_string(row));
        }
    }

    std::cout << "q5_pack_test: OK\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "q5_pack_test failed: " << error.what() << "\n";
    return 1;
  }
}
