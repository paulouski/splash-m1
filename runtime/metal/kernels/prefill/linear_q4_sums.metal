#include "metal/abi/KernelABI.h"

// Row sums of a prefill chunk (Linear::addPrefillSums): plain simd_sum, no
// MPP, moved out of linear_q4.metal so it survives the MACOS15=1 build,
// which drops that file's MPP-only prefill tiles. Reached on Apple7/8 too
// (QwenTarget.cpp's affine mixer output projection, DFlashDraft.cpp's
// context projection).
kernel void prefill_linear_q4_sums32(device const bfloat *input [[buffer(0)]],
                              device float *sums [[buffer(1)]],
                              constant uint &input_size [[buffer(2)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group
                              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  uint quant_groups = input_size / 64;
  input += ulong(tile) * TileM * input_size;
  sums += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      uint origin = row * input_size + quant_group * 64 + simd_lane;
      float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
      if (simd_lane == 0) {
        sums[quant_group * TileM + row] = sum;
      }
    }
  }
}
