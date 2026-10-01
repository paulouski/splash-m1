#include "metal/abi/Linear.h"
#include <metal_stdlib>
using namespace metal;

// Q5/g64 variant of linear_q4_mma.metal (unmodified): identical 32-row,
// four-simdgroup register-matrix prefill tile, extended to read the extra
// "hi" plane (Q5Pack.hpp) and fold its bit into the nibble word before the
// existing nibble|0x6400 exact-half trick. ponytail: full duplication of
// q4mm::project rather than a templated <Bits> parameter, to avoid touching
// the half-scale worker's mid-edit linear_q4_mma.metal.
//
// hi addressing, derived from this kernel's own w[i]/pairs[i] layout (not
// copied from the decode kernel's derivation, which uses a different lane
// map): laneWeights reads one 8-byte (uint2, 16 values) lo4 chunk per column
// fragment i, re-packed into pairs[i] so kb (0..7) selects, via
// (word >> 4*(kb&3)) & 0x000F000F, the pair of values (kb, kb+8) -- the low
// mask lane always holds value kb, the high mask lane always holds value
// kb+8, for every kb (not split by kb<4 the way the repacking bytes are).
// hi's 8-byte lo4 chunk maps to a 2-byte hi chunk (Q5Pack.hpp: hi is a
// quarter the lo4 plane's stride, same tile/group/column addressing): byte
// .x covers values 0..7 (bit kb = value kb's fifth bit), byte .y covers
// values 8..15 (bit kb = value (8+kb)'s fifth bit). So for every kb the two
// hi bits OR in at bit 4 (low lane, value kb) and bit 20 (high lane, value
// kb+8) of the already-masked nibble word -- two independent shift+mask
// ops, no dependency chain (RESULTS.md's M1 root-cause fix).
namespace q5mm {

constexpr constant uint kStorageRows = 32; // storage rows per threadgroup
constexpr constant uint kGroup = 64;      // inputs per quant group
constexpr constant uint kStorageN = 256;  // weight tile width of the packing
constexpr constant uint FN = 4;           // fragments per simdgroup: columns
constexpr constant uint SGN = 2;          // simdgroups per threadgroup: columns
constexpr constant uint kColumns = SGN * FN * 8;
static_assert(kStorageN % kColumns == 0 && kColumns % kGroup == 0);

enum class Epilogue { Affine, Residual, UpWithGate };

template <uint ActiveRows, bool Hi, Epilogue E>
inline void project(device const bfloat *input, device const uchar *weights, device const uchar *hi,
                    device const half *scales, device const half *biases,
                    device bfloat *output, device const bfloat *auxiliary,
                    device const float *sums, uint N, uint K, uint2 tg,
                    uint sg, uint lane) {
  constexpr uint FM = ActiveRows == 32 ? 2 : 1;
  constexpr uint SGM = ActiveRows / (FM * 8);
  static_assert((ActiveRows == 8 || ActiveRows == 16 || ActiveRows == 32) &&
                SGM * FM * 8 == ActiveRows);
  const uint groups = K / kGroup;
  const uint n0 = tg.y * kColumns;
  const uint tile = n0 / kStorageN;
  const uint sgn = sg % SGN, sgm = sg / SGN;
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const uint nBase = sgn * FN * 8;
  const uint rowBase = sgm * FM * 8 + fn;
  const uint nLocal = (n0 + nBase) % kStorageN + fm;
  device const uchar *laneWeights = weights +
      (ulong(tile) * groups * kStorageN + nLocal) * (kGroup / 2) + (fn / 2) * 8;
  device const uchar *laneHi = hi +
      (ulong(tile) * groups * kStorageN + nLocal) * (kGroup / 8) + (fn / 2) * 2;
  device const half *laneScales = scales + ulong(tile) * groups * kStorageN + nLocal;
  device const half *laneBiases = biases + ulong(tile) * groups * kStorageN + nLocal;
  input += (ulong(tg.x) * kStorageRows + rowBase) * K + 8 * fm;
  sums += ulong(tg.x) * kStorageRows * groups + rowBase;
  output += ulong(tg.x) * kStorageRows * N;
  auxiliary += ulong(tg.x) * kStorageRows * N;

  float2 acc[FN][FM];
#pragma unroll
  for (uint i = 0; i < FN; ++i)
#pragma unroll
    for (uint j = 0; j < FM; ++j) acc[i][j] = float2(0);

  uint2 w[FN];
  uchar2 hw[FN];
#pragma unroll
  for (uint i = 0; i < FN; ++i) {
    w[i] = *reinterpret_cast<device const uint2 *>(laneWeights + i * 8 * 32);
    if constexpr (Hi) hw[i] = *reinterpret_cast<device const uchar2 *>(laneHi + i * 8 * 8);
  }
  for (uint g = 0; g < groups; ++g) {
    uint4 x[FM][2];
#pragma unroll
    for (uint j = 0; j < FM; ++j)
#pragma unroll
      for (uint e = 0; e < 2; ++e)
        x[j][e] = *reinterpret_cast<device const uint4 *>(
            input + ulong(j * 8 + e) * K + g * kGroup);
    uint2 pairs[FN];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
      pairs[i] = uint2((w[i].x & 0xFFFFu) | (w[i].y << 16),
                       (w[i].x >> 16) | (w[i].y & 0xFFFF0000u));
    uint hiBits[FN];
    if constexpr (Hi) {
#pragma unroll
      for (uint i = 0; i < FN; ++i) hiBits[i] = (uint(hw[i].x) << 4) | (uint(hw[i].y) << 20);
    }
    float2 dot[FN][FM];
#pragma unroll
    for (uint i = 0; i < FN; ++i)
#pragma unroll
      for (uint j = 0; j < FM; ++j) dot[i][j] = float2(0);
#pragma unroll
    for (uint kb = 0; kb < 8; ++kb) {
      simdgroup_float8x8 b[FM];
#pragma unroll
      for (uint j = 0; j < FM; ++j) {
        const uint shift = (kb & 1) ? 0 : 16;
        reinterpret_cast<thread float2 &>(b[j].thread_elements()) =
            float2(as_type<float>((x[j][0][kb >> 1] << shift) & 0xFFFF0000u),
                   as_type<float>((x[j][1][kb >> 1] << shift) & 0xFFFF0000u));
      }
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        const uint word = kb < 4 ? pairs[i].x : pairs[i].y;
        uint nibbles = (word >> (4 * (kb & 3))) & 0x000F000Fu;
        if constexpr (Hi) nibbles |= (hiBits[i] >> kb) & 0x00100010u;
        simdgroup_half8x8 a;
        if constexpr (Hi && ActiveRows == 32) {
          reinterpret_cast<thread half2 &>(a.thread_elements()) =
              as_type<half2>(nibbles | 0x64006400u);
        } else {
          reinterpret_cast<thread half2 &>(a.thread_elements()) =
              as_type<half2>(nibbles | 0x64006400u) - half2(1024.0h);
        }
#pragma unroll
        for (uint j = 0; j < FM; ++j) {
          simdgroup_float8x8 c, d;
          reinterpret_cast<thread float2 &>(c.thread_elements()) = dot[i][j];
          simdgroup_multiply_accumulate(d, a, b[j], c);
          dot[i][j] = reinterpret_cast<thread float2 &>(d.thread_elements());
        }
      }
    }
    float scale[FN], bias[FN];
#pragma unroll
    for (uint i = 0; i < FN; ++i) {
      scale[i] = float(laneScales[ulong(g) * kStorageN + i * 8]);
      bias[i] = float(laneBiases[ulong(g) * kStorageN + i * 8]);
      if constexpr (Hi && ActiveRows == 32)
        bias[i] = fma(-1024.0f, scale[i], bias[i]);
    }
    if (g + 1 < groups) {
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        w[i] = *reinterpret_cast<device const uint2 *>(
            laneWeights + (ulong(g + 1) * kStorageN + i * 8) * 32);
        if constexpr (Hi)
          hw[i] = *reinterpret_cast<device const uchar2 *>(
              laneHi + (ulong(g + 1) * kStorageN + i * 8) * 8);
      }
    }
#pragma unroll
    for (uint j = 0; j < FM; ++j) {
      const float2 sum = float2(sums[g * kStorageRows + j * 8],
                                sums[g * kStorageRows + j * 8 + 1]);
#pragma unroll
      for (uint i = 0; i < FN; ++i) {
        acc[i][j] = fma(dot[i][j], scale[i], acc[i][j]);
        acc[i][j] = fma(sum, bias[i], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (uint i = 0; i < FN; ++i) {
    const uint n = n0 + nBase + i * 8 + fm;
#pragma unroll
    for (uint j = 0; j < FM; ++j) {
#pragma unroll
      for (uint e = 0; e < 2; ++e) {
        const uint m = rowBase + j * 8 + e;
        float value = float(bfloat(acc[i][j][e]));
        if (E == Epilogue::UpWithGate) {
          const float gate = float(auxiliary[m * N + n]);
          value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * value;
        } else if (E == Epilogue::Residual) {
          value += float(auxiliary[m * N + n]);
        }
        output[m * N + n] = bfloat(value);
      }
    }
  }
}

// Same as q4mm::write_output_sums (kColumns/N addressing is Q4/Q5-independent).
template <uint ActiveRows>
inline void write_output_sums(device const bfloat *output, device float *outputSums,
                              uint N, uint2 tg, uint sg, uint lane) {
  constexpr uint Groups = kColumns / kGroup;
  constexpr uint FM = ActiveRows == 32 ? 2 : 1;
  constexpr uint kSimdgroups = SGN * (ActiveRows / (FM * 8));
  threadgroup_barrier(mem_flags::mem_device);
  output += ulong(tg.x) * kStorageRows * N;
  outputSums += ulong(tg.x) * kStorageRows * (N / kGroup);
  const uint n0 = tg.y * kColumns;
  for (uint task = sg; task < ActiveRows * Groups; task += kSimdgroups) {
    const uint row = task / Groups, local = task % Groups;
    const uint origin = row * N + n0 + local * kGroup + lane;
    const float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (lane == 0) outputSums[(n0 / kGroup + local) * kStorageRows + row] = sum;
  }
}

} // namespace q5mm

// The tile is threadgroup-uniform: outside the projection's nonzero-hi range
// the hi plane is skipped (all zero, so the result is unchanged).
#define Q5MM_PROJECT(E, output, aux) \
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.hi_tile_begin && \
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.hi_tile_end) \
    q5mm::project<32, true, q5mm::Epilogue::E>(input, weights, hi, scales, biases, output, aux, \
                                           sums, p.output_size, p.input_size, tg, sg, lane); \
  else \
    q5mm::project<32, false, q5mm::Epilogue::E>(input, weights, hi, scales, biases, output, aux, \
                                            sums, p.output_size, p.input_size, tg, sg, lane)

#define Q5MM_THREADS                                                          \
  uint2 tg [[threadgroup_position_in_grid]],                                  \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]

kernel void prefill_linear_q5_mma64(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4PrefillParams &p [[buffer(7)]], Q5MM_THREADS) {
  Q5MM_PROJECT(Affine, output, output);
}

kernel void prefill_linear_q5_mma64_residual(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *residual [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    constant Q4PrefillParams &p [[buffer(8)]], Q5MM_THREADS) {
  Q5MM_PROJECT(Residual, output, residual);
}

kernel void prefill_linear_q5_mma64_up_silu_sums(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *gate [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    device float *outputSums [[buffer(8)]], constant Q4PrefillParams &p [[buffer(9)]], Q5MM_THREADS) {
  Q5MM_PROJECT(UpWithGate, output, gate);
  q5mm::write_output_sums<32>(output, outputSums, p.output_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_m8(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4PrefillTailParams &p [[buffer(7)]],
    Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<8, true, q5mm::Epilogue::Affine>(input, weights, hi, scales, biases,
        output, output, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<8, false, q5mm::Epilogue::Affine>(input, weights, hi, scales, biases,
        output, output, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_m16(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device const float *sums [[buffer(6)]], constant Q4PrefillTailParams &p [[buffer(7)]],
    Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<16, true, q5mm::Epilogue::Affine>(input, weights, hi, scales, biases,
        output, output, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<16, false, q5mm::Epilogue::Affine>(input, weights, hi, scales, biases,
        output, output, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_residual_m8(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *residual [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    constant Q4PrefillTailParams &p [[buffer(8)]], Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<8, true, q5mm::Epilogue::Residual>(input, weights, hi, scales, biases,
        output, residual, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<8, false, q5mm::Epilogue::Residual>(input, weights, hi, scales, biases,
        output, residual, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_residual_m16(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *residual [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    constant Q4PrefillTailParams &p [[buffer(8)]], Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<16, true, q5mm::Epilogue::Residual>(input, weights, hi, scales, biases,
        output, residual, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<16, false, q5mm::Epilogue::Residual>(input, weights, hi, scales, biases,
        output, residual, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_up_silu_sums_m8(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *gate [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    device float *outputSums [[buffer(8)]], constant Q4PrefillTailParams &p [[buffer(9)]],
    Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<8, true, q5mm::Epilogue::UpWithGate>(input, weights, hi, scales, biases,
        output, gate, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<8, false, q5mm::Epilogue::UpWithGate>(input, weights, hi, scales, biases,
        output, gate, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  q5mm::write_output_sums<8>(output, outputSums, p.projection.output_size, tg, sg, lane);
}

kernel void prefill_linear_q5_mma64_up_silu_sums_m16(
    device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    device const uchar *hi [[buffer(2)]], device const half *scales [[buffer(3)]],
    device const half *biases [[buffer(4)]], device const bfloat *gate [[buffer(5)]],
    device bfloat *output [[buffer(6)]], device const float *sums [[buffer(7)]],
    device float *outputSums [[buffer(8)]], constant Q4PrefillTailParams &p [[buffer(9)]],
    Q5MM_THREADS) {
  tg.x += p.row_tile_offset;
  if (tg.y * q5mm::kColumns / q5mm::kStorageN >= p.projection.hi_tile_begin &&
      tg.y * q5mm::kColumns / q5mm::kStorageN < p.projection.hi_tile_end)
    q5mm::project<16, true, q5mm::Epilogue::UpWithGate>(input, weights, hi, scales, biases,
        output, gate, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  else
    q5mm::project<16, false, q5mm::Epilogue::UpWithGate>(input, weights, hi, scales, biases,
        output, gate, sums, p.projection.output_size, p.projection.input_size, tg, sg, lane);
  q5mm::write_output_sums<16>(output, outputSums, p.projection.output_size, tg, sg, lane);
}

#undef Q5MM_PROJECT
#undef Q5MM_THREADS
