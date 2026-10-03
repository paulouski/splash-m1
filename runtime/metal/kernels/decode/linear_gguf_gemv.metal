// One-row (M=1) GEMV over PQ2_0 prepared planes: two output rows per thread, 128 threads per 256 rows, so a simdgroup reads
// 256 contiguous bytes of plane0 per 32-group per row pair. gguf_gemv_stage lays row 0 of the bf16 input out by code slot
// (scaled so float(word & mask) * x is q * x) with the sum of each 128-group; the GEMV reads it through the constant
// cache and applies value = d * (q - 1) as d * (sum q x - sum x) once per 128-group. Grid (row block, K split);
// splits > 1 write fp32 partials [matrix][split][row] (row stride out_stride) that gguf_gemv_reduce sums and finishes.
// Row 0 of a tensor lands at output column p.out_offset.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/gguf_tile.h"

enum GemvEpilogue : ushort { GemvNone, GemvResidual };

// Code s (0..15) of a word as the float 1 + q 2^-7 (s % 3 = 0), 1 + q 2^-5 (1) or 1 + q 2^-3 (2): its 2 bits moved to
// bit 16 + 2 (s % 3) of a mantissa by the shift of its group of three, so no integer-to-float conversion is needed. The
// staged x is scaled by 2^7, 2^5, 2^3 to match, and the offset sum x * scale is taken off the group's sum.
__attribute__((always_inline)) inline float gemv_code(uint word, uint s) {
  const uint group = s / 3, bit = 16 + 2 * (s % 3);
  const uint t = group <= 2 ? word << (16 - 6 * group) : word >> (6 * group - 16);
  return as_type<float>((t & (3u << bit)) | 0x3F800000u);
}

// silu(gate) * up of the bf16-rounded gate and up sums, as the M=8 gate/up epilogue.
inline bfloat gemv_gate_up(float gate, float up) { return bfloat(float(bfloat(up)) * splash_silu(float(bfloat(gate)))); }

template <GemvEpilogue Ep, class Out>
inline void gemv_store(device Out *output, device bfloat *aux, uint row, float v) {
  if constexpr (Ep == GemvResidual) v += float(aux[row]);
  output[row] = Out(v);
}

// Stages row 0 of the bf16 input by slot for gguf_gemv_pq20: lane = (32-group g, word, quad j) of a 128-group holds
// slots 4j..4j+3 of the word's 16 codes, each scaled by 4^-s so that float(word & (3 << 2s)) * x is q * x exactly;
// then the sum of each 128-group. One simdgroup per 128-group, 32 float4 per group followed by U sums.
kernel void gguf_gemv_stage(device bfloat *input [[buffer(0)]], device uchar *staged [[buffer(1)]],
                            constant GgufDecodeParams &p [[buffer(2)]], uint u [[simdgroup_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  u += group.x * 8;
  const uint U = p.input_size / 128;
  if (u >= U) return;
  const uint g = simd_lane >> 3, wd = (simd_lane >> 2) & 1, j = simd_lane & 3, base = u * 128 + g * 32;
  float4 x, v;
  float offset = 0.0f;
  for (uint t = 0; t < 4; ++t) {
    const uint ss = 16 * wd + 4 * j + t, e = (((ss >> 3) & 3) << 2) | (ss & 3) | (((ss >> 2) & 1) << 4);
    v[t] = float(input[base + e]);
    x[t] = v[t] * as_type<float>((127u + 7 - 2 * ((4 * j + t) % 3)) << 23);
    offset += x[t];
  }
  ((device float4 *)staged)[u * 32 + simd_lane] = x;
  const float sum = simd_sum(v.x + v.y + v.z + v.w + offset);
  if (simd_lane == 0) ((device float *)(staged + U * 32 * 16))[u] = sum;
}


// The sums of Rows output rows of M matrices (M = 2: gate and up) over this threadgroup's K split, Rows * Threads = 256
// rows of a tile per threadgroup. total[m][r] is row group.x * 256 + r * Threads + tid.
template <uint Rows, uint M>
inline void gemv_tile(constant uchar *staged, device uchar *w0a, device uchar *metaa, device uchar *w0b,
                      device uchar *metab, constant GgufDecodeParams &p, uint split, uint block, uint tid,
                      thread float (&total)[M][Rows]) {
  using F = FmtPQ20;
  constexpr uint T = QUANT_TILE_ROWS / Rows;
  const uint K = p.input_size, G = K / 32, U = K / 128, per = U / p.splits, u0 = split * per;
  constant float4 *xs = (constant float4 *)(staged + u0 * 32 * 16);
  constant float *sums = (constant float *)(staged + U * 32 * 16) + u0;
  const uint row0 = block * QUANT_TILE_ROWS + tid;
  for (uint m = 0; m < M; ++m)
    for (uint r = 0; r < Rows; ++r) total[m][r] = 0.0f;
#pragma unroll 1
  for (uint u = 0; u < per; ++u) {
    float d[M][Rows];
    float4 acc[M][Rows] = {};
    F::Payload pls[4][M][Rows];
    for (uint m = 0; m < M; ++m)
      for (uint r = 0; r < Rows; ++r) {
        const uint row = row0 + r * T, tile = row / QUANT_TILE_ROWS, at = row % QUANT_TILE_ROWS;
        device uchar *wm = m ? w0b : w0a, *mm = m ? metab : metaa;
        d[m][r] = float(as_type<half>(F::loadMeta(mm + ((ulong(tile) * U + u0 + u) * QUANT_TILE_ROWS + at) * 2)));
        for (uint g = 0; g < 4; ++g)
          pls[g][m][r] = F::load(wm + ((ulong(tile) * G + (u0 + u) * 4 + g) * QUANT_TILE_ROWS + at) * 8, nullptr);
      }
    for (uint g = 0; g < 4; ++g)
      for (uint wd = 0; wd < 2; ++wd)
        for (uint j = 0; j < 4; ++j) {
          const float4 x = xs[u * 32 + g * 8 + wd * 4 + j];
          for (uint m = 0; m < M; ++m)
            for (uint r = 0; r < Rows; ++r) {
              const uint word = pls[g][m][r].a[wd];
              acc[m][r].x = fma(gemv_code(word, 4 * j + 0), x.x, acc[m][r].x);
              acc[m][r].y = fma(gemv_code(word, 4 * j + 1), x.y, acc[m][r].y);
              acc[m][r].z = fma(gemv_code(word, 4 * j + 2), x.z, acc[m][r].z);
              acc[m][r].w = fma(gemv_code(word, 4 * j + 3), x.w, acc[m][r].w);
            }
        }
    for (uint m = 0; m < M; ++m)
      for (uint r = 0; r < Rows; ++r)
        total[m][r] = fma(d[m][r], (acc[m][r].x + acc[m][r].y) + (acc[m][r].z + acc[m][r].w) - sums[u], total[m][r]);
  }
}

// Stores the sums: one split directly; more publish fp32 partials [matrix][split][row].
template <uint Rows, uint M, class Store>
inline void gemv_finish(thread float (&total)[M][Rows], constant GgufDecodeParams &p, uint split, uint block, uint tid,
                        device float *partials, Store store) {
  constexpr uint T = QUANT_TILE_ROWS / Rows;
  const uint row0 = block * QUANT_TILE_ROWS + tid;
  if (p.splits == 1) {
    for (uint r = 0; r < Rows; ++r) store(row0 + r * T, total, r);
    return;
  }
  for (uint m = 0; m < M; ++m)
    for (uint r = 0; r < Rows; ++r) partials[(ulong(m) * p.splits + split) * p.out_stride + row0 + r * T] = total[m][r];
}

template <GemvEpilogue Ep, class Out>
kernel void gguf_gemv_pq20(constant uchar *staged [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                           device uchar *meta [[buffer(2)]], device Out *output [[buffer(3)]],
                           device float *partials [[buffer(4)]], device bfloat *aux [[buffer(5)]],
                           constant GgufDecodeParams &p [[buffer(6)]],
                           uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint tid = simd_group * 32 + simd_lane;
  float total[1][2];
  gemv_tile<2, 1>(staged, w0, meta, w0, meta, p, group.y, group.x, tid, total);
  gemv_finish<2, 1>(total, p, group.y, group.x, tid, partials, [&](uint row, thread float (&t)[1][2], uint r) {
    gemv_store<Ep, Out>(output + p.out_offset, aux, row, t[0][r]);
  });
}

kernel void gguf_gemv_pq20_g(constant uchar *staged [[buffer(0)]], device uchar *w0g [[buffer(1)]],
                             device uchar *metag [[buffer(2)]], device uchar *w0u [[buffer(3)]],
                             device uchar *metau [[buffer(4)]], device bfloat *output [[buffer(5)]],
                             device float *partials [[buffer(6)]], constant GgufDecodeParams &p [[buffer(7)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint tid = simd_group * 32 + simd_lane;
  float total[2][1];
  gemv_tile<1, 2>(staged, w0g, metag, w0u, metau, p, group.y, group.x, tid, total);
  gemv_finish<1, 2>(total, p, group.y, group.x, tid, partials, [&](uint row, thread float (&t)[2][1], uint) {
    output[p.out_offset + row] = gemv_gate_up(t[0][0], t[1][0]);
  });
}

template <GemvEpilogue Ep, class Out>
kernel void gguf_gemv_reduce(device float *partials [[buffer(0)]], device Out *output [[buffer(1)]],
                             device bfloat *aux [[buffer(2)]], constant GgufDecodeParams &p [[buffer(3)]],
                             uint row [[thread_position_in_grid]]) {
  float v = 0.0f;
  for (uint s = 0; s < p.splits; ++s) v += partials[ulong(s) * p.out_stride + row];
  gemv_store<Ep, Out>(output + p.out_offset, aux, row, v);
}

kernel void gguf_gemv_reduce_g(device float *partials [[buffer(0)]], device bfloat *output [[buffer(1)]],
                               constant GgufDecodeParams &p [[buffer(2)]], uint row [[thread_position_in_grid]]) {
  float g = 0.0f, u = 0.0f;
  for (uint s = 0; s < p.splits; ++s) {
    g += partials[ulong(s) * p.out_stride + row];
    u += partials[ulong(p.splits + s) * p.out_stride + row];
  }
  output[p.out_offset + row] = gemv_gate_up(g, u);
}

template <class Out>
using GemvKernel = void(constant uchar *, device uchar *, device uchar *, device Out *, device float *, device bfloat *,
                        constant GgufDecodeParams &, uint2, uint, uint);
template <class Out>
using GemvReduceKernel = void(device float *, device Out *, device bfloat *, constant GgufDecodeParams &, uint);
#define GGUF_GEMV(ep, Ep, Out)                                                                                    \
  template [[host_name("gguf_gemv_pq20_" #ep)]] kernel GemvKernel<Out> gguf_gemv_pq20<Ep, Out>;                  \
  template [[host_name("gguf_gemv_reduce_" #ep)]] kernel GemvReduceKernel<Out> gguf_gemv_reduce<Ep, Out>;
GGUF_GEMV(a, GemvNone, bfloat)
GGUF_GEMV(a_f32, GemvNone, float)
GGUF_GEMV(r, GemvResidual, bfloat)
#undef GGUF_GEMV
