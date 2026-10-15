#pragma once
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// Expert tiles of one or two live rows on registers instead of the simdgroup MMA (kernels/shared/moe_gguf_mma.metal): a lane
// owns an output column and dequantizes its weights in registers, the rows' bf16 inputs stay in threadgroup memory, and K is
// split over simdgroups. Products of a half weight and a bf16 input are exact in fp32, so only the order of the fp32 sums
// differs from the MMA tile. Includers set `#pragma clang fp reassociate(off)` first.
namespace gguf_mma {

constant constexpr uint kDotRows = 2;  // tiles with at most this many live rows run here

inline float4 bf16x4(uint2 v) {
  return float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xFFFF0000u), as_type<float>(v.y << 16),
                as_type<float>(v.y & 0xFFFF0000u));
}

// acc[r][j][c] += columns origin + 32 j over steps [step_begin, step_end); xs: row r's bf16 inputs from step_begin at xs + r * xstride.
template <class F, ushort R, ushort CPL>
inline void dot_steps(threadgroup const ushort *xs, uint xstride, device uchar *w0, device uchar *w1, device uchar *meta,
                      uint input_size, uint origin, threadgroup half2 *tl, uint lane, uint step_begin, uint step_end,
                      thread float (&acc)[R][CPL][4]) {
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  device uchar *c0[CPL], *c1[CPL], *cm[CPL];
  typename F::Meta hdr[CPL];
#pragma unroll
  for (ushort j = 0; j < CPL; ++j) {
    const uint o = origin + 32 * j;
    const uint plane_tile = o / QUANT_TILE_ROWS, plane_row = o % QUANT_TILE_ROWS + lane;
    c0[j] = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
    c1[j] = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
    cm[j] = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
    hdr[j] = F::loadMeta(cm[j] + ulong(step_begin / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  }
  for (uint step = step_begin; step < step_end; ++step) {
    typename F::Payload packed[CPL];
#pragma unroll
    for (ushort j = 0; j < CPL; ++j) {
      packed[j] = F::load(c0[j] + ulong(step) * QUANT_TILE_ROWS * F::P0, c1[j] + ulong(step) * QUANT_TILE_ROWS * F::P1);
      if (step > step_begin && step % F::MetaGroups == 0)
        hdr[j] = F::loadMeta(cm[j] + ulong(step / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
    }
    threadgroup const ushort *xp = xs + (step - step_begin) * 32;
    QuantCoef k[CPL];
#pragma unroll
    for (ushort j = 0; j < CPL; ++j) k[j] = group_coef<F>(hdr[j], F::chunk(packed[j], 0), ushort(step % F::MetaGroups));
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
      float4 xl[R], xh[R];
#pragma unroll
      for (ushort r = 0; r < R; ++r) {
        xl[r] = bf16x4(*reinterpret_cast<threadgroup const uint2 *>(xp + r * xstride + 4 * c));
        xh[r] = bf16x4(*reinterpret_cast<threadgroup const uint2 *>(xp + r * xstride + 16 + 4 * c));
      }
#pragma unroll
      for (ushort j = 0; j < CPL; ++j) {
        half4 lo, hi;
        dequant_chunk<F, false>(F::chunk(packed[j], c), k[j], tl, lo, hi);
        const float4 wl = float4(lo), wh = float4(hi);
#pragma unroll
        for (ushort r = 0; r < R; ++r) {
          float a = acc[r][j][c];
          a = fma(wl.x, xl[r].x, a); a = fma(wl.y, xl[r].y, a); a = fma(wl.z, xl[r].z, a); a = fma(wl.w, xl[r].w, a);
          a = fma(wh.x, xh[r].x, a); a = fma(wh.y, xh[r].y, a); a = fma(wh.z, xh[r].z, a); a = fma(wh.w, xh[r].w, a);
          acc[r][j][c] = a;
        }
      }
    }
  }
}

// The R rows' K bf16 inputs into xs, by all THREADS threads of the threadgroup.
template <ushort R, uint THREADS>
inline void dot_stage_inputs(device const bfloat *const (&xrow)[R], uint K, threadgroup ushort *xs, uint thread_index) {
  const uint vecs = K / 8;
  for (uint i = thread_index; i < R * vecs; i += THREADS) {
    const uint r = i / vecs, v = i % vecs;
    device const bfloat *src = xrow[0];
#pragma unroll
    for (ushort q = 1; q < R; ++q) src = r == q ? xrow[q] : src;
    *reinterpret_cast<threadgroup uint4 *>(xs + r * K + v * 8) = *reinterpret_cast<device const uint4 *>(src + v * 8);
  }
}

// Gate and up (SwiGLU epilogue of the gu_k2 tile) of a tile of kDotRows rows or fewer, on the threadgroup memory `raw`
// (>= kDotRows * input_size * 2 bytes). Simdgroups: {gate, up} x CG column groups x KS K slices, CPL columns per lane;
// 2 KS CG * 32 threads and CG * CPL * 32 columns per threadgroup.
template <class FG, ushort KS, ushort CG, ushort CPL>
inline void moe_gate_up_dot(device const bfloat *input, device const uint *routes, const MoeTileDescriptor tile,
                            device uchar *gw0, device uchar *gw1, device uchar *gmeta, device uchar *gsw0, device uchar *gsw1,
                            device uchar *gsmeta, device uchar *uw0, device uchar *uw1, device uchar *umeta, device uchar *usw0,
                            device uchar *usw1, device uchar *usmeta, device bfloat *output, constant MoeGgufGateUpParams &p,
                            uint2 group, uint simd_lane, uint simd_group, threadgroup uint *raw) {
  constexpr uint THREADS = 2 * KS * CG * 32, CW = CG * CPL * 32;
  threadgroup ushort *xs = (threadgroup ushort *)raw;
  threadgroup float *part = (threadgroup float *)raw;
  const uint thread_index = simd_group * 32 + simd_lane;
  const uint K = p.gate.input_size;
  const uint up = simd_group / (KS * CG), rem = simd_group % (KS * CG), cg = rem / KS, s = rem % KS;
  const MoeGgufSegment seg = moe_gguf_segment_for<true>(tile.expert, up ? p.up : p.gate, up ? uw0 : gw0, up ? uw1 : gw1,
                                                        up ? umeta : gmeta, up ? usw0 : gsw0, up ? usw1 : gsw1, up ? usmeta : gsmeta);
  const uint origin = group.x * CW + cg * 32 * CPL + moe_gguf_first_row(tile.expert, up ? p.up : p.gate);
  const uint steps = K / 32, per = steps / KS;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    device const bfloat *xrow[R];
#pragma unroll
    for (ushort r = 0; r < R; ++r) xrow[r] = input + ulong(routes[tile.first_row + r] / p.routes_per_row) * K;
    dot_stage_inputs<R, THREADS>(xrow, K, xs, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[R][CPL][4];
#pragma unroll
    for (ushort r = 0; r < R; ++r)
#pragma unroll
      for (ushort j = 0; j < CPL; ++j)
#pragma unroll
        for (ushort c = 0; c < 4; ++c) acc[r][j][c] = 0.0f;
    const uint begin = s * per, end = s == KS - 1 ? steps : begin + per;
    const auto body = [&](auto format) {
      dot_steps<decltype(format), R, CPL>(xs + begin * 32, K, seg.w0, seg.w1, seg.meta, K, origin, (threadgroup half2 *)raw,
                                          simd_lane, begin, end, acc);
    };
    if (seg.format == uint(FG::Id)) body(FG());
    else body(FmtQ80());
    threadgroup_barrier(mem_flags::mem_threadgroup);   // xs is dead: the partial sums reuse it
#pragma unroll
    for (ushort r = 0; r < R; ++r)
#pragma unroll
      for (ushort j = 0; j < CPL; ++j)
        part[((up * KS + s) * R + r) * CW + cg * 32 * CPL + 32 * j + simd_lane] =
            (acc[r][j][0] + acc[r][j][1]) + (acc[r][j][2] + acc[r][j][3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = thread_index; o < R * CW; o += THREADS) {
      const uint r = o / CW, col = o % CW;
      float g = 0.0f, u = 0.0f;
#pragma unroll
      for (ushort q = 0; q < KS; ++q) {
        g += part[(q * R + r) * CW + col];
        u += part[((KS + q) * R + r) * CW + col];
      }
      output[(ulong(tile.first_row) + r) * p.gate.output_size + group.x * CW + col] =
          bfloat(float(bfloat(u)) * splash_silu(float(bfloat(g))));
    }
  };
  if (tile.rows == 1) run(integral_constant<ushort, 1>{});
  else run(integral_constant<ushort, 2>{});
}

// Down of a tile of kDotRows rows or fewer: `raw` holds the inputs and then the partial sums (>= kDotRows * input_size * 2
// bytes), `tl` the codebook pair table. KS * CG * 32 threads, CG * CPL * 32 columns per threadgroup.
template <class FD, ushort KS, ushort CG, ushort CPL>
inline void moe_down_dot(device bfloat *input, const MoeTileDescriptor tile, device uchar *w0, device uchar *w1, device uchar *meta,
                         device uchar *sw0, device uchar *sw1, device uchar *smeta, device bfloat *output,
                         constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                         threadgroup uint *raw, threadgroup half2 *tl) {
  constexpr uint THREADS = KS * CG * 32, CW = CG * CPL * 32;
  threadgroup ushort *xs = (threadgroup ushort *)raw;
  threadgroup float *part = (threadgroup float *)raw;
  const uint thread_index = simd_group * 32 + simd_lane;
  const MoeGgufSegment seg = moe_gguf_segment_for<false>(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  const uint K = p.input_size, steps = K / 32, per = steps / KS;
  const uint cg = simd_group / KS, s = simd_group % KS;
  const uint origin = group.x * CW + cg * 32 * CPL;
  quant_pair_table<FD>(tl, thread_index, THREADS);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    device const bfloat *xrow[R];
#pragma unroll
    for (ushort r = 0; r < R; ++r) xrow[r] = input + ulong(tile.first_row) * K + ulong(r) * K;
    dot_stage_inputs<R, THREADS>(xrow, K, xs, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[R][CPL][4];
#pragma unroll
    for (ushort r = 0; r < R; ++r)
#pragma unroll
      for (ushort j = 0; j < CPL; ++j)
#pragma unroll
        for (ushort c = 0; c < 4; ++c) acc[r][j][c] = 0.0f;
    const uint begin = s * per, end = s == KS - 1 ? steps : begin + per;
    const auto body = [&](auto format) {
      dot_steps<decltype(format), R, CPL>(xs + begin * 32, K, seg.w0, seg.w1, seg.meta, K, origin, tl, simd_lane, begin, end, acc);
    };
    if (seg.format == uint(FD::Id)) body(FD());
    else body(FmtQ80());
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort r = 0; r < R; ++r)
#pragma unroll
      for (ushort j = 0; j < CPL; ++j)
        part[(s * R + r) * CW + cg * 32 * CPL + 32 * j + simd_lane] = (acc[r][j][0] + acc[r][j][1]) + (acc[r][j][2] + acc[r][j][3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = thread_index; o < R * CW; o += THREADS) {
      const uint r = o / CW, col = o % CW;
      float t = 0.0f;
#pragma unroll
      for (ushort q = 0; q < KS; ++q) t += part[(q * R + r) * CW + col];
      output[(ulong(tile.first_row) + r) * p.output_size + group.x * CW + col] = bfloat(t);
    }
  };
  if (tile.rows == 1) run(integral_constant<ushort, 1>{});
  else run(integral_constant<ushort, 2>{});
}

} // namespace gguf_mma
