#pragma once
#include "metal/kernels/common/quant_formats.h"

// The staged kernels' dequantization (kernels/shared/gguf_linear.metal and
// kernels/shared/moe_gguf.metal, through kernels/common/gguf_staged_tile.h),
// shared with the dequantization test. Includers set `#pragma clang fp
// reassociate(off)` first, so the source order of float operations holds here.
// One thread writes one column's group of 32 (kernels/common/quant_formats.h) as half, each value rounded once;
// chunk c's pairs 0, 1 go to dst + 4c and pairs 2, 3 to dst + 16 + 4c.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
// The coefficients of a group (j of its meta unit), whose chunk q it scales: with ScaleInChunk its scale is in every
// chunk of the group.
template <class F>
inline QuantCoef group_coef(typename F::Meta meta, typename F::Chunk q, ushort j) {
  if constexpr (F::ScaleInChunk) return F::coef(meta, q); else return F::coef(meta, j);
}
// Chunk q's elements 4c..4c+3 (lo) and 16+4c..16+4c+3 (hi) of its group, whose coefficients are k.
template <class F>
inline void dequant_chunk(typename F::Chunk q, QuantCoef k, threadgroup half2 *tl, thread half4 &lo, thread half4 &hi) {
  if constexpr (F::Kind == QuantLinear) {
    const uint4 p = F::codes(q);
    lo = half4(staged_linear<F>(p.x, k.s.x, k.m.x), staged_linear<F>(p.y, k.s.x, k.m.x));
    hi = half4(staged_linear<F>(p.z, k.s.y, k.m.y), staged_linear<F>(p.w, k.s.y, k.m.y));
  } else if constexpr (F::Kind == QuantCodebook) {   // value * s in Scale: one rounding to half either way
    typedef typename F::Scale S;
    const uchar4 b = as_type<uchar4>(F::indices(q));
    lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
    hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
  } else if constexpr (F::Kind == QuantInt8) {
    typedef typename F::Scale S;
    const uint2 v = F::values(q);
    lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
    hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
  } else {
    const uint2 g = F::grid(q); const uint s = F::signs(q);
    lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
    hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
    lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
    hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
  }
}
// The group's values chunk by chunk: put(c, lo, hi) takes chunk c's elements.
template <class F, class Put>
inline void dequant32_chunks(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, Put put) {
  const QuantCoef k = group_coef<F>(meta, F::chunk(w, 0), j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    half4 lo, hi;
    dequant_chunk<F>(F::chunk(w, c), k, tl, lo, hi);
    put(c, lo, hi);
  }
}
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, threadgroup half *dst) {
  dequant32_chunks<F>(w, meta, j, tl, [&](ushort c, half4 lo, half4 hi) {
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  });
}
