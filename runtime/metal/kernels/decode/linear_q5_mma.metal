// Q5/g64 variant of linear_q4_mma.metal (unmodified): same SimdgroupF32
// register tile, extended to read the extra "hi" plane (Q5Pack.hpp) and fold
// its bit into the nibble word before the existing exact-half trick. See
// prefill/linear_q5_mma.metal's header for the shared hi addressing rule and
// its derivation. ponytail: full duplication of q4sgf::decode rather than a
// templated <Bits> parameter, to avoid touching the half-scale worker's
// mid-edit linear_q4_mma.metal; a later pass could fold the two back
// together once that edit lands.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/q4_sgmatrix.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

// Apple7/8 (M1/M2) register-matrix Q5 decode projection. Identical structure
// to q4sgf::decode (same table/sums/split/epilogue handling); the only
// difference is constructing the weight as a 5-bit value before the
// nibble|0x6400 exact-half trick:
//   hi plane layout (Q5Pack.hpp): one bit per weight, 8 bytes per (column,
//   group), mirroring the 32-byte lo4 plane at a quarter the stride. For the
//   8-byte lo4 chunk `w[f]` this kernel already reads (uint2, 16 values, h
//   selecting .x (values 0..7) or .y (values 8..15)), the matching 2-byte hi
//   chunk covers the same 16 values: hi.x holds values 0..7 (bit s = value
//   s's fifth bit), hi.y holds values 8..15 (bit s = value (8+s)'s fifth
//   bit). For a fixed (h, s) the nibble extraction already isolates value
//   h*8+s (low lane, mask bit 0..3) and value h*8+s+4 (high lane, mask bit
//   16..19), so the two hi bits OR in at bit 4 (low lane) and bit 20 (high
//   lane) of the same masked word, independent of each other (no dependency
//   chain, the fix RESULTS.md's M1 root-cause pass converged on).
namespace q5sgf {
using namespace q4sg;
using sgmatrix::te;
enum class Epilogue { Affine, Residual, GateUp };

__attribute__((always_inline)) inline void mma(thread float2 &c, half2 a, float2 b) {
  simdgroup_half8x8 A;
  simdgroup_float8x8 B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

template <bool Hi, Epilogue E, uint L, class Out>
__attribute__((always_inline)) inline void decodeTile(device const bfloat *table, device const uchar *w0,
                   device const uchar *hi0, device const half *sc0, device const half *bi0,
                   device Out *out, device const float *sums,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1, device const uchar *hi1,
                   device const half *sc1, device const half *bi1,
                   constant Q4PersistentParams &p, uint2 tg, uint tid, uint sg, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint N = p.output_size, K = p.input_size, groups = K / 64;
  const uint splits = p.groups;
  const uint first = tg.y * (groups / splits);
  const uint end = tg.y + 1 == splits ? groups : first + groups / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (gateUp ? 8 : 16);
  const uint tile = base / 256;
  auto stream = [&](uint f) { return gateUp ? f : 0u; };
  auto column = [&](uint f) { return base + fm + (gateUp ? 0u : f * 8); };
  // Per-lane weight/hi pointers (tile and column folded in once); the group offset stays 32-bit.
  device const uchar *lanes[2] = {
      w0 + ulong(tile) * groups * 8192 + (column(0) % 256) * 32 + c * 8,
      (gateUp ? w1 : w0) + ulong(tile) * groups * 8192 + (column(1) % 256) * 32 + c * 8};
  device const uchar *hiLanes[2] = {
      hi0 + ulong(tile) * groups * 2048 + (column(0) % 256) * 8 + c * 2,
      (gateUp ? hi1 : hi0) + ulong(tile) * groups * 2048 + (column(1) % 256) * 8 + c * 2};
  // 32-bit element offsets relative to the projection's own buffers.
  const uint prmTile = tile * groups * 256;
  uint2 words[2];
  uchar2 hiWords[2];
  auto load = [&](uint g) __attribute__((always_inline)) {
#pragma unroll
    for (uint f = 0; f < 2; ++f) {
      words[f] = *reinterpret_cast<device const uint2 *>(lanes[f] + g * 8192u);
      if constexpr (Hi)
        hiWords[f] = *reinterpret_cast<device const uchar2 *>(hiLanes[f] + g * 2048u);
    }
  };
  float2 acc[2][L];
#pragma unroll
  for (uint f = 0; f < 2; ++f)
#pragma unroll
    for (uint r = 0; r < L; ++r) acc[f][r] = float2(0);
  load(first);
  for (uint g = first; g < end; ++g) {
    uint2 w[2] = {words[0], words[1]};
    uchar2 hb[2] = {hiWords[0], hiWords[1]};
    if (g + 1 < end) load(g + 1);
    float2 dot[2][L];
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) dot[f][r] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
      vec<bfloat, 8> x[L];
#pragma unroll
      for (uint r = 0; r < L; ++r)
        x[r] = reinterpret_cast<device const vec<bfloat, 8> *>(
            table + r * K * kRows + g * kXtPerGroup)[(h * 8 + fm) * 4 + c];
      uint hiBits[2] = {0, 0};
      if constexpr (Hi) {
#pragma unroll
        for (uint f = 0; f < 2; ++f) {
          const uint hbyte = h ? hb[f].y : hb[f].x;
          hiBits[f] = ((hbyte & 0x0Fu) << 4) | ((hbyte & 0xF0u) << 16);
        }
      }
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        float2 b[L];
#pragma unroll
        for (uint r = 0; r < L; ++r) b[r] = float2(reinterpret_cast<thread bfloat2 *>(&x[r])[s]);
#pragma unroll
        for (uint f = 0; f < 2; ++f) {
          uint nibbles = ((h ? w[f].y : w[f].x) >> (4 * s)) & 0x000F000Fu;
          if constexpr (Hi) nibbles |= (hiBits[f] >> s) & 0x00100010u;
          const half2 a = as_type<half2>(nibbles | 0x64006400u);
#pragma unroll
          for (uint r = 0; r < L; ++r) mma(dot[f][r], a, b[r]);
        }
      }
    }
#pragma unroll
    for (uint f = 0; f < 2; ++f) {
      const uint prm = prmTile + g * 256 + column(f) % 256;
      const float scale = float((stream(f) ? sc1 : sc0)[prm]);
      // weights are 1024 + q: fold -1024 * scale into the bias (rowSums * bias).
      const float bias = fma(-1024.0f, scale, float((stream(f) ? bi1 : bi0)[prm]));
#pragma unroll
      for (uint r = 0; r < L; ++r) {
        device const float *rowSums = sums + r * (K / 8) + g * kRows + fn;
        acc[f][r] = fma(dot[f][r], scale, acc[f][r]);
        acc[f][r] = fma(float2(rowSums[0], rowSums[1]), bias, acc[f][r]);
      }
    }
  }
  if (splits > 1) {
    auto slot = [&](uint r, uint s, uint f) {
      return partials + (ulong(r) * splits * 2 + s * 2 + stream(f)) * 8 * N + column(f);
    };
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r) {
        const auto at = slot(r, tg.y, f);
        at[fn * N] = acc[f][r].x;
        at[(fn + 1) * N] = acc[f][r].y;
      }
    if (!split_arrive_last(counters + tg.x, splits, tid, arrival)) return;
#pragma unroll
    for (uint f = 0; f < 2; ++f)
#pragma unroll
      for (uint r = 0; r < L; ++r)
        acc[f][r] = split_sum(acc[f][r], tg.y, splits, [&](uint s) {
          const auto at = slot(r, s, f);
          return float2(at[fn * N], at[(fn + 1) * N]);
        });
    split_release(counters + tg.x, tid);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r) {
    device Out *o = out + ulong(r) * kRows * N;
    if (gateUp) {
      const uint n = column(0);
      const float2 gate = float2(bfloat2(acc[0][r])), up = float2(bfloat2(acc[1][r]));
      const float2 value = gate / (1.0f + fast::exp2(-1.44269504089f * gate)) * up;
      o[fn * N + n] = bfloat(value.x);
      o[(fn + 1) * N + n] = bfloat(value.y);
    } else {
      device const bfloat *res = residual + ulong(r) * kRows * N;
#pragma unroll
      for (uint f = 0; f < 2; ++f) {
        const uint n = column(f);
        float2 value = acc[f][r];
        if constexpr (!is_same_v<Out, float>) value = float2(bfloat2(value));
        if (E == Epilogue::Residual)
          value += float2(float(res[fn * N + n]), float(res[(fn + 1) * N + n]));
        o[fn * N + n] = Out(value.x);
        o[(fn + 1) * N + n] = Out(value.y);
      }
    }
  }
}

// The tile is threadgroup-uniform: outside the projection's nonzero-hi range
// the hi plane is skipped (all zero, so the result is unchanged).
template <Epilogue E, uint L, class Out>
__attribute__((always_inline)) inline void decode(device const bfloat *table, device const uchar *w0,
                   device const uchar *hi0, device const half *sc0, device const half *bi0,
                   device Out *out, device const float *sums,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1, device const uchar *hi1,
                   device const half *sc1, device const half *bi1,
                   constant Q4PersistentParams &p, uint2 tg, uint tid, uint sg, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  const uint tile = (tg.x * (gateUp ? 32 : 64) + sg * (gateUp ? 8 : 16)) / 256;
  if (tile >= p.hi_tile_begin && tile < p.hi_tile_end)
    decodeTile<true, E, L, Out>(table, w0, hi0, sc0, bi0, out, sums, partials, counters, residual,
                                w1, hi1, sc1, bi1, p, tg, tid, sg, lane, arrival);
  else
    decodeTile<false, E, L, Out>(table, w0, hi0, sc0, bi0, out, sums, partials, counters, residual,
                                 w1, hi1, sc1, bi1, p, tg, tid, sg, lane, arrival);
}
} // namespace q5sgf

#define Q5_SGF_INPUTS(Out) \
    device const bfloat *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    device const uchar *hi [[buffer(2)]], \
    device const half *scales [[buffer(3)]], device const half *biases [[buffer(4)]], \
    device Out *output [[buffer(5)]], device const float *sums [[buffer(6)]], \
    device coherent(device) float *partials [[buffer(7)]], device atomic_uint *counters [[buffer(8)]]
#define Q5_SGF_THREADS \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

#define Q5_SGF_AFFINE(Name, L, Out) \
kernel void Name(Q5_SGF_INPUTS(Out), constant Q4PersistentParams &p [[buffer(9)]], Q5_SGF_THREADS) { \
  threadgroup uint arrival; \
  q5sgf::decode<q5sgf::Epilogue::Affine, L, Out>(table, weights, hi, scales, biases, output, sums, \
      partials, counters, table, weights, hi, scales, biases, p, tg, tid, sg, lane, &arrival); \
}
#define Q5_SGF_KERNELS(SUFFIX, L) \
Q5_SGF_AFFINE(decode_linear_q5_sgf##SUFFIX, L, bfloat) \
Q5_SGF_AFFINE(decode_linear_q5_sgf##SUFFIX##_f32, L, float) \
kernel void decode_linear_q5_sgf_residual##SUFFIX(Q5_SGF_INPUTS(bfloat), \
    device const bfloat *residual [[buffer(9)]], constant Q4PersistentParams &p [[buffer(10)]], \
    Q5_SGF_THREADS) { \
  threadgroup uint arrival; \
  q5sgf::decode<q5sgf::Epilogue::Residual, L, bfloat>(table, weights, hi, scales, biases, output, sums, \
      partials, counters, residual, weights, hi, scales, biases, p, tg, tid, sg, lane, &arrival); \
} \
kernel void decode_linear_q5_sgf_gate_up##SUFFIX(Q5_SGF_INPUTS(bfloat), \
    device const uchar *up [[buffer(9)]], device const uchar *upHi [[buffer(10)]], \
    device const half *upScales [[buffer(11)]], device const half *upBiases [[buffer(12)]], \
    constant Q4PersistentParams &p [[buffer(13)]], Q5_SGF_THREADS) { \
  threadgroup uint arrival; \
  q5sgf::decode<q5sgf::Epilogue::GateUp, L, bfloat>(table, weights, hi, scales, biases, output, sums, \
      partials, counters, output, up, upHi, upScales, upBiases, p, tg, tid, sg, lane, &arrival); \
}
Q5_SGF_KERNELS(, 1)
Q5_SGF_KERNELS(_m16, 2)
Q5_SGF_KERNELS(_m24, 3)
Q5_SGF_KERNELS(_m32, 4)
#undef Q5_SGF_KERNELS
#undef Q5_SGF_AFFINE
#undef Q5_SGF_INPUTS
#undef Q5_SGF_THREADS
