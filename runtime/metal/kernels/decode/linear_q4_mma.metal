// Keep the source order of float operations (as decode_linear_q4_sg does).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/q4_sgmatrix.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

// Apple7/8 (M1/M2) register-matrix Q4 decode projection (the SimdgroupF32
// tile). These GPUs have no bfloat arithmetic and issue simdgroup matrix
// products on the ordinary ALU pipes, so the kernel is bound by instructions
// per MMA rather than by weight bandwidth:
//   A  the weights as exact half: nibble | 0x6400 is 1024 + q (ulp 1), and
//      subtracting 1024 leaves q;
//   B  shared X^T table values widened to fp32; selected decode shapes stage
//      their BF16-rounded values through FP16, which can round, overflow, or underflow;
//   C  fp32 accumulation, preserving products for the values actually read.
// One threadgroup covers every request lane of a batch: each unpacked weight
// fragment feeds one MMA per lane, instead of a threadgroup per lane unpacking
// the same weights again. A simdgroup computes 16 columns (8 for each of the
// gate/up streams) for all L eight-row tiles. The table, row sums, K split and
// its partial-sum workspace follow decode_linear_q4_sg; lanes' tiles are
// consecutive in every buffer. The destination type Out is bf16, or fp32 for
// a plain projection's logits (ops::Projection::destination).
namespace q4sgf {
using namespace q4sg;
using sgmatrix::te;
enum class Epilogue { Affine, Residual, GateUp };

template <class Input> struct InputFragment;
template <> struct InputFragment<bfloat> { vec<bfloat, 8> values; };
template <> struct InputFragment<half> { float4 values[2]; };

template <class Input>
__attribute__((always_inline)) inline void decodeInput(
    device const Input *table, uint base, uint fragIndex, thread InputFragment<Input> &out) {
  if constexpr (is_same_v<Input, bfloat>) {
    out.values = reinterpret_cast<device const vec<bfloat, 8> *>(table + base)[fragIndex];
  } else {
    device const half4 *x = reinterpret_cast<device const half4 *>(table + base) + fragIndex * 2;
    out.values[0] = float4(x[0]);
    out.values[1] = float4(x[1]);
  }
}

template <class Input>
__attribute__((always_inline)) inline float2 inputPair(thread InputFragment<Input> &input, uint s) {
  if constexpr (is_same_v<Input, bfloat>)
    return float2(reinterpret_cast<thread bfloat2 *>(&input.values)[s]);
  else
    return reinterpret_cast<thread float2 *>(&input.values[0])[s];
}

// c += a x b for exact half weights and fp32 activations.
__attribute__((always_inline)) inline void mma(thread float2 &c, half2 a, float2 b) {
  simdgroup_half8x8 A;
  simdgroup_float8x8 B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

template <Epilogue E, uint L, class Out, class Input = bfloat>
__attribute__((always_inline)) inline void decode(device const Input *table, device const uchar *w0,
                   device const half *sc0, device const half *bi0,
                   device Out *out, device const float *sums,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *residual, device const uchar *w1,
                   device const half *sc1, device const half *bi1,
                   constant Q4Params &p, uint2 tg, uint tid, uint sg, uint lane,
                   threadgroup uint *arrival) {
  constexpr bool gateUp = E == Epilogue::GateUp;
  constexpr uint tileN = gateUp ? 32 : 64;
  const uint N = p.output_size, K = p.input_size, groups = K / 64;
  const uint splits = p.persistent_groups;
  const uint first = tg.y * (groups / splits);
  const uint end = tg.y + 1 == splits ? groups : first + groups / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * tileN + sg * (gateUp ? 8 : 16);
  const uint tile = base / 256;
  // Fragment f: the gate/up streams share one column; plain fragments are
  // eight columns apart in one stream.
  auto stream = [&](uint f) { return gateUp ? f : 0u; };
  auto column = [&](uint f) { return base + fm + (gateUp ? 0u : f * 8); };
  // Per-lane weight pointers (tile and column folded in once); the group offset stays 32-bit.
  device const uchar *lanes[2] = {
      w0 + ulong(tile) * groups * 8192 + (column(0) % 256) * 32 + c * 8,
      (gateUp ? w1 : w0) + ulong(tile) * groups * 8192 + (column(1) % 256) * 32 + c * 8};
  // 32-bit element offsets relative to the projection's own buffers.
  const uint prmTile = tile * groups * 256;
  uint2 words[2][2];
  auto load = [&](uint j, uint g) __attribute__((always_inline)) {
#pragma unroll
    for (uint f = 0; f < 2; ++f)
      words[j][f] = *reinterpret_cast<device const uint2 *>(lanes[f] + g * 8192u);
  };
  float2 acc[2][L];
#pragma unroll
  for (uint f = 0; f < 2; ++f)
#pragma unroll
    for (uint r = 0; r < L; ++r) acc[f][r] = float2(0);
  // GateUp: two groups per iteration (independent MMA chains, folded in group order).
  uint2 w[2][2];
  auto run = [&](auto count, uint g) __attribute__((always_inline)) {
    constexpr uint J = decltype(count)::value;
    // Zeroed as a whole, as in decode_linear_q4_sg (GPU validation).
    float2 dot[J][2][L];
#pragma unroll
    for (uint j = 0; j < J; ++j)
#pragma unroll
      for (uint f = 0; f < 2; ++f)
#pragma unroll
        for (uint r = 0; r < L; ++r) dot[j][f][r] = float2(0);
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
      // k-steps 4h .. 4h + 3 of every lane's table.
      InputFragment<Input> x[J][L];
#pragma unroll
      for (uint j = 0; j < J; ++j)
#pragma unroll
        for (uint r = 0; r < L; ++r)
          decodeInput(table, r * K * kRows + (g + j) * kXtPerGroup,
                      (h * 8 + fm) * 4 + c, x[j][r]);
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
#pragma unroll
        for (uint j = 0; j < J; ++j) {
          float2 b[L];
#pragma unroll
          for (uint r = 0; r < L; ++r) b[r] = inputPair(x[j][r], s);
#pragma unroll
          for (uint f = 0; f < 2; ++f) {
            const uint nibbles = ((h ? w[j][f].y : w[j][f].x) >> (4 * s)) & 0x000F000Fu;
            const half2 a = as_type<half2>(nibbles | 0x64006400u);
#pragma unroll
            for (uint r = 0; r < L; ++r) mma(dot[j][f][r], a, b[r]);
          }
        }
      }
    }
#pragma unroll
    for (uint j = 0; j < J; ++j)
#pragma unroll
      for (uint f = 0; f < 2; ++f) {
        const uint prm = prmTile + (g + j) * 256 + column(f) % 256;
        const float scale = float((stream(f) ? sc1 : sc0)[prm]);
        // weights are 1024 + q: fold -1024 * scale into the bias (rowSums * bias).
        const float bias = fma(-1024.0f, scale, float((stream(f) ? bi1 : bi0)[prm]));
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          device const float *rowSums = sums + r * (K / 8) + (g + j) * kRows + fn;
          acc[f][r] = fma(dot[j][f][r], scale, acc[f][r]);
          acc[f][r] = fma(float2(rowSums[0], rowSums[1]), bias, acc[f][r]);
        }
      }
  };
  uint g = first;
  if constexpr (gateUp) {
    if (g + 1 < end) {
      load(0, g);
      load(1, g + 1);
    } else if (g < end) {
      load(0, g);
    }
    for (; g + 1 < end; g += 2) {
      w[0][0] = words[0][0], w[0][1] = words[0][1], w[1][0] = words[1][0], w[1][1] = words[1][1];
      if (g + 2 < end) load(0, g + 2);
      if (g + 3 < end) load(1, g + 3);
      run(integral_constant<uint, 2>{}, g);
    }
  } else {
    load(0, g);
    for (; g < end; ++g) {
      w[0][0] = words[0][0], w[0][1] = words[0][1];
      if (g + 1 < end) load(0, g + 1);
      run(integral_constant<uint, 1>{}, g);
    }
  }
  if (gateUp && g < end) {
    w[0][0] = words[0][0], w[0][1] = words[0][1];
    run(integral_constant<uint, 1>{}, g);
  }
  if (splits > 1) {
    // Each lane's tile owns splits x two streams of eight fp32 rows.
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
    // Same protocol as decode_linear_q4_sg (split_reduce.h).
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
} // namespace q4sgf

#define Q4_SGF_TYPED_INPUTS(Out, Table) \
    device const Table *table [[buffer(0)]], device const uchar *weights [[buffer(1)]], \
    device const half *scales [[buffer(2)]], device const half *biases [[buffer(3)]], \
    device Out *output [[buffer(4)]], device const float *sums [[buffer(5)]], \
    device coherent(device) float *partials [[buffer(6)]], device atomic_uint *counters [[buffer(7)]]
#define Q4_SGF_INPUTS(Out) Q4_SGF_TYPED_INPUTS(Out, bfloat)
#define Q4_SGF_THREADS \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

// One instance per batch width: M16/M24/M32 cover two to four lanes. The
// plain projection also writes fp32 (_f32: the logits); the input table
// stands in for the residual it does not read.
#define Q4_SGF_AFFINE(Name, L, Out) \
kernel void Name(Q4_SGF_INPUTS(Out), constant Q4Params &p [[buffer(8)]], Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::Affine, L, Out>(table, weights, scales, biases, output, sums, \
      partials, counters, table, weights, scales, biases, p, tg, tid, sg, lane, &arrival); \
}
#define Q4_SGF_KERNELS(SUFFIX, L) \
Q4_SGF_AFFINE(decode_linear_q4_sgf##SUFFIX, L, bfloat) \
Q4_SGF_AFFINE(decode_linear_q4_sgf##SUFFIX##_f32, L, float) \
[[max_total_threads_per_threadgroup(128)]] kernel void decode_linear_q4_sgf_residual##SUFFIX(Q4_SGF_INPUTS(bfloat), \
    device const bfloat *residual [[buffer(8)]], constant Q4Params &p [[buffer(9)]], \
    Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::Residual, L, bfloat>(table, weights, scales, biases, output, sums, \
      partials, counters, residual, weights, scales, biases, p, tg, tid, sg, lane, &arrival); \
} \
kernel void decode_linear_q4_sgf_gate_up##SUFFIX(Q4_SGF_INPUTS(bfloat), \
    device const uchar *up [[buffer(8)]], device const half *upScales [[buffer(9)]], \
    device const half *upBiases [[buffer(10)]], constant Q4Params &p [[buffer(11)]], \
    Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::GateUp, L, bfloat>(table, weights, scales, biases, output, sums, \
      partials, counters, output, up, upScales, upBiases, p, tg, tid, sg, lane, &arrival); \
}
Q4_SGF_KERNELS(, 1)
Q4_SGF_KERNELS(_m16, 2)
Q4_SGF_KERNELS(_m24, 3)
Q4_SGF_KERNELS(_m32, 4)
#define Q4_SGF_TYPED_AFFINE(Name, Out, Table) \
kernel void Name(Q4_SGF_TYPED_INPUTS(Out, Table), constant Q4Params &p [[buffer(8)]], Q4_SGF_THREADS) { \
  threadgroup uint arrival; \
  q4sgf::decode<q4sgf::Epilogue::Affine, 1, Out, Table>(table, weights, scales, biases, output, sums, \
      partials, counters, reinterpret_cast<device const bfloat *>(table), weights, scales, biases, p, tg, tid, sg, lane, &arrival); \
}
Q4_SGF_TYPED_AFFINE(decode_linear_q4_sgf_halftable, bfloat, half)
Q4_SGF_TYPED_AFFINE(decode_linear_q4_sgf_halftable_f32, float, half)
#undef Q4_SGF_TYPED_AFFINE
kernel void decode_linear_q4_sgf_halftable_gate_up(
    Q4_SGF_TYPED_INPUTS(bfloat, half), device const uchar *up [[buffer(8)]],
    device const half *upScales [[buffer(9)]], device const half *upBiases [[buffer(10)]],
    constant Q4Params &p [[buffer(11)]], Q4_SGF_THREADS) {
  threadgroup uint arrival;
  q4sgf::decode<q4sgf::Epilogue::GateUp, 1, bfloat, half>(table, weights, scales, biases, output, sums,
      partials, counters, output, up, upScales, upBiases, p, tg, tid, sg, lane, &arrival);
}
#undef Q4_SGF_KERNELS
#undef Q4_SGF_AFFINE
#undef Q4_SGF_INPUTS
#undef Q4_SGF_TYPED_INPUTS
#undef Q4_SGF_THREADS
