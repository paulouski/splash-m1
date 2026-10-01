#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/sgmatrix.h"

// Apple7/8 (M1/M2) register-matrix draft attention (the SimdgroupF32 tile),
// the non-MPP alternative to draft_attention_bf16_split (draft.metal, guarded
// to __METAL_VERSION__ >= 400) for the macOS 15 build (Makefile MACOS15=1,
// Metal 3.2, MetalPerformancePrimitives unavailable). Ports
// draft_attention_split_phase's ring/window bookkeeping unchanged (same
// common_start/physical_start/wrapped/resume/live_tiles, same per-row
// hidden_prefix mask, same 0.08838834765f scale, same non-causal current-row
// block); only the QK^T and PV matmuls move from mpp::tensor_ops to
// simdgroup_matrix, following vision_mma::attention (shared/vision_mma.metal):
// eight-row-pair fused rows via the sgmatrix Lane fragment convention, both
// operands widened from bf16 to exact fp32 (no bf16 arithmetic, as
// linear_q4_mma.metal's SimdgroupF32 tile also avoids on these GPUs), fp32
// accumulate. The result lands in the same fp32 partial layout
// draft_attention_bf16_reduce (draft.metal, unguarded) already consumes.
namespace draft_sgf {
using sgmatrix::Lane;
using sgmatrix::lane_map;
using sgmatrix::te;

// c += a x b on fp32 fragments (vision_mma::mma).
__attribute__((always_inline)) inline void mma(thread float2 &c, float2 a, float2 b) {
  simdgroup_float8x8 A, B, C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

constexpr constant uint kM = 32, kD = 128, kChunk = 16, kCurrentN = 8;

// One KV head's attention split for its four query heads' eight rows each:
// one simdgroup (of four) per query head, lane (fm, fn) owning fused rows
// sg*8+fn and sg*8+fn+1 at dimension 8t+fm, as vision_mma::attention's (fm,
// fn) rows sg*8+fn, sg*8+fn+1. Dispatch is {kv heads, lanes, splits} with 128
// threads (four simdgroups) per threadgroup, matching draft_attention_bf16_split.
inline void split_phase(
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    device bfloat *query_keys, device bfloat *query_values,
    device float *partial, uint cache_stride, uint cache_length, uint split,
    uint splits, threadgroup float *stage_k, threadgroup float *stage_v,
    uint thread_index, uint lane, uint simd_group) {
  constexpr uint N = 128;
  constexpr uint Window = SPLASH_DRAFT_SLIDING_WINDOW;
  uint common_start = cache_length >= Window - 1 ? cache_length - (Window - 1) : 0;
  uint old_count = cache_length - common_start;
  uint physical_start = common_start % Window;
  uint physical_end = physical_start + old_count;
  uint wrapped = physical_end > Window ? (physical_end - Window + N - 1) / N : 0;
  uint tail_end = (min(physical_end, Window) + N - 1) / N;
  uint resume = max(physical_start / N, wrapped);
  uint live_tiles = wrapped + tail_end - resume;
  // A split with no tile that is not the last one leaves only row maxima of
  // -inf, which the reduce skips, as draft_attention_split_phase.
  if (split >= live_tiles && split + 1 != splits) {
    if (thread_index < kM)
      partial[kM * kD + thread_index] = -INFINITY;
    return;
  }

  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint row = simd_group * 8 + fn;
  device const bfloat *q0 = queries + row * kD + fm;
  device const bfloat *q1 = q0 + kD;
  // Q^T fragments: dimension 8t+fm of rows (row, row + 1), widened exactly.
  float2 query[kD / 8];
#pragma unroll
  for (uint t = 0; t < kD / 8; ++t) query[t] = float2(q0[8 * t], q1[8 * t]);

  float2 out[kD / 8];
#pragma unroll
  for (uint d = 0; d < kD / 8; ++d) out[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);

  // This lane's fused rows' own visible-prefix start (row_start/hidden_prefix
  // of draft_attention_split_phase), constant for the whole split.
  const uint2 proposal_row(fn, fn + 1);
  const uint2 query_position = cache_length + proposal_row;
  const uint2 row_start(
      query_position.x >= Window - 1 ? query_position.x - (Window - 1) : 0,
      query_position.y >= Window - 1 ? query_position.y - (Window - 1) : 0);
  const uint2 hidden_prefix = row_start - common_start;

  for (uint tile = split; tile < live_tiles; tile += splits) {
    uint slot = (tile < wrapped ? tile : resume + tile - wrapped) * N;
    for (uint base = 0; base < N; base += kChunk) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint u = thread_index; u < kChunk * kD; u += 128) {
        uint token = u / kD, dim = u % kD;
        uint physical = slot + base + token;
        stage_k[token * kD + dim] = float(keys[physical * kD + dim]);
        stage_v[token * kD + dim] = float(values[dim * cache_stride + physical]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // S^T tile s: token base + 8s + fm x rows (row, row + 1).
      float2 score[kChunk / 8];
#pragma unroll
      for (uint s = 0; s < kChunk / 8; ++s) {
        score[s] = float2(0.0f);
        threadgroup const float *kr = stage_k + (8 * s + fm) * kD + fn;
#pragma unroll
        for (uint t = 0; t < kD / 8; ++t)
          mma(score[s], *reinterpret_cast<threadgroup const float2 *>(kr + 8 * t), query[t]);
      }

      float2 local_max(-INFINITY);
#pragma unroll
      for (uint s = 0; s < kChunk / 8; ++s) {
        uint token = slot + base + 8 * s + fm;
        uint logical_key = token >= physical_start ? token - physical_start
                                                   : token + Window - physical_start;
        bool validX = logical_key < old_count && logical_key >= hidden_prefix.x;
        bool validY = logical_key < old_count && logical_key >= hidden_prefix.y;
        float2 raw = score[s] * 0.08838834765f;
        score[s] = float2(validX ? raw.x : -INFINITY, validY ? raw.y : -INFINITY);
        local_max = max(local_max, score[s]);
      }
      // Lane bits 1, 2 and 4 select fm: the other keys of these rows.
      local_max = max(local_max, simd_shuffle_xor(local_max, 2));
      local_max = max(local_max, simd_shuffle_xor(local_max, 4));
      local_max = max(local_max, simd_shuffle_xor(local_max, 16));
      const float2 next_max = max(row_max, local_max);
      float2 local_sum(0.0f);
#pragma unroll
      for (uint s = 0; s < kChunk / 8; ++s) {
        score[s] = float2(fast::exp(score[s].x - next_max.x), fast::exp(score[s].y - next_max.y));
        local_sum += score[s];
      }
      local_sum += simd_shuffle_xor(local_sum, 2);
      local_sum += simd_shuffle_xor(local_sum, 4);
      local_sum += simd_shuffle_xor(local_sum, 16);
      // A row with no visible key so far (row_max still -inf) keeps its
      // all-zero accumulator: scale 0 has the same effect as the original's
      // scale-1-on-empty special case, since out[] is already zero.
      const float2 scale(row_max.x == -INFINITY ? 0.0f : fast::exp(row_max.x - next_max.x),
                         row_max.y == -INFINITY ? 0.0f : fast::exp(row_max.y - next_max.y));
      row_sum = row_sum * scale + local_sum;
      row_max = next_max;

      // O^T tile d = O^T tile d x scale + V^T (dimensions 8d+fm) x P^T.
#pragma unroll
      for (uint d = 0; d < kD / 8; ++d) {
        float2 partial_out(0.0f);
        threadgroup const float *vr = stage_v + fn * kD + 8 * d + fm;
#pragma unroll
        for (uint s = 0; s < kChunk / 8; ++s)
          mma(partial_out, float2(vr[8 * s * kD], vr[(8 * s + 1) * kD]), score[s]);
        out[d] = out[d] * scale + partial_out;
      }
    }
  }

  if (split + 1 == splits) {
    // The eight current rows are non-causal: every proposal row sees all
    // eight, as draft_attention_split_phase's last-split addition.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = thread_index; u < kCurrentN * kD; u += 128) {
      uint token = u / kD, dim = u % kD;
      stage_k[token * kD + dim] = float(query_keys[token * kD + dim]);
      stage_v[token * kD + dim] = float(query_values[dim * kCurrentN + token]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 score(0.0f);
    threadgroup const float *kr = stage_k + fm * kD + fn;
#pragma unroll
    for (uint t = 0; t < kD / 8; ++t)
      mma(score, *reinterpret_cast<threadgroup const float2 *>(kr + 8 * t), query[t]);
    score = score * 0.08838834765f;
    float2 local_max = max(score, simd_shuffle_xor(score, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 probability(fast::exp(score.x - next_max.x), fast::exp(score.y - next_max.y));
    float2 local_sum = probability;
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(row_max.x == -INFINITY ? 0.0f : fast::exp(row_max.x - next_max.x),
                       row_max.y == -INFINITY ? 0.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;

#pragma unroll
    for (uint d = 0; d < kD / 8; ++d) {
      threadgroup const float *vr = stage_v + fn * kD + 8 * d + fm;
      float2 partial_out(0.0f);
      mma(partial_out, float2(vr[0], vr[kD]), probability);
      out[d] = out[d] * scale + partial_out;
    }
  }

  device float *out0 = partial + row * kD;
  device float *out1 = partial + (row + 1) * kD;
#pragma unroll
  for (uint d = 0; d < kD / 8; ++d) {
    out0[8 * d + fm] = out[d].x;
    out1[8 * d + fm] = out[d].y;
  }
  if (fm == 0) {
    partial[kM * kD + row] = row_max.x;
    partial[kM * kD + row + 1] = row_max.y;
    partial[kM * kD + kM + row] = row_sum.x;
    partial[kM * kD + kM + row + 1] = row_sum.y;
  }
}
} // namespace draft_sgf

// Register-tile (Apple7/8) alternative to draft_attention_bf16_split
// (draft.metal): same bindings, params and {kv heads, lanes, splits} grid,
// but 128 threads (four simdgroups, one per query head) per threadgroup
// instead of 256. Selected by DraftAttention.cpp when appleGpuFamily < 9; its
// output feeds the same draft_attention_bf16_reduce.
kernel void draft_attention_bf16_split_sgf(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys0 [[buffer(1)]], device bfloat *keys1 [[buffer(2)]],
    device bfloat *keys2 [[buffer(3)]], device bfloat *keys3 [[buffer(4)]],
    device bfloat *values0 [[buffer(5)]],
    device bfloat *values1 [[buffer(6)]],
    device bfloat *values2 [[buffer(7)]],
    device bfloat *values3 [[buffer(8)]],
    device bfloat *query_keys [[buffer(9)]],
    device bfloat *query_values [[buffer(10)]],
    constant DraftAttentionBatchParams &params [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint AttentionM = 32, KVHeads = 8;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Attention = 4096;
  constexpr ulong HeadDim = 128;
  constexpr ulong PartialFloats = AttentionM * HeadDim + 2 * AttentionM;
  uint batch = group.y;
  if (batch >= params.lanes)
    return;
  device bfloat *keys =
      batch == 0 ? keys0 : (batch == 1 ? keys1 : (batch == 2 ? keys2 : keys3));
  device bfloat *values = batch == 0
                              ? values0
                              : (batch == 1 ? values1
                                            : (batch == 2 ? values2 : values3));
  device float *partials =
      reinterpret_cast<device float *>(queries +
                                       params.lanes * Rows * Attention) +
      ((batch * KVHeads + group.x) * params.splits + group.z) * PartialFloats;
  threadgroup float stage_k[draft_sgf::kChunk * draft_sgf::kD];
  threadgroup float stage_v[draft_sgf::kChunk * draft_sgf::kD];
  draft_sgf::split_phase(
      queries + batch * Rows * Attention + group.x * AttentionM * HeadDim,
      keys + group.x * params.cache_stride * HeadDim,
      values + group.x * params.cache_stride * HeadDim,
      query_keys + batch * KVHeads * Rows * HeadDim + group.x * Rows * HeadDim,
      query_values + batch * KVHeads * HeadDim * Rows +
          group.x * Rows * HeadDim,
      partials, params.cache_stride, params.cache_length[batch], group.z,
      params.splits, stage_k, stage_v, thread_index, lane, simd_group);
}
