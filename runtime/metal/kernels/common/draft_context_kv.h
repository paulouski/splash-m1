#pragma once

#include "metal/abi/KernelABI.h"

template <uint PackedWidth, uint KeyOffset>
inline void draft_context_kv_phase(
    device const bfloat *context_qkv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys, device bfloat *values,
    DraftContextParams params, uint active_tokens, uint task,
    uint thread_index, uint lane, uint simd_group,
    threadgroup float *reductions, threadgroup bfloat *normalized) {
  constexpr uint KVHeads = 8, HeadDim = 128, Window = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr uint KWidth = KVHeads * HeadDim;
  static_assert(PackedWidth == KeyOffset + 2 * KWidth);
  uint row = task / KVHeads;
  if (row >= active_tokens)
    return;
  uint head_index = task % KVHeads;
  uint position = params.start_position + row;
  uint slot = position % Window;
  device const bfloat *source =
      context_qkv + ulong(row) * PackedWidth + KeyOffset + head_index * HeadDim;
  device bfloat *key =
      keys + (ulong(head_index) * params.cache_stride + slot) * HeadDim;
  device bfloat *value =
      values + ulong(head_index) * HeadDim * params.cache_stride + slot;

  float element = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
  float square_sum = simd_sum(element * element);
  if (lane == 0)
    reductions[simd_group] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / HeadDim + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim) {
    normalized[thread_index] =
        bfloat(element * reductions[0] * float(k_norm[thread_index]));
    value[ulong(thread_index) * params.cache_stride] =
        source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim / 2) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + HeadDim / 2]);
    float cosine = rope_cos[ulong(row) * (HeadDim / 2) + thread_index];
    float sine = rope_sin[ulong(row) * (HeadDim / 2) + thread_index];
    key[thread_index] = bfloat(first * cosine - second * sine);
    key[thread_index + HeadDim / 2] = bfloat(second * cosine + first * sine);
  }
}

template <uint PackedWidth, uint KeyOffset>
inline void draft_context_kv_commit_phase(
    device const bfloat *context_qkv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys0, device bfloat *keys1, device bfloat *keys2,
    device bfloat *keys3, device bfloat *values0, device bfloat *values1,
    device bfloat *values2, device bfloat *values3,
    device const uint *retained, constant DraftContextBatchParams &params,
    uint group, uint thread_index, uint lane, uint simd_group,
    threadgroup float *reductions, threadgroup bfloat *normalized) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint KVHeads = 8;
  constexpr uint RopeLaneStride = Rows * 64;
  uint batch = group / (Rows * KVHeads);
  uint task = group % (Rows * KVHeads);
  if (batch >= params.lanes)
    return;
  device bfloat *keys =
      batch == 0 ? keys0 : (batch == 1 ? keys1 : (batch == 2 ? keys2 : keys3));
  device bfloat *values = batch == 0
                              ? values0
                              : (batch == 1 ? values1
                                            : (batch == 2 ? values2 : values3));
  DraftContextParams lane_params{Rows, params.cache_stride,
                                  params.start_position[batch]};
  draft_context_kv_phase<PackedWidth, KeyOffset>(
      context_qkv + ulong(batch) * Rows * PackedWidth, k_norm,
      rope_cos + ulong(batch) * RopeLaneStride,
      rope_sin + ulong(batch) * RopeLaneStride, keys, values, lane_params,
      min(retained[batch], Rows), task, thread_index, lane, simd_group,
      reductions, normalized);
}
