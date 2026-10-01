#include "metal/kernels/common/paged_attention_tile.h"

// Split into its own file so this Q8 verify reduce survives the
// MACOS15_EXCLUDED_KERNELS drop of attention_q8.metal: unlike that file's
// split kernels, it calls no MPP function
// (splash_q8_attention_reduce_row_shared, paged_attention_tile.h, is
// unconditionally defined at every Metal version), and
// PagedAttention::verifyPlan binds this pipeline for any Q8-format verify
// regardless of the split tile (register or MPP), so it is reached on
// Apple7/8 too.
template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_q8_verify_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashQ8VerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr ushort M = SPLASH_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashQ8HeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint batch = group.z;
  constant SplashQ8VerifyAttentionParams &lane_params = params[batch];
  if (!splash_q8_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  ulong group_stride = ulong(lane_params.chunk_stride) * QueryHeadsPerKVHead * D;
  splash_q8_attention_reduce_row_shared<QueryHeadsPerKVHead,
                                   SPLASH_TARGET_VERIFY_ROWS>(
      partials, statistics,
      output + (ulong(batch) * KVHeads + kv_head) * group_stride,
      lane_params.committed_tokens, lane_params.active_rows,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
}

kernel void verify_attention_q8_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8VerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[SplashVerifyMaximumSplits];
  threadgroup float group_values[8];
  splash_q8_verify_attention_reduce_phase<4, 6>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}

kernel void verify_attention_q8_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8VerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[SplashVerifyMaximumSplits];
  threadgroup float group_values[8];
  splash_q8_verify_attention_reduce_phase<2, 8>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}
