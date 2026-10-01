#include "metal/kernels/common/paged_attention_tile.h"

// Split into its own file so this Q8 prefill reduce survives the
// MACOS15_EXCLUDED_KERNELS drop of attention_q8.metal: unlike that file's
// split kernels, it calls no MPP function (splash_q8_attention_reduce_row,
// paged_attention_tile.h, is unconditionally defined at every Metal
// version), and PagedAttention::prefillPlan binds this pipeline for any
// Q8-format prefill regardless of the split tile (register or MPP), so it
// is reached on Apple7/8 too.
template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_q8_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashQ8PrefillAttentionParams &params, uint3 group,
    uint thread_index) {
  constexpr ushort M = SPLASH_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashQ8HeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * SplashPrefillTileRows;
  if (!splash_q8_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return;
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  splash_q8_attention_reduce_row<QueryHeadsPerKVHead,
                                   SPLASH_PREFILL_ATTENTION_TILE_ROWS>(
      partials, statistics, output + tile_offset,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index);
}

kernel void prefill_attention_q8_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8PrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  splash_q8_prefill_attention_reduce_phase<4, 6>(partials, statistics,
                                                   output, params, group,
                                                   thread_index);
}

kernel void prefill_attention_q8_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8PrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  splash_q8_prefill_attention_reduce_phase<2, 8>(partials, statistics,
                                                   output, params, group,
                                                   thread_index);
}
