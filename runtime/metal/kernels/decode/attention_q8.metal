#include "metal/kernels/common/paged_verify_tile.h"

// Verify tiles process one lane's eight rows per KV head and history split.
// Verify and prefill share the device-operand page loop in paged_attention_tile.h.

// Verify entries: one lane per group.z, eight rows, one configured history
// partition, with scratch for scores, probabilities and row statistics.
#define Q8_VERIFY_SPLIT_SIGNATURE(Name)                                        \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device int8_t *cache_keys [[buffer(1)]],                                 \
      device const float *key_scales_buffer [[buffer(2)]],                     \
      device int8_t *cache_values [[buffer(3)]],                               \
      device const float *value_scales_buffer [[buffer(4)]],                   \
      device float *partials [[buffer(5)]],                                    \
      device float *statistics [[buffer(6)]],                                  \
      device const uint *page_table0 [[buffer(7)]],                            \
      device const uint *page_table1 [[buffer(8)]],                            \
      device const uint *page_table2 [[buffer(9)]],                            \
      device const uint *page_table3 [[buffer(10)]],                           \
      constant SplashQ8VerifyAttentionParams *params [[buffer(11)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]])

#define Q8_VERIFY_SCRATCH(Group)                                               \
  constexpr uint M = Group * SPLASH_TARGET_VERIFY_ROWS;                        \
  constexpr uint N = SplashQ8PageTokens;                                       \
  alignas(16) threadgroup float scores[M * N];                                 \
  alignas(16) threadgroup bfloat probabilities[M * N];                         \
  threadgroup float row_max[M];                                                \
  threadgroup float row_sum[M];                                                \
  threadgroup float previous_scale[M];                                         \
  threadgroup atomic_uint rescale;

#define Q8_VERIFY_TILE_AT(Heads, Group)                                        \
  const SplashQ8VerifyTile tile =                                              \
      splash_q8_verify_attention_tile_at<Heads, Group>(                        \
          queries, page_table0, page_table1, page_table2, page_table3, params, \
          group);                                                              \
  if (!tile.active)                                                            \
    return;

#define Q8_VERIFY_SPLIT(Name, Heads, Group, ScaleInSoftmax)                    \
  Q8_VERIFY_SPLIT_SIGNATURE(Name) {                                            \
    Q8_VERIFY_SCRATCH(Group)                                                   \
    Q8_VERIFY_TILE_AT(Heads, Group)                                            \
    splash_paged_attention_tile<Heads, Group,                                  \
                                      SPLASH_TARGET_VERIFY_ROWS,               \
                                      ScaleInSoftmax>(                         \
        tile.queries, cache_keys, key_scales_buffer, cache_values, value_scales_buffer,\
        tile.page_table, tile.kv_head, tile.committed_tokens, tile.active_rows,\
        tile.splits, tile.split, partials, statistics, tile.slot, scores,      \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }

Q8_VERIFY_SPLIT(verify_attention_q8_split, 4, 6, true)
Q8_VERIFY_SPLIT(verify_attention_q8_split_cooperative_scale,
                       4, 6, false)
Q8_VERIFY_SPLIT(verify_attention_q8_split_kv2_g8, 2, 8, true)
Q8_VERIFY_SPLIT(
    verify_attention_q8_split_cooperative_scale_kv2_g8, 2, 8, false)
#define BF16_VERIFY_SPLIT_SIGNATURE(Name)                                      \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *cache_keys [[buffer(1)]],                                 \
      device bfloat *cache_values [[buffer(2)]],                               \
      device float *partials [[buffer(3)]],                                    \
      device float *statistics [[buffer(4)]],                                  \
      device const uint *page_table0 [[buffer(5)]],                            \
      device const uint *page_table1 [[buffer(6)]],                            \
      device const uint *page_table2 [[buffer(7)]],                            \
      device const uint *page_table3 [[buffer(8)]],                            \
      constant SplashQ8VerifyAttentionParams *params [[buffer(9)]],            \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]])

#define BF16_VERIFY_SPLIT(Name, Heads, Group)                                  \
  BF16_VERIFY_SPLIT_SIGNATURE(Name) {                                          \
    Q8_VERIFY_SCRATCH(Group)                                                   \
    Q8_VERIFY_TILE_AT(Heads, Group)                                            \
    splash_paged_attention_tile<Heads, Group,                                  \
                                      SPLASH_TARGET_VERIFY_ROWS,               \
                                      true>(                                   \
        tile.queries, cache_keys, nullptr, cache_values, nullptr,              \
        tile.page_table, tile.kv_head, tile.committed_tokens, tile.active_rows,\
        tile.splits, tile.split, partials, statistics, tile.slot, scores,      \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }

BF16_VERIFY_SPLIT(verify_attention_bf16_split, 4, 6)
BF16_VERIFY_SPLIT(verify_attention_bf16_split_kv2_g8, 2, 8)
#undef BF16_VERIFY_SPLIT
#undef BF16_VERIFY_SPLIT_SIGNATURE
#undef Q8_VERIFY_SPLIT
#undef Q8_VERIFY_TILE_AT
#undef Q8_VERIFY_SCRATCH
#undef Q8_VERIFY_SPLIT_SIGNATURE

// splash_q8_verify_attention_reduce_phase / verify_attention_q8_reduce[_kv2_g8]
// moved to attention_q8_reduce.metal: they call no MPP function (unlike the
// split kernels above), and PagedAttention::verifyPlan binds the reduce
// pipeline unconditionally for any Q8-format verify, register tile
// included -- so it must survive the MACOS15_EXCLUDED_KERNELS drop of this
// file.
