#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_verify_tile.h"

// Same bindings as verify_attention_q8_split; the grid is (KV heads, splits,
// lanes) with 32 x G threads, one simdgroup per eight fused rows.
#define Q8_SGF_VERIFY_SPLIT(Name, Heads, Group)                                \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]],                    \
      constant SplashVerifyAttentionParams *params [[buffer(7)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup uint query_words[Group * 1024];                                \
    const SplashVerifyTile tile =                                              \
        splash_verify_attention_tile_at<Heads, Group>(                         \
            queries, page_table0, page_table1, page_table2, page_table3,       \
            params, group);                                                    \
    if (!tile.active || sg >= Group)                                           \
      return;                                                                  \
    q8sgf::tile<Heads, Group>(                                                 \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, SPLASH_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, sg, lane, query_words);   \
  }

Q8_SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf, 4, 6)
Q8_SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf_kv2_g8, 2, 8)
#undef Q8_SGF_VERIFY_SPLIT
