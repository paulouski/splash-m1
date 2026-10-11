#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_verify_tile.h"
#include "metal/abi/Qwen4.h"

// Same bindings as verify_attention_q8_split and verify_attention_bf16_split;
// the grid is (KV heads, splits, lanes) with 32 x G threads, one simdgroup per
// eight fused rows.
#define SGF_VERIFY_SPLIT(Name, Heads, Group, CacheElement)                     \
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
    q8sgf::tile<Heads, Group, CacheElement>(                                   \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, SPLASH_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, sg, lane, query_words);   \
  }

SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf, 4, 6, int8_t)
SGF_VERIFY_SPLIT(verify_attention_q8_split_sgf_kv2_g8, 2, 8, int8_t)
SGF_VERIFY_SPLIT(verify_attention_bf16_split_sgf, 4, 6, bfloat)
SGF_VERIFY_SPLIT(verify_attention_bf16_split_sgf_kv2_g8, 2, 8, bfloat)
#undef SGF_VERIFY_SPLIT

// Qwen3.8-Flash-Next's group of twelve, split by query rows over two
// threadgroups of six simdgroups (group.x is 2 kv_head + half), and reading the
// QSA bitmaps: a lane's eight rows from bitmap row mask.row0 + 8 lane
// (mask_words == 0: every row attends causally). Buffers 8 to 10 are always bound;
// the page lists (list_stride != 0) hold each lane's compacted kept pages.
#define SGF_VERIFY_SPLIT_G12(Name, CacheElement)                               \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]],                    \
      constant SplashVerifyAttentionParams *params [[buffer(7)]],              \
      device const uint *block_mask [[buffer(8)]],                             \
      device const uint *page_lists [[buffer(9)]],                             \
      constant Qwen4QsaMaskParams &mask [[buffer(10)]],                        \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup uint query_words[6 * 1024];                                    \
    const SplashVerifyTile tile =                                              \
        splash_verify_attention_tile_at<2, 12>(                                \
            queries, page_table0, page_table1, page_table2, page_table3,       \
            params, uint3(group.x >> 1, group.y, group.z));                    \
    if (!tile.active || sg >= 6)                                               \
      return;                                                                  \
    q8sgf::tile<2, 12, CacheElement, 2, true>(                                 \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, SPLASH_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, sg, lane, query_words,    \
        group.x & 1,                                                           \
        block_mask + ulong(mask.row0 + group.z * SPLASH_TARGET_VERIFY_ROWS) *  \
                         mask.mask_words,                                      \
        mask.mask_words,                                                       \
        mask.list_stride && mask.mask_words                                    \
            ? page_lists + ulong(group.z) * mask.list_stride + mask.list_offset \
            : nullptr,                                                         \
        mask.min_pages);                                                       \
  }

SGF_VERIFY_SPLIT_G12(verify_attention_q8_split_sgf_kv2_g12, int8_t)
SGF_VERIFY_SPLIT_G12(verify_attention_bf16_split_sgf_kv2_g12, bfloat)
#undef SGF_VERIFY_SPLIT_G12
