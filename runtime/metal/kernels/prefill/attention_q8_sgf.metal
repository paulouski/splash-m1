#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/abi/Qwen4.h"

// Same bindings and grid as prefill_attention_q8_split and
// prefill_attention_bf16_split (KV heads, query tiles, history splits) with
// 32 x G threads, one simdgroup per eight fused rows. Slots, partials and
// statistics are the MPP tile's, so prefill_attention_reduce combines them
// unchanged.
#define SGF_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                    \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup uint query_words[Group * 1024];                                \
    constexpr uint D = SplashKvHeadDimension;                                  \
    const uint kv_head = group.x, tile = group.y, split = group.z;             \
    const uint tile_start = tile * SplashPrefillTileRows;                      \
    if (!splash_prefill_attention_contract_valid(params) ||                    \
        kv_head >= Heads || split >= params.split_count ||                     \
        tile_start >= params.rows || sg >= Group)                              \
      return;                                                                  \
    const uint active_rows =                                                   \
        min(SplashPrefillTileRows, params.rows - tile_start);                  \
    const ulong tile_offset =                                                  \
        (ulong(kv_head) * params.chunk_stride + tile_start) * Group * D;       \
    const ulong slot =                                                         \
        (ulong(tile) * Heads + kv_head) * params.split_count + split;          \
    q8sgf::tile<Heads, Group, CacheElement>(                                   \
        queries + tile_offset, page_table, params.kv, kv_head,                 \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, sg, lane, query_words);             \
  }

SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf, 4, 6, int8_t)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv2_g8, 2, 8, int8_t)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf, 4, 6, bfloat)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv2_g8, 2, 8, bfloat)
#undef SGF_PREFILL_SPLIT

// Qwen3.8-Flash-Next's group of twelve: two threadgroups of six simdgroups per
// KV head (group.x is 2 kv_head + half), reading the QSA bitmaps from bitmap
// row mask.row0 + tile_start (mask_words == 0: every row attends causally).
// Buffers 5 and 6 are always bound.
#define SGF_PREFILL_SPLIT_G12(Name, CacheElement)                              \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      device const uint *block_mask [[buffer(5)]],                             \
      constant Qwen4QsaMaskParams &mask [[buffer(6)]],                         \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup uint query_words[6 * 1024];                                    \
    constexpr uint D = SplashKvHeadDimension;                                  \
    const uint kv_head = group.x >> 1, tile = group.y, split = group.z;        \
    const uint tile_start = tile * SplashPrefillTileRows;                      \
    if (!splash_prefill_attention_contract_valid(params) ||                    \
        kv_head >= 2 || split >= params.split_count ||                         \
        tile_start >= params.rows || sg >= 6)                                  \
      return;                                                                  \
    const uint active_rows =                                                   \
        min(SplashPrefillTileRows, params.rows - tile_start);                  \
    const ulong tile_offset =                                                  \
        (ulong(kv_head) * params.chunk_stride + tile_start) * 12 * D;          \
    const ulong slot =                                                         \
        (ulong(tile) * 2 + kv_head) * params.split_count + split;              \
    q8sgf::tile<2, 12, CacheElement, 2, true>(                                 \
        queries + tile_offset, page_table, params.kv, kv_head,                 \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, sg, lane, query_words, group.x & 1, \
        block_mask + ulong(mask.row0 + tile_start) * mask.mask_words,          \
        mask.mask_words);                                                      \
  }

SGF_PREFILL_SPLIT_G12(prefill_attention_q8_split_sgf_kv2_g12, int8_t)
SGF_PREFILL_SPLIT_G12(prefill_attention_bf16_split_sgf_kv2_g12, bfloat)
#undef SGF_PREFILL_SPLIT_G12
