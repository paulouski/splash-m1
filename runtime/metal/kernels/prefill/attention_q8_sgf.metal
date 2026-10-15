#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_attention_vp.h"
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
    q8sgf::tile<2, 12, CacheElement, 2, true, true>(                           \
        queries + tile_offset, page_table, params.kv, kv_head,                 \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, sg, lane, query_words, group.x & 1, \
        block_mask + ulong(mask.row0 + tile_start) * mask.mask_words,          \
        mask.mask_words);                                                      \
  }

SGF_PREFILL_SPLIT_G12(prefill_attention_q8_split_sgf_kv2_g12, int8_t)
SGF_PREFILL_SPLIT_G12(prefill_attention_bf16_split_sgf_kv2_g12, bfloat)
#undef SGF_PREFILL_SPLIT_G12

// N1: the masked int8 G12 prefill over virtual pages. Each simdgroup walks the list of its eight fused rows'
// admitted blocks (prefill_attention_vp_listbuild) instead of every page of the context; same grid, partials and
// reduce as prefill_attention_q8_split_sgf_kv2_g12. `lists` holds (offset, virtual pages) pairs per
// (tile, half, simdgroup), then from word mask.list_offset mask.list_stride block ids per segment.
kernel void prefill_attention_q8_split_sgf_vp_kv2_g12(
    device bfloat *queries [[buffer(0)]],
    device float *partials [[buffer(1)]],
    device float *statistics [[buffer(2)]],
    device const SplashKvPage *page_table [[buffer(3)]],
    constant SplashPrefillAttentionParams &params [[buffer(4)]],
    device const uint *block_mask [[buffer(5)]],
    device const uint *lists [[buffer(6)]],
    constant Qwen4QsaMaskParams &mask [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint query_words[6 * 1024];
  constexpr uint D = SplashKvHeadDimension;
  const uint kv_head = group.x >> 1, tile = group.y, split = group.z;
  const uint tile_start = tile * SplashPrefillTileRows;
  if (!splash_prefill_attention_contract_valid(params) ||
      kv_head >= 2 || split >= params.split_count ||
      tile_start >= params.rows || sg >= 6)
    return;
  const uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  const ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) * 12 * D;
  const ulong slot = (ulong(tile) * 2 + kv_head) * params.split_count + split;
  const uint half_index = group.x & 1;
  const uint2 li = reinterpret_cast<device const uint2 *>(lists)[(tile * 2 + half_index) * 6 + sg];
  q8sgf::tile_vp<2, 12, 2>(
      queries + tile_offset, page_table, params.kv, kv_head,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      split, partials, statistics, slot, sg, lane, query_words, half_index,
      block_mask + ulong(mask.row0 + tile_start) * mask.mask_words,
      mask.mask_words, lists + mask.list_offset + li.x, li.y);
}

// N1 list builder: one threadgroup of 256 threads per segment s = (tile * 2 + half) * 6 + sg. The segment's
// list is the ascending block ids of the union of its fused rows' (at most two query rows') admitted blocks,
// causal-trimmed, padded with VpInvalid to whole virtual pages of eight; s owns mask.list_stride entries.
// The host sets list_stride to 2 * (top_blocks + 1) rounded up to 8, which a segment's union never exceeds.
kernel void prefill_attention_vp_listbuild(
    device const uint *block_mask [[buffer(0)]],
    constant SplashPrefillAttentionParams &params [[buffer(1)]],
    device uint *lists [[buffer(2)]],
    constant Qwen4QsaMaskParams &mask [[buffer(3)]],
    uint seg [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint simd_sum[8];
  const uint Capacity = mask.list_stride;
  const uint tile = seg / 12, half_index = (seg / 6) & 1, sg = seg % 6;
  const uint tile_start = tile * SplashPrefillTileRows;
  const uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  const uint base = params.committed_tokens + tile_start;
  // The query rows of the segment's first and last fused row, clamped as the attention kernel does.
  const uint first_fused = half_index * 48 + sg * 8;
  const uint r0 = min(first_fused / 12, active_rows - 1), r1 = min((first_fused + 7) / 12, active_rows - 1);
  device const uint *row0 = block_mask + ulong(mask.row0 + tile_start + r0) * mask.mask_words;
  device const uint *row1 = block_mask + ulong(mask.row0 + tile_start + r1) * mask.mask_words;
  const uint last0 = (base + r0) / 4, last1 = (base + r1) / 4;
  // Thread t owns `per` consecutive words, so the thread prefix sum keeps the ids ascending.
  const uint words = min(mask.mask_words, last1 / 32 + 1);
  const uint per = (words + 255) / 256;
  const uint w_begin = min(t * per, words), w_end = min(w_begin + per, words);
  const auto admitted_word = [&](uint w) {
    const uint first = w * 32;
    const uint m0 = last0 >= first + 31 ? ~0u : last0 < first ? 0u : (2u << (last0 - first)) - 1u;
    const uint m1 = last1 >= first + 31 ? ~0u : last1 < first ? 0u : (2u << (last1 - first)) - 1u;
    return (row0[w] & m0) | (row1[w] & m1);
  };
  uint count = 0;
  for (uint w = w_begin; w < w_end; ++w)
    count += popcount(admitted_word(w));
  const uint in_simd = simd_prefix_exclusive_sum(count);
  if (lane == 31)
    simd_sum[simd] = in_simd + count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = 0, total = 0;
  for (uint s = 0; s < 8; ++s) {
    if (s < simd)
      before += simd_sum[s];
    total += simd_sum[s];
  }
  if (total > Capacity) {
    if (t == 0)
      reinterpret_cast<device uint2 *>(lists)[seg] = uint2(seg * Capacity, 0);
    return;
  }
  const uint vpages = max(1u, (total + 7) / 8);
  device uint *out = lists + mask.list_offset + ulong(seg) * Capacity;
  uint position = before + in_simd;
  for (uint w = w_begin; w < w_end; ++w)
    for (uint bits = admitted_word(w); bits; bits &= bits - 1)
      out[position++] = w * 32 + ctz(bits);
  for (uint e = total + t; e < vpages * 8; e += 256)
    out[e] = 0xFFFFFFFFu;
  if (t == 0)
    reinterpret_cast<device uint2 *>(lists)[seg] = uint2(seg * Capacity, vpages);
}
