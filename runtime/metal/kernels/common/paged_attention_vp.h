#pragma once

#include "metal/kernels/common/paged_attention_sgf.h"

// N1: "virtual page" variant of q8sgf::tile (int8, G12, masked prefill).
// A simdgroup walks a list (prefill_attention_vp_listbuild) of 4-token blocks (global block id = logical page * 8 +
// block in page, VpInvalid pads the last virtual page to 8). Eight list entries form one
// 32-token virtual page and run the stock page body unchanged: slot j of the page_token map
// (2 * (k >> 1) + (s >> 1)) is list[8 v + j]. Only addressing changes: K/key scales read the
// block of the score slot's page, V reads two 4-byte loads (slots fn, fn + 1) from the real
// dim-major layout (possibly two pages), admission uses the real token and the row's own bit.
namespace q8sgf {
constexpr constant uint VpInvalid = 0xFFFFFFFFu;

__attribute__((always_inline)) inline uint vp_bit(device const uint *row, uint block) {
  return block == VpInvalid ? 0u : (row[block >> 5] >> (block & 31)) & 1u;
}

// list: 8 * virtual_pages entries of this simdgroup (8-entry aligned).
template <uint KVHeads, uint QueryHeadsPerKVHead, uint Halves>
__attribute__((always_inline)) inline void tile_vp(
    device const bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    uint sg, uint lane, threadgroup uint *query_words, uint half_index,
    device const uint *block_mask, uint mask_words, device const uint *list,
    uint virtual_pages) {
  using CacheElement = int8_t;
  constexpr uint G = QueryHeadsPerKVHead;
  constexpr uint N = SplashKvPageTokens;
  constexpr uint D = SplashKvHeadDimension;
  static_assert(N == 32 && D == 256, "fragment maps assume Page32 and D = 256");
  const uint visible = committed_tokens + active_rows;
  const uint pages = virtual_pages;
  // The reduce reads the splits of the stock page count; the list's pages are spread over those (trailing splits may be empty).
  const uint stock_pages = splash_attention_pages(visible);
  const uint stock_per_split = splash_attention_pages_per_split(stock_pages, splits);
  const uint written = (stock_pages + stock_per_split - 1) / stock_per_split;
  const uint per_split = splash_attention_pages_per_split(pages, written);
  const uint page_begin = split * per_split;
  // A split past the list still writes its (empty) partials: the reduce reads the stock page count's splits.
  const uint page_end = min(pages, page_begin + per_split);
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  uint row = sg * 8 + fn;
  if constexpr (Halves > 1)
    row += half_index * (G / Halves) * 8;
  const uint2 limit(
      min(visible, committed_tokens + min(row / G, active_rows - 1) + 1),
      min(visible, committed_tokens + min((row + 1) / G, active_rows - 1) + 1));

  threadgroup uint *queries = query_words + sg * 1024 + lane;
  {
    device const ushort *q0 =
        reinterpret_cast<device const ushort *>(tile_queries) + row * D;
    device const ushort *q1 = q0 + D;
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint t = 0; t < 8; ++t) {
        const uint d = 64 * b + block_dimension(fm, t);
        queries[(b * 8 + t) * 32] = uint(q0[d]) | (uint(q1[d]) << 16);
      }
  }
  device const uint *mask_row0 =
      block_mask + ulong(min(row / G, active_rows - 1)) * mask_words;
  device const uint *mask_row1 =
      block_mask + ulong(min((row + 1) / G, active_rows - 1)) * mask_words;
  float2 output[32];
#pragma unroll
  for (uint d = 0; d < 32; ++d)
    output[d] = float2(0.0f);
  float2 row_max(-INFINITY), row_sum(0.0f);
  const uint in_block = 2 * (fm & 1);
  const SplashKvAddressing<KVHeads, CacheElement> addressing(kv, kv_head);

  for (uint index = page_begin; index < page_end; ++index) {
    // Score slots 2 (fm >> 1) + {0, 1}; value slots fn + {0, 1} (fn is even).
    const uint2 gs = *reinterpret_cast<device const uint2 *>(list + 8 * index + 2 * (fm >> 1));
    const uint2 gv = *reinterpret_cast<device const uint2 *>(list + 8 * index + fn);
    const uint2 blocks(vp_bit(mask_row0, gs.x) | (vp_bit(mask_row0, gs.y) << 1),
                       vp_bit(mask_row1, gs.x) | (vp_bit(mask_row1, gs.y) << 1));
    if (simd_all(blocks.x == 0 && blocks.y == 0))
      continue;
    // Invalid slots read block 0 of page 0 and are never admitted.
    const uint ga = gs.x == VpInvalid ? 0u : gs.x, gb = gs.y == VpInvalid ? 0u : gs.y;
    const uint gc = gv.x == VpInvalid ? 0u : gv.x, gd = gv.y == VpInvalid ? 0u : gv.y;
    const SplashKvPageTensors<CacheElement> ta = addressing.page(page_table[ga >> 3]);
    const SplashKvPageTensors<CacheElement> tb = addressing.page(page_table[gb >> 3]);
    const SplashKvPageTensors<CacheElement> tc = addressing.page(page_table[gc >> 3]);
    const SplashKvPageTensors<CacheElement> td = addressing.page(page_table[gd >> 3]);
    const uint slot_a = (ga & 7) * 4 + in_block, slot_b = (gb & 7) * 4 + in_block;
    device const CacheElement *key_row[4] = {
        ta.keys + slot_a * D + 8 * fn, ta.keys + (slot_a + 1) * D + 8 * fn,
        tb.keys + slot_b * D + 8 * fn, tb.keys + (slot_b + 1) * D + 8 * fn};
    const float2 k01 = *reinterpret_cast<device const float2 *>(ta.key_scales + slot_a);
    const float2 k45 = *reinterpret_cast<device const float2 *>(tb.key_scales + slot_b);
    const float2 v01 = *reinterpret_cast<device const float2 *>(ta.value_scales + slot_a);
    const float2 v45 = *reinterpret_cast<device const float2 *>(tb.value_scales + slot_b);
    const float key_scale[4] = {k01.x, k01.y, k45.x, k45.y};
    const float value_scale[4] = {v01.x, v01.y, v45.x, v45.y};
    const uint token_of[4] = {ga * 4 + in_block, ga * 4 + in_block + 1,
                              gb * 4 + in_block, gb * 4 + in_block + 1};

    float2 scores[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      scores[s] = float2(0.0f);
#pragma unroll
    for (uint b = 0; b < 4; ++b)
#pragma unroll
      for (uint s = 0; s < 4; ++s) {
        KeyRun<CacheElement> run;
        run.load(key_row[s] + 64 * b);
#pragma unroll
        for (uint t = 0; t < 8; ++t)
          mma(scores[s], run.step(t), bfloat_pair(queries[(b * 8 + t) * 32]));
      }

    float2 local_max(-INFINITY);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      float2 score = scores[s];
      score *= key_scale[s];
      score *= 0.0625f;
      scores[s] = float2(admitted(token_of[s], limit.x, blocks.x, s) ? score.x : -INFINITY,
                         admitted(token_of[s], limit.y, blocks.y, s) ? score.y : -INFINITY);
      local_max = max(local_max, scores[s]);
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    local_max = max(local_max, simd_shuffle_xor(local_max, 4));
    local_max = max(local_max, simd_shuffle_xor(local_max, 16));
    const float2 next_max = max(row_max, local_max);
    float2 probability[4];
    float2 local_sum(0.0f);
#pragma unroll
    for (uint s = 0; s < 4; ++s) {
      probability[s] = float2(
          admitted(token_of[s], limit.x, blocks.x, s) ? fast::exp(scores[s].x - next_max.x) : 0.0f,
          admitted(token_of[s], limit.y, blocks.y, s) ? fast::exp(scores[s].y - next_max.y) : 0.0f);
      local_sum += probability[s];
    }
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 16);
    const float2 scale(
        next_max.x == -INFINITY || next_max.x == row_max.x
            ? 1.0f : fast::exp(row_max.x - next_max.x),
        next_max.y == -INFINITY || next_max.y == row_max.y
            ? 1.0f : fast::exp(row_max.y - next_max.y));
    row_sum = row_sum * scale + local_sum;
    row_max = next_max;
    if (simd_any(scale.x != 1.0f || scale.y != 1.0f)) {
#pragma unroll
      for (uint d = 0; d < 32; ++d)
        output[d] *= scale;
    }
    float2 weights[4];
#pragma unroll
    for (uint s = 0; s < 4; ++s)
      weights[s] = float2(
          admitted(token_of[s], limit.x, blocks.x, s) ? float(bfloat(probability[s].x * value_scale[s])) : 0.0f,
          admitted(token_of[s], limit.y, blocks.y, s) ? float(bfloat(probability[s].y * value_scale[s])) : 0.0f);

    // Dim-major V: a lane's tokens of slot fn and fn + 1 are two 4-byte runs, in up to two pages.
    device const CacheElement *value_c = tc.values + fm * N + (gc & 7) * 4;
    device const CacheElement *value_d = td.values + fm * N + (gd & 7) * 4;
#pragma unroll
    for (uint d = 0; d < 32; ++d) {
      ValueRun<CacheElement> run;
      run.words = uint2(*reinterpret_cast<device const uint *>(value_c + d * 8 * N),
                        *reinterpret_cast<device const uint *>(value_d + d * 8 * N));
      mma(output[d], run.step(0), weights[0]);
      mma(output[d], run.step(1), weights[1]);
      mma(output[d], run.step(2), weights[2]);
      mma(output[d], run.step(3), weights[3]);
    }
  }

  device float *target = partials + slot * (8 * G) * D;
#pragma unroll
  for (uint d = 0; d < 32; ++d) {
    target[row * D + 8 * d + fm] = output[d].x;
    target[(row + 1) * D + 8 * d + fm] = output[d].y;
  }
  if (fm == 0) {
    device float *stat = statistics + (slot * (8 * G) + row) * 2;
    stat[0] = row_max.x;
    stat[1] = row_sum.x;
    stat[2] = row_max.y;
    stat[3] = row_sum.y;
  }
}
} // namespace q8sgf
