#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/abi/QuantFormat.h"

#include <metal_stdlib>

using namespace metal;

// SPLASH_DRAFT_SHORTLIST (ops/Qwen4.hpp, model/DraftShortlist.hpp): the MTP draft head over a gathered subset of the
// logits head's rows. Gathered rows keep the tiled layout, so the production rows kernel scores them as a head of
// fewer rows.

// One thread per (selected row, group or meta unit); x is the row, so a simdgroup's destination writes are adjacent.
kernel void qwen4_shortlist_gather(device const uint *source0 [[buffer(0)]],
                                   device const uint *source1 [[buffer(1)]],
                                   device const uint *source_meta [[buffer(2)]],
                                   device uint *destination0 [[buffer(3)]],
                                   device uint *destination1 [[buffer(4)]],
                                   device uint *destination_meta [[buffer(5)]],
                                   device const uint *ids [[buffer(6)]],
                                   constant Qwen4GatherParams &p [[buffer(7)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  const uint i = group.x * 64 + tid;
  if (i >= p.count) return;
  const uint source = ids[i], destination = p.first + i, j = group.y;
  if (j < p.groups) {
    const uint words0 = p.plane0 / 4, words1 = p.plane1 / 4;
    const ulong a = quant_tile_index(source, j, p.groups), b = quant_tile_index(destination, j, p.groups);
    for (uint k = 0; k < words0; ++k) destination0[b * words0 + k] = source0[a * words0 + k];
    for (uint k = 0; k < words1; ++k) destination1[b * words1 + k] = source1[a * words1 + k];
  } else {
    const uint words = p.meta / 4, u = j - p.groups;
    const ulong a = quant_tile_index(source, u, p.units), b = quant_tile_index(destination, u, p.units);
    for (uint k = 0; k < words; ++k) destination_meta[b * words + k] = source_meta[a * words + k];
  }
}

// The order-preserving unsigned key of a float.
inline uint topk_key(float x) {
  const uint b = as_type<uint>(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void qwen4_topk_histogram(device const float *logits [[buffer(0)]],
                                 device atomic_uint *histogram [[buffer(1)]],
                                 constant Qwen4TopKParams &p [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  device const float *x = logits + ulong(group.y) * p.vocabulary;
  device atomic_uint *h = histogram + ulong(group.y) * 65536;
  for (uint e = 0; e < 4; ++e) {
    const uint index = group.x * 1024 + e * 256 + tid;
    if (index < p.vocabulary) atomic_fetch_add_explicit(h + (topk_key(x[index]) >> 16), 1u, memory_order_relaxed);
  }
}

// The bin holding the k-th largest key of each row; clears the histogram for the next call.
kernel void qwen4_topk_threshold(device uint *histogram [[buffer(0)]],
                                 device uint *threshold [[buffer(1)]],
                                 device uint *count [[buffer(2)]],
                                 constant Qwen4TopKParams &p [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[1024];
  device uint *h = histogram + ulong(row) * 65536;
  uint s = 0;
  for (uint b = 0; b < 64; ++b) s += h[t * 64 + b];
  sums[t] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint acc = 0, bin = 0;
    for (int u = 1023; u >= 0; --u) {
      if (acc + sums[u] >= p.k) {
        for (int b = 63; b >= 0; --b) {
          acc += h[u * 64 + b];
          if (acc >= p.k) {
            bin = uint(u * 64 + b);
            break;
          }
        }
        break;
      }
      acc += sums[u];
    }
    threshold[row] = bin;
    count[2 * row] = 0;
    count[2 * row + 1] = 0;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint b = 0; b < 64; ++b) h[t * 64 + b] = 0;
}

kernel void qwen4_topk_collect(device const float *logits [[buffer(0)]],
                               device const uint *threshold [[buffer(1)]],
                               device atomic_uint *count [[buffer(2)]],
                               device uint *keys [[buffer(3)]],
                               device uint *ids [[buffer(4)]],
                               constant Qwen4TopKParams &p [[buffer(5)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  device const float *x = logits + ulong(group.y) * p.vocabulary;
  const uint bin = threshold[group.y];
  for (uint e = 0; e < 4; ++e) {
    const uint index = group.x * 1024 + e * 256 + tid;
    if (index >= p.vocabulary) continue;
    const uint key = topk_key(x[index]);
    if ((key >> 16) < bin) continue;
    // Keys above the threshold bin (fewer than k) fill from the front, the bin's own from the back, so the
    // overflow of a crowded bin never drops a key above it.
    const bool above = (key >> 16) > bin;
    const uint slot = atomic_fetch_add_explicit(count + 2 * group.y + (above ? 0 : 1), 1u, memory_order_relaxed);
    const uint place = above ? slot : p.capacity - 1 - slot;
    if (slot < p.capacity) {
      keys[group.y * p.capacity + place] = key;
      ids[group.y * p.capacity + place] = index;
    }
  }
}

// Bitonic sort of a row's candidates (at most p.capacity <= 2048), descending by key and, among equal keys, by
// ascending id; writes its first k ids. The candidates are the front run of keys above the threshold bin and the
// back run of the bin's own.
kernel void qwen4_topk_sort(device const uint *keys [[buffer(0)]],
                            device const uint *ids [[buffer(1)]],
                            device const uint *count [[buffer(2)]],
                            device uint *top_ids [[buffer(3)]],
                            constant Qwen4TopKParams &p [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint sorted_keys[QWEN4_TOPK_CANDIDATES], sorted_ids[QWEN4_TOPK_CANDIDATES];
  const uint above = min(count[2 * row], p.capacity);
  const uint tied = min(count[2 * row + 1], p.capacity - above);
  const uint n = above + tied;
  for (uint i = t; i < QWEN4_TOPK_CANDIDATES; i += 1024) {
    const uint place = i < above ? i : p.capacity - tied + (i - above);
    sorted_keys[i] = i < n ? keys[row * p.capacity + place] : 0u;
    sorted_ids[i] = i < n ? ids[row * p.capacity + place] : 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= QWEN4_TOPK_CANDIDATES; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      const uint lo = ((t & ~(j - 1)) << 1) | (t & (j - 1)), hi = lo + j;
      const bool descending = (lo & k) == 0;
      const uint a = sorted_keys[lo], b = sorted_keys[hi], ia = sorted_ids[lo], ib = sorted_ids[hi];
      const bool lo_after = a < b || (a == b && ia > ib);
      const bool hi_after = b < a || (a == b && ib > ia);
      if (descending ? lo_after : hi_after) {
        sorted_keys[lo] = b;
        sorted_keys[hi] = a;
        sorted_ids[lo] = ib;
        sorted_ids[hi] = ia;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = t; i < p.k; i += 1024) top_ids[row * p.k + i] = sorted_ids[i];
}

// qwen4_mtp_argmax over a shortlist head: the logits rows are p.vocabulary (the shortlist's rows) apart, and the
// argmax index maps to its token id through `map`.
kernel void qwen4_mtp_argmax_map(device const float *logits [[buffer(0)]],
                                 device uint *proposals [[buffer(1)]],
                                 device uint *candidates [[buffer(2)]],
                                 device const uint *map [[buffer(3)]],
                                 constant Qwen4MtpParams &p [[buffer(4)]],
                                 uint lane_index [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint v = tid; v < p.vocabulary; v += threads) {
    const float value = x[v];
    if (value > best) {
      best = value;
      id = v;
    }
  }
  // Ties toward the lower row; a padding row repeats an earlier row's id, so the result is the same.
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_id = simd_shuffle_down(id, offset);
    if (other > best || (other == best && other_id < id)) {
      best = other;
      id = other_id;
    }
  }
  if (lane == 0) {
    best_values[sg] = best;
    best_ids[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  const uint groups = (threads + 31) / 32;
  for (uint g = 1; g < groups; ++g)
    if (best_values[g] > best || (best_values[g] == best && best_ids[g] < id)) {
      best = best_values[g];
      id = best_ids[g];
    }
  id = map[id < p.vocabulary ? id : 0];
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    proposals[lane_index * p.proposals + slot] = id;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = id;
  }
}
