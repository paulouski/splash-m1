#pragma once

#include "metal/kernels/common/paged_attention_tile.h"

// The tile one verify threadgroup owns: group.x is the KV head, group.y the
// history split and group.z the lane, whose parameters select the page table
// and the query tile. A threadgroup past its lane's split count, or whose
// lane fails the contract, is inactive and does nothing.
struct SplashVerifyTile {
  device bfloat *queries;
  device const SplashKvPage *page_table;
  ulong slot;
  uint kv_head;
  uint split;
  uint splits;
  uint committed_tokens;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline SplashVerifyTile splash_verify_attention_tile_at(
    device bfloat *queries, device const SplashKvPage *page_table0,
    device const SplashKvPage *page_table1, device const SplashKvPage *page_table2,
    device const SplashKvPage *page_table3,
    constant SplashVerifyAttentionParams *params, uint3 group) {
  constexpr uint D = SplashKvHeadDimension;
  SplashVerifyTile tile{};
  uint kv_head = group.x;
  uint split = group.y;
  uint batch = group.z;
  constant SplashVerifyAttentionParams &lane_params = params[batch];
  if (!splash_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || split >= lane_params.split_count)
    return tile;
  constexpr ulong group_stride =
      ulong(SPLASH_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  tile.queries = queries + (ulong(batch) * KVHeads + kv_head) * group_stride;
  tile.page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  tile.slot =
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits + split;
  tile.kv_head = kv_head;
  tile.split = split;
  tile.splits = lane_params.split_count;
  tile.committed_tokens = lane_params.committed_tokens;
  tile.active = true;
  return tile;
}
