// GGUF MoE experts for Apple7/8 (M1/M2) on the simdgroup MMA tile (kernels/common/gguf_mma_tile.h): the bindings,
// grid and outputs of moe_expert_gguf_m* (kernels/shared/moe_gguf.metal), whose MPP matmul2d runs far below the MMA
// rate there. A tile runs the 8-, 16- or 32-row MMAs that hold its live rows.
// Unaligned: experts of an output_size that is not a multiple of QUANT_TILE_ROWS (Qwen3.8-Flash-Next's 640), whose
// planes do not start on a plane tile: the routed segment keeps its base and the expert's first row joins the origin.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

using namespace gguf_mma;

template <ushort Rows, GgufEpilogue Ep, bool Unaligned = false>
kernel void moe_expert_gguf_mma(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                                device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                                device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                                device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                                constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
                                uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment_for<Unaligned>(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  const auto run = [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    if constexpr (Unaligned)
      accumulate_any<FM>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin + moe_gguf_first_row(tile.expert, p),
                         stage + simd_group * kSimdgroupStage, tl, thread_index, GGUF_STAGED_THREADS, simd_lane, 0,
                         p.input_size / kStep, acc);
    else
      accumulate_any<FM>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, stage + simd_group * kSimdgroupStage, tl,
                         thread_index, GGUF_STAGED_THREADS, simd_lane, 0, p.input_size / kStep, acc);
    elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[o]));
      output[o] = bfloat(v);
    });
  };
  if constexpr (Rows == 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 16) run(integral_constant<uint, 2>{});
  else run(integral_constant<uint, 4>{});
}
using MoeExpertGgufMmaKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                    device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                    device bfloat *, device bfloat *, constant MoeGgufExpertParams &, uint2, uint, uint);
template [[host_name("moe_expert_gguf_mma_m8_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone>;
template [[host_name("moe_expert_gguf_mma_m8_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate>;
template [[host_name("moe_expert_gguf_mma_m32_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone>;
template [[host_name("moe_expert_gguf_mma_m32_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate>;
template [[host_name("moe_expert_gguf_mma_m8_a_u")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone, true>;
template [[host_name("moe_expert_gguf_mma_m8_g_u")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate, true>;
template [[host_name("moe_expert_gguf_mma_m32_a_u")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone, true>;
template [[host_name("moe_expert_gguf_mma_m32_g_u")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate, true>;

// Gate and up of the unaligned 8-row tile in one dispatch (Qwen3.8-Flash-Next's 640-wide experts, K = 2560): the first
// four simdgroups run the gate, the last four the up, each with K split in two halves over a second pair of simdgroups
// (the second half's fp32 sums are added to the first's through threadgroup memory) and a column pair of the tile, sharing the tile's
// input rows, which are read from `input` through the tile's routes instead of a gathered copy (rows beyond the
// routed ones read as zeros, as moe_gather_rows writes them). The gate's sums, rounded to bf16, reach the up's
// simdgroups through threadgroup memory for the up's SwiGLU epilogue, so only the activated rows are stored.
kernel void moe_expert_gguf_mma_gu_k2(
    device const bfloat *input [[buffer(0)]], device const uint *routes [[buffer(1)]],
    device const MoeTileDescriptor *tiles [[buffer(2)]], device const uint *tile_count [[buffer(3)]],
    device uchar *gw0 [[buffer(4)]], device uchar *gw1 [[buffer(5)]], device uchar *gmeta [[buffer(6)]],
    device uchar *gsw0 [[buffer(7)]], device uchar *gsw1 [[buffer(8)]], device uchar *gsmeta [[buffer(9)]],
    device uchar *uw0 [[buffer(10)]], device uchar *uw1 [[buffer(11)]], device uchar *umeta [[buffer(12)]],
    device uchar *usw0 [[buffer(13)]], device uchar *usw1 [[buffer(14)]], device uchar *usmeta [[buffer(15)]],
    device bfloat *output [[buffer(16)]], constant MoeGgufGateUpParams &p [[buffer(17)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[8 * kSimdgroupStage];
  threadgroup half2 tl[2 * kQuantPairTableEntries];
  threadgroup float2 second[2 * 2 * 32 * FN];
  threadgroup float gate_sums[2 * 32 * FN * 2];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment sg_gate = moe_gguf_segment_for<true>(tile.expert, p.gate, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  const MoeGgufSegment sg_up = moe_gguf_segment_for<true>(tile.expert, p.up, uw0, uw1, umeta, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  // Every thread fills both formats' pair tables (a barrier each for codebook formats, uniform per tile).
  quant_format_switch(sg_gate.format, [&](auto format) {
    quant_pair_table<decltype(format)>(tl, thread_index, 256);
  });
  quant_format_switch(sg_up.format, [&](auto format) {
    quant_pair_table<decltype(format)>(tl + kQuantPairTableEntries, thread_index, 256);
  });
  const bool up = simd_group >= 4;
  const uint sg = simd_group & 3, half_index = sg >> 1, column_group = sg & 1;
  const MoeGgufSegment s = up ? sg_up : sg_gate;
  const sgmatrix::Lane l = sgmatrix::lane_map(simd_lane);
  RoutedRows x;
#pragma unroll
  for (uint e = 0; e < 2; ++e) {
    const uint route = routes[group.y * 8 + l.fn + e];
    x.row[e] = route == ~0u ? nullptr : input + ulong(route / p.routes_per_row) * p.gate.input_size;
  }
  const ulong out = ulong(group.y) * 8 * p.gate.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + column_group * kColumns;
  const uint steps = p.gate.input_size / kStep;
  float2 acc[FN][1];
  zero(acc);
  quant_format_switch(s.format, [&](auto format) {
    accumulate<decltype(format), 1, RoutedRows, false>(x, s.w0, s.w1, s.meta, p.gate.input_size,
                                    origin + moe_gguf_first_row(tile.expert, up ? p.up : p.gate),
                                    stage + simd_group * kSimdgroupStage, tl + up * kQuantPairTableEntries, simd_lane,
                                    half_index * (steps / 2), half_index ? steps : steps / 2, acc);
  });
  const uint slot = ((up * 2 + column_group) * 32 + simd_lane) * FN;
  if (half_index)
    for (uint i = 0; i < FN; ++i) second[slot + i] = acc[i][0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!half_index) {
    for (uint i = 0; i < FN; ++i) acc[i][0] += second[slot + i];
    if (!up)
      elements(acc, l, [&](uint row, uint column, float v) {
        gate_sums[(column_group * 32 + simd_lane) * FN * 2 + (column >> 3) * 2 + row - l.fn] = float(bfloat(v));
      });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (half_index || !up) return;
  elements(acc, l, [&](uint row, uint column, float v) {
    const float gate = gate_sums[(column_group * 32 + simd_lane) * FN * 2 + (column >> 3) * 2 + row - l.fn];
    output[out + ulong(row) * p.gate.output_size + origin + column] = bfloat(float(bfloat(v)) * splash_silu(gate));
  });
}
