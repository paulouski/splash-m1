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

// The 8-row tile of the unaligned experts (Qwen3.8-Flash-Next's gate and up passes: K = 2560, 640 columns) with K split
// in two halves over a second pair of simdgroups, the second half's sums added to the first's through threadgroup
// memory: the K chain a decode tile with few live rows waits on is halved. Four simdgroups: columns [0, 32) and
// [32, 64) of the tile for the first half of K, then for the second. Each half fills the same pair table.
template <GgufEpilogue Ep>
kernel void moe_expert_gguf_mma_k2(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                   device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                                   device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                                   device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                                   device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                                   constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[2 * GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[kQuantPairTableEntries];
  threadgroup float2 second[2 * 32 * FN];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment_for<true>(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * 8 * p.input_size;
  const ulong out = ulong(group.y) * 8 * p.output_size;
  const uint half_index = simd_group >> 1, column_group = simd_group & 1;
  const uint origin = group.x * GGUF_TILE_COLUMNS + column_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  const uint steps = p.input_size / kStep;
  float2 acc[FN][1];
  zero(acc);
  accumulate_any<1>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin + moe_gguf_first_row(tile.expert, p),
                    stage + simd_group * kSimdgroupStage, tl, thread_index % GGUF_STAGED_THREADS, GGUF_STAGED_THREADS, simd_lane,
                    half_index * (steps / 2),
                    half_index ? steps : steps / 2, acc);
  if (half_index)
    for (uint i = 0; i < FN; ++i) second[(column_group * 32 + simd_lane) * FN + i] = acc[i][0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (half_index) return;
  for (uint i = 0; i < FN; ++i) acc[i][0] += second[(column_group * 32 + simd_lane) * FN + i];
  elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
    const ulong o = out + ulong(row) * p.output_size + origin + column;
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[o]));
    output[o] = bfloat(v);
  });
}
template [[host_name("moe_expert_gguf_mma_m8_a_u_k2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma_k2<EpNone>;
template [[host_name("moe_expert_gguf_mma_m8_g_u_k2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma_k2<EpUpWithGate>;
