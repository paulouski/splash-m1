// The Neural Engine's operands for a projection's trailing columns (ops::AneOffload).
#pragma clang fp reassociate(off)
#include "metal/abi/Ane.h"
#include "metal/kernels/common/gguf_staged.h"

using namespace metal;

constant constexpr uint kAneStageThreads = 256;

// One threadgroup per weight row: each thread dequantizes a group of 32 inputs through the format's dequantizer
// (kernels/common/gguf_staged.h), the threadgroup takes the row's largest magnitude m, and the row goes out as int8
// round(w * 127 / m). The scale the ANE output is multiplied by is m * 128 / 127: the program dequantizes int8 by
// 1/128. K is at most kAneStageThreads groups.
kernel void ane_stage(device uchar *w0 [[buffer(0)]], device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]],
                      device char *surface [[buffer(3)]], device float *scales [[buffer(4)]],
                      constant AneStageParams &p [[buffer(5)]], uint row_in_grid [[threadgroup_position_in_grid]],
                      uint thread_index [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[kQuantPairTableEntries];
  threadgroup float peaks[kAneStageThreads / 32];
  const uint groups = p.input_size / 32, row = p.first_row + row_in_grid;
  const bool live = thread_index < groups;
  half4 lo[4], hi[4];
  float peak = 0.0f;
  quant_format_switch(p.format, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, thread_index, kAneStageThreads);
    if (!live) return;
    const uint units = groups / F::MetaGroups;
    const typename F::Payload packed = F::load(w0 + quant_tile_index(row, thread_index, groups) * F::P0,
                                               w1 + quant_tile_index(row, thread_index, groups) * F::P1);
    const typename F::Meta header = F::loadMeta(meta + quant_tile_index(row, thread_index / F::MetaGroups, units) * F::MetaBytes);
    dequant32_chunks<F>(packed, header, thread_index % F::MetaGroups, tl, [&](ushort c, half4 l, half4 h) {
      lo[c] = l;
      hi[c] = h;
    });
  });
  if (live) {
    for (ushort c = 0; c < 4; ++c) {
      const float4 a = abs(float4(lo[c])), b = abs(float4(hi[c]));
      peak = max(peak, max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))));
    }
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simdgroup] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = 0.0f;
  for (uint i = 0; i < kAneStageThreads / 32; ++i) m = max(m, peaks[i]);
  const float scale = m > 0.0f ? m / 127.0f * 128.0f : 128.0f;
  const uint surface_row = p.surface_row + row_in_grid;
  if (thread_index == 0) scales[surface_row] = scale;
  if (!live) return;
  const float inverse = 128.0f / scale;
  device char4 *out = (device char4 *)(surface + ulong(surface_row) * p.surface_stride + thread_index * 32);
  for (ushort c = 0; c < 4; ++c) {
    out[c] = char4(clamp(rint(float4(lo[c]) * inverse), -127.0f, 127.0f));
    out[4 + c] = char4(clamp(rint(float4(hi[c]) * inverse), -127.0f, 127.0f));
  }
}

// Rows of bf16 to the fp16 input surface, clamped to the finite fp16 range.
kernel void ane_pack_bf16(device const bfloat *input [[buffer(0)]], device half *surface [[buffer(1)]],
                          constant AnePackParams &p [[buffer(2)]], uint2 index [[thread_position_in_grid]]) {
  if (index.x >= p.width || index.y >= p.rows) return;
  const float v = float(input[ulong(index.y) * p.width + index.x]);
  surface[ulong(index.y) * p.surface_stride + index.x] = half(clamp(v, -65504.0f, 65504.0f));
}

// The fp16 output surface times each column's scale, into the output's columns from out_column on; a value that is
// not finite sets *status.
kernel void ane_unpack_bf16(device const half *surface [[buffer(0)]], device const float *scales [[buffer(1)]],
                            device bfloat *output [[buffer(2)]], device atomic_uint *status [[buffer(3)]],
                            constant AnePackParams &p [[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
  if (index.x >= p.width || index.y >= p.rows) return;
  const float v = float(surface[ulong(index.y) * p.surface_stride + index.x]);
  if (!isfinite(v)) atomic_store_explicit(status, 1u, memory_order_relaxed);
  output[ulong(index.y) * p.out_stride + p.out_column + index.x] = bfloat(v * scales[index.x]);
}
