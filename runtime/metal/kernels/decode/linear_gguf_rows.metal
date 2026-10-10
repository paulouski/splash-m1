// One to four rows of a decode projection as dot products over the packed planes, for the vocabulary head, whose
// 8-row MMA tile spends half its time on the seven rows nobody reads. A simdgroup owns 32 columns (a lane one column,
// as the staged MMA tile) and walks K in steps of one group: the lane dequantizes its group to half (the staged tile's
// values) and takes its dot product with each row's inputs in fp32, four partial sums per row (one per chunk) added
// in a fixed order at the end. Grid (column tiles, lanes), two simdgroups.
#pragma clang fp reassociate(off)
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"

constant constexpr uint kRowsStep = 32;

// bf16 element j of four packed in v, as fp32.
inline float rows_bf16(uint2 v, uint j) {
  const uint w = j < 2 ? v.x : v.y;
  return as_type<float>((j & 1) ? (w & 0xFFFF0000u) : (w << 16));
}

template <class F, uint R>
kernel void gguf_rows(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                      device uchar *meta [[buffer(3)]], device float *output [[buffer(4)]],
                      constant GgufRowsParams &p [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint column0 = group.x * GGUF_TILE_COLUMNS + simd_group * 32, steps = p.input_size / kRowsStep;
  const uint groups = steps, units = groups / F::MetaGroups;
  const uint plane_tile = column0 / QUANT_TILE_ROWS, plane_row = column0 % QUANT_TILE_ROWS + lane;
  device uchar *c0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *c1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *cm = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  device const bfloat *x = input + ulong(group.y * SPLASH_TARGET_VERIFY_ROWS + p.first_row) * p.input_size;
  threadgroup half2 *tl = nullptr;
  float acc[R][4];
#pragma unroll
  for (uint r = 0; r < R; ++r)
#pragma unroll
    for (uint c = 0; c < 4; ++c) acc[r][c] = 0.0f;
  typename F::Payload packed = F::load(c0, c1);
  uint unit = 0;
  typename F::Meta hdr = F::loadMeta(cm);
  for (uint step = 0; step < steps; ++step) {
    typename F::Payload next_packed = packed;
    typename F::Meta next_hdr = hdr;
    uint next_unit = unit;
    if (step + 1 < steps) {
      next_packed = F::load(c0 + ulong(step + 1) * QUANT_TILE_ROWS * F::P0, c1 + ulong(step + 1) * QUANT_TILE_ROWS * F::P1);
      next_unit = (step + 1) / F::MetaGroups;
      if (next_unit != unit) next_hdr = F::loadMeta(cm + ulong(next_unit) * QUANT_TILE_ROWS * F::MetaBytes);
    }
    dequant32_chunks<F>(packed, hdr, ushort(step % F::MetaGroups), tl, [&](ushort c, half4 lo, half4 hi) {
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        device const bfloat *row = x + ulong(r) * p.input_size + step * kRowsStep;
        const uint2 xl = *reinterpret_cast<device const uint2 *>(row + 4 * c);
        const uint2 xh = *reinterpret_cast<device const uint2 *>(row + 16 + 4 * c);
#pragma unroll
        for (uint t = 0; t < 4; ++t) acc[r][c] = fma(float(lo[t]), rows_bf16(xl, t), acc[r][c]);
#pragma unroll
        for (uint t = 0; t < 4; ++t) acc[r][c] = fma(float(hi[t]), rows_bf16(xh, t), acc[r][c]);
      }
    });
    packed = next_packed;
    hdr = next_hdr;
    unit = next_unit;
  }
#pragma unroll
  for (uint r = 0; r < R; ++r)
    output[ulong(group.y * SPLASH_TARGET_VERIFY_ROWS + p.first_row + r) * p.out_stride + column0 + lane] =
        (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
}

#define GGUF_ROWS(F, f, R)                                                                                          \
  template [[host_name("gguf_rows_" #f "_r" #R)]] [[kernel]] void gguf_rows<F, R>(                                  \
      device bfloat *, device uchar *, device uchar *, device uchar *, device float *, constant GgufRowsParams &,  \
      uint2, uint, uint);
GGUF_ROWS(FmtQ4K, q4k, 1) GGUF_ROWS(FmtQ4K, q4k, 2) GGUF_ROWS(FmtQ4K, q4k, 3) GGUF_ROWS(FmtQ4K, q4k, 4)
#undef GGUF_ROWS
