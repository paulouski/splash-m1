#pragma once

// Parameter layouts shared by host dispatch code and the kernels that feed the Neural Engine (ops::AneOffload).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Rows [first_row, first_row + rows) of a block segment (metal/abi/QuantFormat.h) of input_size inputs, as int8 rows
// at row `surface_row` of a surface of surface_stride bytes per row; each row's scale goes to scales[surface_row].
struct AneStageParams {
  uint32_t input_size;
  uint32_t format;
  uint32_t first_row;
  uint32_t rows;
  uint32_t surface_row;
  uint32_t surface_stride;
};

// `rows` rows of `width` bf16 values to fp16 rows of surface_stride halves, or back (a column offset into rows of
// out_stride values).
struct AnePackParams {
  uint32_t rows;
  uint32_t width;
  uint32_t surface_stride;
  uint32_t out_stride;
  uint32_t out_column;
};

static_assert(sizeof(AneStageParams) == 24, "ANE stage parameters are 24 bytes on both sides");
static_assert(sizeof(AnePackParams) == 20, "ANE pack parameters are 20 bytes on both sides");
