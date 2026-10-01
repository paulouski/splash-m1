#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;
  // Q5 only: the 256-row storage tiles [hi_tile_begin, hi_tile_end) that hold
  // nonzero hi bits; the others skip the hi plane.
  uint32_t hi_tile_begin = 0;
  uint32_t hi_tile_end = 0xFFFFFFFFu;
};

static_assert(sizeof(Q4Params) == 20,
              "Q4 decode projection parameters are 20 bytes on both sides");

// Separate from ops::LinearMatrix so host-only fields cannot change the ABI.
struct Q4PrefillParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t hi_tile_begin = 0;
  uint32_t hi_tile_end = 0xFFFFFFFFu;
};

static_assert(sizeof(Q4PrefillParams) == 16,
              "Q4 prefill projection parameters are 16 bytes on both sides");

struct Q4PrefillTailParams {
  Q4PrefillParams projection;
  uint32_t row_tile_offset;
};

static_assert(sizeof(Q4PrefillTailParams) == 20,
              "Q4 prefill tail parameters are 20 bytes on both sides");
