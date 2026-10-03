#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A Q4 projection's matrix, for prefill and the decode tiles whose grid
// covers it (a K-split tile reads its split count from the grid). Separate
// from ops::LinearMatrix so host-only fields cannot change the ABI.
struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
  // Q5 only: the 256-row storage tiles [hi_tile_begin, hi_tile_end) that hold
  // nonzero hi bits; the others skip the hi plane.
  uint32_t hi_tile_begin = 0;
  uint32_t hi_tile_end = 0xFFFFFFFFu;
};

static_assert(sizeof(Q4Params) == 16,
              "Q4 projection parameters are 16 bytes on both sides");

// The persistent decode tiles' matrix and their grid's `groups`
// threadgroups, which stride over the column tiles. The stride is a
// parameter rather than [[threadgroups_per_grid]]: compiled at -O3, a loop
// striding by that attribute ran nondeterministically under Metal shader
// validation, which the kernel tests run with. The register-matrix (sgf)
// decode tiles take their K-split count here too.
struct Q4PersistentParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t groups;
  uint32_t hi_tile_begin = 0;
  uint32_t hi_tile_end = 0xFFFFFFFFu;
};

static_assert(sizeof(Q4PersistentParams) == 20,
              "Q4 persistent decode parameters are 20 bytes on both sides");

struct Q4PrefillTailParams {
  Q4Params projection;
  uint32_t row_tile_offset;
};

static_assert(sizeof(Q4PrefillTailParams) == 20,
              "Q4 prefill tail parameters are 20 bytes on both sides");
