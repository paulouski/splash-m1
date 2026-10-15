#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct MoeRouteParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t experts;
  uint32_t top_k;
  // Bit r set: row r (of at most 32) selects no expert; 0 routes every row.
  uint32_t inactive_rows;
};

static_assert(sizeof(MoeRouteParams) == 20,
              "MoE routing parameters are 20 bytes on both sides");

// Every row carries top_k routed experts followed by the shared expert, whose
// id is `experts` and whose routing weight is the sigmoid of its scalar gate.
// Routes are sorted by expert. Tile t covers the `rows` live grouped rows
// from first_row, which is t * tile_rows when the layout is padded (padding
// rows carry the route ~0u) and the tile's offset into the experts' packed
// routes when compact (MoeGroupParams::compact: no padding rows; a matmul
// that reads past `rows` reads the next tile's rows and must not store them).
//
// Written by grouping kernels; the host uses this layout to size storage.
struct MoeTileDescriptor {
  uint32_t expert;
  uint32_t rows;
  uint32_t first_row;
};

static_assert(sizeof(MoeTileDescriptor) == 12,
              "MoE tile descriptors are 12 bytes on both sides");

struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  uint32_t compact;
};

static_assert(sizeof(MoeGroupParams) == 20,
              "MoE grouping parameters are 20 bytes on both sides");

struct MoeGatherParams {
  uint32_t tile_rows;
  uint32_t input_size;
  uint32_t routes_per_row;
  uint32_t compact;
};

static_assert(sizeof(MoeGatherParams) == 16,
              "MoE gather parameters are 16 bytes on both sides");

// SPLASH_MOE_STATS: copies a layer's routed expert ids and routing weights
// (rows x top_k of the rows x routes_per_row select output) to slot
// [layer][row][route] of a stats buffer that holds capacity rows per layer.
struct MoeStatsParams {
  uint32_t layer;
  uint32_t rows;
  uint32_t top_k;
  uint32_t routes_per_row;
  uint32_t capacity;
};

static_assert(sizeof(MoeStatsParams) == 20,
              "MoE stats parameters are 20 bytes on both sides");

struct MoeExpertParams {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t reserved0;
  uint64_t expert_stride_bytes_0;
  uint64_t expert_stride_bytes_1;
};

static_assert(sizeof(MoeExpertParams) == 32,
              "MoE expert parameters are 32 bytes on both sides");

// A GGUF expert pass (ops/MoE.cpp): every routed expert of the projection
// is one image segment of experts * output_size rows, expert e's planes
// starting at row e * output_size (a tile when output_size is a multiple of
// QUANT_TILE_ROWS); the shared expert (id `experts`) has a segment of its own. Formats are GGUF_FMT_* (metal/abi/QuantFormat.h).
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
};

static_assert(sizeof(MoeGgufExpertParams) == 20,
              "MoE GGUF expert parameters are 20 bytes on both sides");

// The fused gate + up pass over the unaligned 8-row tile (moe_expert_gguf_mma_gu_k2): both projections' parameters
// and the route divisor that maps a grouped row's route to its input row.
struct MoeGgufGateUpParams {
  MoeGgufExpertParams gate;
  MoeGgufExpertParams up;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGgufGateUpParams) == 44,
              "MoE GGUF gate/up parameters are 44 bytes on both sides");

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");
