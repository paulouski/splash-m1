#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(MoeExpertParams, expert_stride_bytes_0) == 16);

bool matches(const Q8Projection &projection, uint32_t output,
             uint32_t input) noexcept {
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t parameterBytes = elements / 32;
  const AffineWeights &planes = projection.planes;
  return planes.weights && planes.scales && planes.biases &&
         projection.outputSize == output && projection.inputSize == input &&
         planes.weights.sizeBytes() >= elements &&
         planes.scales.sizeBytes() >= parameterBytes &&
         planes.biases.sizeBytes() >= parameterBytes;
}

bool matches(const ExpertProjection &projection, uint32_t experts,
             uint32_t output, uint32_t input) noexcept {
  if (!projection.packed || projection.experts != experts || !experts ||
      projection.outputSize != output || projection.inputSize != input)
    return false;
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t payloadBytes = elements / 2 + elements / 16;
  const uint64_t stride = projection.expertStrideBytes;
  const uint64_t available = projection.packed.sizeBytes();
  if (!stride || stride < payloadBytes || stride % sizeof(uint16_t) ||
      available < payloadBytes)
    return false;
  // The shader reads [weights][BF16 scales][BF16 biases] at each stride.
  // Allow padding between experts, without requiring it after the last one.
  // Division proves the last payload fits without overflowing expert*stride.
  return uint64_t{experts - 1} <= (available - payloadBytes) / stride;
}

// A float segment holds [output][input] floats in plane0; a quantized one
// its planes in a GGUF_FMT_* format.
bool matches(const QuantizedSegment &segment, uint32_t output, uint32_t input,
             bool floatWeights) noexcept {
  if (!segment.plane0 || segment.isFloat() != floatWeights ||
      segment.outputSize != output || segment.inputSize != input)
    return false;
  return floatWeights
             ? segment.plane0.sizeBytes() >= uint64_t{output} * input * (segment.bf16Values ? sizeof(uint16_t) : sizeof(float))
             : segment.formatId < GGUF_FMT_COUNT && segment.meta;
}

bool matches(const BlockExpertProjection &projection, uint32_t experts,
             uint32_t output, uint32_t input) noexcept {
  return matches(projection.routed, experts * output, input, false) &&
         matches(projection.shared, output, input, false);
}

void validate(const MoeWeights &weights, MoeShape shape) {
  if (shape.weightLayout != weights.layout())
    throw std::invalid_argument("MoE weight layout does not match plan");
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (shape.weightLayout == WeightLayout::Block32) {
    const BlockMoeWeights &blocks = weights.blocks();
    if (!shape.valid() ||
        !matches(blocks.router, shape.experts, hidden, true) ||
        !matches(blocks.sharedScalarGate, 1, hidden, true) ||
        !matches(blocks.gate, shape.experts, intermediate, hidden) ||
        !matches(blocks.up, shape.experts, intermediate, hidden) ||
        !matches(blocks.down, shape.experts, hidden, intermediate))
      throw std::invalid_argument("block MoE weights do not match execution shape");
    return;
  }
  const AffineMoeWeights &affine = weights.affine();
  if (!shape.valid() || !matches(affine.router, 256, hidden) ||
      !matches(affine.sharedScalarGate, 256, hidden) ||
      !matches(affine.expertGate, shape.experts, intermediate, hidden) ||
      !matches(affine.expertUp, shape.experts, intermediate, hidden) ||
      !matches(affine.expertDown, shape.experts, hidden, intermediate) ||
      !matches(affine.sharedGate, 1, intermediate, hidden) ||
      !matches(affine.sharedUp, 1, intermediate, hidden) ||
      !matches(affine.sharedDown, 1, hidden, intermediate)) {
    throw std::invalid_argument("MoE weights do not match execution shape");
  }
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows,
                          bool splitExperts, bool compact, MoeGgufTile ggufTile) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  const uint64_t groupedRows = compact ? moeCompactGroupedRows(rows, shape) : uint64_t{tiles} * tileRows;
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  const uint32_t outputWidth = splitExperts ? widest : shape.hiddenSize;
  // The router's fp32 scores, a row of routerWidth() expert slots per row,
  // live in the grouped input until the gather overwrites them. Register
  // plans also hold the down pass's Table16 tiles there.
  const uint64_t scoreBytes = uint64_t{rows} * shape.routerWidth() * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(uint32_t),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * outputWidth * sizeof(uint16_t), sumsBytes};
}

// Pipelines, column tiles and threadgroup width of a decode plan's fused
// gate/up and down passes; see MoeExpertSimdgroups for the four-simdgroup
// form's geometry and measurements.
struct ExpertPasses final {
  const char *gateUp;
  const char *down;
  uint32_t gateUpColumns;
  uint32_t downColumns;
  uint32_t threads;
};

ExpertPasses fusedExpertPasses(const MoeConfig &config) noexcept {
  // Four simdgroups: gate/up 16 and down 32 columns each, arranged 2 x 2
  // over the 32-row tile and 4 x 1 over the 8-row tile.
  if (config.kernel == MoeExpertKernel::Register)
    return config.expertTile == MoeExpertTile::M32
               ? ExpertPasses{"moe_expert_gate_up_q4_mma_m32",
                              "moe_expert_down_q4_mma_m32", 32, 64, 128}
               : ExpertPasses{"moe_expert_gate_up_q4_mma_m8",
                              "moe_expert_down_q4_mma_m8", 64, 128, 128};
  const uint32_t threads = static_cast<uint32_t>(config.m8Simdgroups) * 32;
  if (config.m8Simdgroups == MoeExpertSimdgroups::Four)
    return {"moe_expert_gate_up_q4_m8_n128_sg4",
            "moe_expert_down_q4_m8_n256_sg4", 128, 256, threads};
  return {"moe_expert_gate_up_q4_m8", "moe_expert_down_q4_m8", 128, 128,
          threads};
}

void addAffineExperts(metal::CommandGraph &graph, const MoeScratch &scratch,
                      const AffineMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  // The two expert strides are the gate and up slabs of the fused tile; a
  // single-matrix pass reads only the first, so its params repeat one stride.
  const MoeExpertParams gateUp{shape.hiddenSize, shape.expertIntermediateSize,
                               shape.experts, 0,
                               weights.expertGate.expertStrideBytes,
                               weights.expertUp.expertStrideBytes};
  const MoeExpertParams gate{shape.hiddenSize, shape.expertIntermediateSize,
                             shape.experts, 0,
                             weights.expertGate.expertStrideBytes,
                             weights.expertGate.expertStrideBytes};
  const MoeExpertParams up{shape.hiddenSize, shape.expertIntermediateSize,
                           shape.experts, 0, weights.expertUp.expertStrideBytes,
                           weights.expertUp.expertStrideBytes};
  const MoeExpertParams down{shape.expertIntermediateSize, shape.hiddenSize,
                             shape.experts, 0,
                             weights.expertDown.expertStrideBytes,
                             weights.expertDown.expertStrideBytes};
  if (plan.splitExperts()) {
    // The gate lands in expertOutput, which the down pass overwrites only
    // after the up pass has consumed it.
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.sharedGate.packed, scratch.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_up_silu_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertUp.packed,
               weights.sharedUp.packed, scratch.expertOutput,
               scratch.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    // The workspace holds the same grouped rows whatever the column tile;
    // only the grid's column count and the threadgroup width follow it.
    const ExpertPasses passes = fusedExpertPasses(plan.configuration());
    graph.add(passes.gateUp,
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.expertUp.packed, weights.sharedGate.packed,
               weights.sharedUp.packed, scratch.expertIntermediate},
              gateUp,
              {shape.expertIntermediateSize / passes.gateUpColumns, tiles, 1},
              {passes.threads, 1, 1});
    graph.add(passes.down,
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / passes.downColumns, tiles, 1},
              {passes.threads, 1, 1});
  }
}

// Unaligned experts on the 8-row MMA tile run gate and up as one pass that reads the layer's input rows through
// the grouped routes, so no gather precedes it.
bool fusedGateUp(const MoePlan &plan) {
  return plan.configuration().ggufTile == MoeGgufTile::Mma && plan.tileRows() == 8 &&
         plan.shape().expertIntermediateSize % QUANT_TILE_ROWS != 0;
}

// The formats whose tiles of one or two live rows run on registers (kernels/common/gguf_dot_tile.h): gate/up and down
// pass the kernel name suffix, null where the MMA tile runs every tile. The shared tile holds min(rows, 8) rows and
// takes the dot body at two rows or fewer, which reads Q8_0 only.
const char *gateUpDotFormat(const BlockExpertProjection &gate, const BlockExpertProjection &up, uint32_t hidden,
                            uint32_t rows) noexcept {
  if (gate.routed.formatId != up.routed.formatId || hidden > 5120 ||
      (rows <= 2 && (gate.shared.formatId != GGUF_FMT_Q80 || up.shared.formatId != GGUF_FMT_Q80)))
    return nullptr;
  switch (gate.routed.formatId) {
  case GGUF_FMT_IQ2XXS: return "iq2xxs";
  case GGUF_FMT_IQ2XS: return "iq2xs";
  case GGUF_FMT_IQ2S: return "iq2s";
  case GGUF_FMT_IQ3XXS: return "iq3xxs";
  case GGUF_FMT_IQ3S: return "iq3s";
  default: return nullptr;
  }
}
const char *downDotFormat(const BlockExpertProjection &down, uint32_t intermediate) noexcept {
  if (down.shared.formatId != GGUF_FMT_Q80 || intermediate > 1280)
    return nullptr;
  switch (down.routed.formatId) {
  case GGUF_FMT_IQ4NL: return "iq4nl";
  case GGUF_FMT_Q20: return "q20";
  default: return nullptr;
  }
}

// The three GGUF expert passes over grouped tiles: gate into expertOutput
// (the down pass overwrites it after the up pass consumed it), up with
// silu(gate) into expertIntermediate, down into expertOutput. Register plans
// read Table16 tiles from groupedInput: the gather writes the gate/up input's
// and a prepare dispatch the down input's.
void addGgufExperts(metal::CommandGraph &graph, const metal::MetalBuffer &layerInput, const MoeScratch &scratch,
                    const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k, const char *dot = nullptr) {
    std::vector<metal::MetalBuffer> bindings{input};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput});
    const bool mma = plan.configuration().ggufTile == MoeGgufTile::Mma;
    // Experts of a width that is not a whole plane tile start inside one; only
    // the MMA kernels address those (moe_gguf_first_row).
    const bool unaligned = n % QUANT_TILE_ROWS != 0;
    if (unaligned && !mma)
      throw std::invalid_argument("GGUF experts of this width run on the MMA expert tile only");
    const std::string kernel = table16 ? "moe_expert_gguf_sg"
                                       : (mma ? "moe_expert_gguf_mma_m" : "moe_expert_gguf_m") +
                                             std::to_string(plan.tileRows());
    graph.add(kernel + (up ? "_g" : "_a") + (unaligned ? "_u" : "") + (dot ? std::string("_dot_") + dot : ""),
              std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  projection.shared.formatId},
              {n / GGUF_TILE_COLUMNS, tiles, 1},
              {table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (fusedGateUp(plan)) {
    std::vector<metal::MetalBuffer> bindings{layerInput, scratch.groupedRoutes, scratch.tileDescriptors, scratch.tileCount};
    for (const BlockExpertProjection *projection : {&weights.gate, &weights.up})
      bindings.insert(bindings.end(),
                      {projection->routed.plane0, projection->routed.plane1Slot(), projection->routed.meta,
                       projection->shared.plane0, projection->shared.plane1Slot(), projection->shared.meta});
    bindings.push_back(scratch.expertIntermediate);
    const auto params = [&](const BlockExpertProjection &projection) {
      return MoeGgufExpertParams{hidden, intermediate, shape.experts, projection.routed.formatId,
                                 projection.shared.formatId};
    };
    const char *dot = plan.configuration().ggufFewRowDot ? gateUpDotFormat(weights.gate, weights.up, hidden, plan.rows()) : nullptr;
    graph.add(dot ? std::string("moe_expert_gguf_mma_gu_k2_dot_") + dot : "moe_expert_gguf_mma_gu_k2", std::move(bindings),
              MoeGgufGateUpParams{params(weights.gate), params(weights.up), shape.routesPerToken()},
              {intermediate / GGUF_TILE_COLUMNS, tiles, 1}, {2 * GGUF_MOE_SPLIT_THREADS, 1, 1});
  } else {
    pass(weights.gate, false, scratch.groupedInput, scratch.expertOutput,
         intermediate, hidden);
    pass(weights.up, true, scratch.groupedInput, scratch.expertIntermediate,
         intermediate, hidden);
  }
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, (intermediate + 255) / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate,
       plan.configuration().ggufFewRowDot && plan.tileRows() == 8 &&
               hidden % GGUF_TILE_COLUMNS == 0 && hidden % QUANT_TILE_ROWS == 0
           ? downDotFormat(weights.down, intermediate)
           : nullptr);
}

} // namespace

// Prefill plans and GGUF plans run the three expert passes of the split
// plan, affine decode plans the fused gate/up tile.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config),
      splitExperts_(shape.weightLayout == WeightLayout::Block32 ||
                    (phase == MoePhase::Prefill && config.kernel == MoeExpertKernel::Mpp)) {
  // Affine plans have 32-row prefill and 8-row decode kernels, GGUF plans
  // 8-row kernels in both phases and 32-row prefill kernels.
  const bool gguf = shape.weightLayout == WeightLayout::Block32;
  const bool prefill = phase == MoePhase::Prefill;
  if ((!prefill && config.expertTile != MoeExpertTile::M8) ||
      (!gguf && prefill && config.expertTile != MoeExpertTile::M32))
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if ((config.kernel != MoeExpertKernel::Mpp && config.kernel != MoeExpertKernel::Register) ||
      (config.kernel == MoeExpertKernel::Register && gguf))
    throw std::invalid_argument("invalid MoE expert kernel configuration");
  if (config.ggufTile == MoeGgufTile::Register &&
      (shape.weightLayout != WeightLayout::Block32 || config.expertTile != MoeExpertTile::M8))
    throw std::invalid_argument("the register expert tile takes block 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), splitExperts_, compact(), config.ggufTile);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const MoeWeights &weights, const MoePlan &plan,
              std::span<const uint32_t> laneActiveRows) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  const uint32_t tileRows = plan.tileRows();
  validate(weights, shape);
  const uint32_t tiles = plan.maximumTiles();
  const MoeWorkspace &required = plan.workspace();
  const uint64_t rowBytes = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  if (buffers.input.sizeBytes() < rowBytes ||
      buffers.residual.sizeBytes() < rowBytes ||
      buffers.output.sizeBytes() < rowBytes)
    throw std::invalid_argument("MoE row buffers are smaller than execution shape");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    if ((scratch.*field.buffer).sizeBytes() < required.*field.bytes)
      throw std::invalid_argument("MoE grouped scratch is smaller than its bound");
  uint32_t inactiveRows = 0;
  if (!laneActiveRows.empty()) {
    constexpr uint32_t laneRows = SPLASH_TARGET_VERIFY_ROWS;
    if (rows > 32 || rows != laneActiveRows.size() * laneRows)
      throw std::invalid_argument("MoE active rows do not match the decode lanes");
    for (size_t lane = 0; lane < laneActiveRows.size(); ++lane) {
      if (!laneActiveRows[lane] || laneActiveRows[lane] > laneRows)
        throw std::invalid_argument("invalid MoE active rows");
      for (uint32_t row = laneActiveRows[lane]; row < laneRows; ++row)
        inactiveRows |= 1u << (lane * laneRows + row);
    }
  }
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken, inactiveRows};
  const bool block = weights.layout() == WeightLayout::Block32;
  if (block) {
    // fp32 scores of the F32 router in rows of routerWidth() expert slots, as
    // the select kernel reads.
    const uint32_t width = shape.routerWidth();
    addGgufFloat(graph, buffers.input, weights.blocks().router, scratch.groupedInput, rows,
                 width, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
    graph.add(width > 256 ? "moe_route_select_f32_e512" : "moe_route_select_f32",
              {scratch.groupedInput, buffers.input,
               weights.blocks().sharedScalarGate.plane0, scratch.selectedExperts,
               scratch.routingWeights},
              routeParams, {rows, 1, 1}, {width, 1, 1});
  } else {
    const AffineMoeWeights &affine = weights.affine();
    const MoeRouteTile route = moeRouteTile(rows, plan.configuration().routeWideRows);
    graph.add(route.rows == 8 ? "moe_route_scores_q8_m8"
                              : "moe_route_scores_q8_m32",
              {buffers.input, affine.router.planes.weights, affine.router.planes.scales,
               affine.router.planes.biases, scratch.groupedInput},
              routeParams,
              {(rows + route.rows - 1) / route.rows, 256 / route.experts, 1});
    graph.add("moe_route_select_q8",
              {scratch.groupedInput, buffers.input,
               affine.sharedScalarGate.planes.weights,
               affine.sharedScalarGate.planes.scales,
               affine.sharedScalarGate.planes.biases, scratch.selectedExperts,
               scratch.routingWeights},
              routeParams, {rows, 1, 1});
  }
  graph.add(shape.routerWidth() > 256 ? "moe_group_routes_e512" : "moe_group_routes",
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            MoeGroupParams{rows, shape.expertsPerToken, tileRows,
                           shape.experts, plan.compact()},
            {1, 1, 1}, {shape.routerWidth(), 1, 1});
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken(), plan.compact()};
  // The fused gate/up pass reads the input rows through the routes: no gathered copy.
  if (!(block && fusedGateUp(plan))) {
    if (plan.configuration().ggufTile == MoeGgufTile::Register)
      graph.add("moe_gather_table16",
                {buffers.input, scratch.groupedRoutes, scratch.tileCount,
                 scratch.groupedInput, scratch.groupedSums},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else
      graph.add("moe_gather_rows",
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / 256, 1});
  }
  if (block)
    addGgufExperts(graph, buffers.input, scratch, weights.blocks(), plan);
  else
    addAffineExperts(graph, scratch, weights.affine(), plan);
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
}

void MoE::addStats(metal::CommandGraph &graph, const MoeScratch &scratch, const metal::MetalBuffer &stats,
                   const MoeShape &shape, uint32_t layer, uint32_t rows, uint32_t statsRows) {
  const MoeStatsParams params{layer, rows, shape.expertsPerToken, shape.routesPerToken(), statsRows};
  graph.add("moe_stats_record", {scratch.selectedExperts, scratch.routingWeights, stats}, params,
            {(rows * shape.expertsPerToken + 63) / 64, 1, 1}, {64, 1, 1});
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, MoePhase::Prefill);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  return MoePlan(shape, lanes * SPLASH_TARGET_VERIFY_ROWS, config, MoePhase::Decode);
}

} // namespace splash::ops
