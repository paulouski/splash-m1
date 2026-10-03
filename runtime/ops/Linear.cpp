#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kAffinePrefillTileRows = 32;
constexpr uint32_t kQuantGroup = 64;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "simdgroup Q4 tiles require eight verify rows per lane");
// The kernels sum their input per block of four quant groups (256 inputs);
// Split128 partitions K in whole blocks.
constexpr uint32_t kInputSumBlock = 4 * kQuantGroup;

bool simdgroupTile(LinearTile tile) noexcept {
  return tile == LinearTile::Simdgroup || tile == LinearTile::SimdgroupF32;
}
bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256;
}
bool q4SimdgroupF32HalfTable(LinearWorkload w, LinearConfig config) noexcept {
  return w.weightLayout == WeightLayout::Affine64 && w.bits == 4 &&
      w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
      w.matrix.inputSize == 5120 && config.tile == LinearTile::SimdgroupF32 &&
      ((w.epilogue == LinearEpilogue::GateUp && w.matrix.outputSize == 17408) ||
       (w.epilogue == LinearEpilogue::None && w.matrix.outputSize == 16640));
}
// The decode tiles whose threadgroups stream several column tiles.
bool persistentTile(LinearTile tile) noexcept {
  return tile == LinearTile::N128 || tile == LinearTile::N256 || tile == LinearTile::Paired128 ||
         tile == LinearTile::Paired256;
}
// The tiles of block (GGUF) projections.
bool blockTile(LinearTile tile) noexcept {
  return tile == LinearTile::GgufStaged || tile == LinearTile::GgufPrefill || tile == LinearTile::GgufRegister;
}
// Simdgroups fixed by the kernel instance: the paired N256 tile runs four,
// Split128 the N128 tile's eight, and the Apple7/8 register tiles four.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Simdgroup:
  case LinearTile::SimdgroupF32:
  case LinearTile::Mma64:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split128: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (w.bits != 4 && w.bits != 5) throw std::invalid_argument("invalid linear bits");
  if (w.bits == 5 && w.weightLayout != WeightLayout::Affine64)
    throw std::invalid_argument("Q5 requires the affine weight layout");
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid linear decode workload");
  }
}

// A buffer a plan does not use needs no bytes and may be absent.
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument(std::string("projection ") + what + " buffer holds " +
                                std::to_string(buffer.sizeBytes()) + " bytes, needs " + std::to_string(bytes));
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, all matrix row tiles, and the one-lane
// Paired256 (plain) tile.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (simdgroupTile(tile)) return w.phase == LinearPhase::Decode;
  if (tile == LinearTile::Mma64) return w.phase == LinearPhase::Prefill;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  if (tile == LinearTile::Paired256)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

} // namespace

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / SPLASH_TARGET_VERIFY_ROWS * table16_sums_per_tile(width) * sizeof(float)
       : layout == LinearInput::Table64 || layout == LinearInput::Table64Half
             ? uint64_t{width} * rows / 16 : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % SPLASH_TARGET_VERIFY_ROWS || width % 64 ||
      scratch.input.sizeBytes() < tableBytes(width, rows) ||
      scratch.sums.sizeBytes() < tableSumsBytes(layout, width, rows))
    throw std::invalid_argument("linear table scratch is below requirement");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table16 ? "_table16" : layout == LinearInput::Table64 ? "_table64"
       : layout == LinearInput::Table64Half ? "_table64_halftable" : "";
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  requireBytes(p.affine().weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2, "weight");
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.affine().scales, bytes, "scale");
  requireBytes(p.affine().biases, bytes, "bias");
  // The fifth bit, one per weight (Q5Pack.hpp): a quarter of the nibble plane's bytes.
  if (p.bits == 5) requireBytes(p.affine().hi, uint64_t{matrix.outputSize} * matrix.inputSize / 8, "hi");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockStorageRows();
  if (workload_.phase != LinearPhase::Prefill) return workload_.rows;
  return ((workload_.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows) * kAffinePrefillTileRows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup:
  case LinearTile::SimdgroupF32: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::Mma64:
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return GGUF_TILE_COLUMNS;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::groups() const noexcept {
  return config_.groups ? config_.groups : workload_.matrix.outputSize / tileColumns();
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  switch (config_.tile) {
  case LinearTile::GgufStaged: return GGUF_STAGED_THREADS;
  case LinearTile::GgufPrefill: return GGUF_PREFILL_THREADS;
  case LinearTile::GgufRegister: return GGUF_REGISTER_THREADS;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::Paired256:
  case LinearTile::Simdgroup:
  case LinearTile::SimdgroupF32:
  case LinearTile::Mma64: return static_cast<uint32_t>(config_.simdgroups) * 32;
  }
  return 0;
}
bool LinearPlan::usesSimdgroup() const noexcept { return simdgroupTile(config_.tile); }
bool LinearPlan::registerMatrix() const noexcept {
  return usesSimdgroup() || config_.tile == LinearTile::Mma64;
}
LinearInput LinearPlan::input() const noexcept {
  if (rotated_) return LinearInput::Plain;
  if (q4SimdgroupF32HalfTable(workload_, config_)) return LinearInput::Table64Half;
  if (config_.tile == LinearTile::GgufRegister) return LinearInput::Table16;
  return usesSimdgroup() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockScratchSize();
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  if (!usesSimdgroup()) return {};
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile. Single-partition kernels use neither.
  const LinearInput input = this->input();
  return {tableBytes(input, k, workload_.rows), tableSumsBytes(input, k, workload_.rows),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : sizeof(float),
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : sizeof(uint32_t)};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // GGUF tiles run gate/up as a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp &&
       (!secondPipeline_.empty() || workload_.weightLayout == WeightLayout::Block32));
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  const bool ggufTile = blockTile(config.tile);
  if (ggufTile != (w.weightLayout == WeightLayout::Block32))
    throw std::invalid_argument("block projections run the GGUF tiles, affine ones the Q4 tiles");
  if (w.phase == LinearPhase::Decode && persistentTile(config.tile)
          ? !config.groups || config.groups > w.matrix.outputSize / tileColumns()
          : config.groups != 0)
    throw std::invalid_argument("a persistent decode tile takes 1 to its column tiles in groups, every other plan 0");
  if (ggufTile) {
    // Kernel names follow the segment formats (LinearGguf.cpp).
    requireBlockConfiguration();
    return;
  }
  const bool splitsK = simdgroupTile(config.tile) || config.tile == LinearTile::Split128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the simdgroup or Split128 Q4 tile");
  if (config.simdgroups == LinearSimdgroups::Four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  if (w.bits == 5 && config.tile != LinearTile::SimdgroupF32 && config.tile != LinearTile::Mma64)
    throw std::invalid_argument("Q5 requires the SimdgroupF32 or Mma64 tile");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (oneLaneTile(config.tile) || splitsK)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (config.tile == LinearTile::Mma64 && w.bits == 5) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q5_mma64_up_silu_sums"
          : residual ? "prefill_linear_q5_mma64_residual" : "prefill_linear_q5_mma64";
    } else if (config.tile == LinearTile::Mma64) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_mma64_up_silu_sums"
          : residual ? "prefill_linear_q4_mma64_residual" : "prefill_linear_q4_mma64";
    } else if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  if (config.tile == LinearTile::Mma64)
    throw std::invalid_argument("the Mma64 tile is prefill-only");
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (oneLaneTile(config.tile) && lane != 0)
    throw std::invalid_argument("paired Q4 tile requires one lane");
  if (usesSimdgroup()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (!config.validSplits() || groups % config.splits)
      throw std::invalid_argument("simdgroup Q4 requires whole power-of-two K partitions");
    if (config.tile == LinearTile::SimdgroupF32 && w.bits == 5) {
      // One threadgroup covers every lane, so each batch width has its kernel.
      static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 4);
      constexpr std::array gateUp{"decode_linear_q5_sgf_gate_up", "decode_linear_q5_sgf_gate_up_m16",
          "decode_linear_q5_sgf_gate_up_m24", "decode_linear_q5_sgf_gate_up_m32"};
      constexpr std::array affine{"decode_linear_q5_sgf", "decode_linear_q5_sgf_m16",
          "decode_linear_q5_sgf_m24", "decode_linear_q5_sgf_m32"};
      constexpr std::array withResidual{"decode_linear_q5_sgf_residual",
          "decode_linear_q5_sgf_residual_m16", "decode_linear_q5_sgf_residual_m24",
          "decode_linear_q5_sgf_residual_m32"};
      pipeline_ = (w.epilogue == LinearEpilogue::GateUp ? gateUp
                   : residual ? withResidual : affine)[lane];
      return;
    }
    if (config.tile == LinearTile::SimdgroupF32) {
      // One threadgroup covers every lane, so each batch width has its kernel.
      static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 4);
      if (q4SimdgroupF32HalfTable(w, config)) {
        pipeline_ = w.epilogue == LinearEpilogue::GateUp
            ? "decode_linear_q4_sgf_halftable_gate_up" : "decode_linear_q4_sgf_halftable";
        return;
      }
      constexpr std::array gateUp{"decode_linear_q4_sgf_gate_up", "decode_linear_q4_sgf_gate_up_m16",
          "decode_linear_q4_sgf_gate_up_m24", "decode_linear_q4_sgf_gate_up_m32"};
      constexpr std::array affine{"decode_linear_q4_sgf", "decode_linear_q4_sgf_m16",
          "decode_linear_q4_sgf_m24", "decode_linear_q4_sgf_m32"};
      constexpr std::array withResidual{"decode_linear_q4_sgf_residual",
          "decode_linear_q4_sgf_residual_m16", "decode_linear_q4_sgf_residual_m24",
          "decode_linear_q4_sgf_residual_m32"};
      pipeline_ = (w.epilogue == LinearEpilogue::GateUp ? gateUp
                   : residual ? withResidual : affine)[lane];
      return;
    }
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (config.tile == LinearTile::Split128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (!config.validSplits() || config.splits < 2 ||
        w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("Split128 requires 2, 4 or 8 K partitions of 256-input blocks");
    constexpr std::array plainNames{"decode_linear_q4_n128_split", "decode_linear_q4_n128_split_m16",
        "decode_linear_q4_n128_split_m24", "decode_linear_q4_n128_split_m32"};
    constexpr std::array residualNames{"decode_linear_q4_n128_split_residual",
        "decode_linear_q4_n128_split_residual_m16", "decode_linear_q4_n128_split_residual_m24",
        "decode_linear_q4_n128_split_residual_m32"};
    constexpr std::array upSiluNames{"decode_linear_q4_n128_split_up_silu",
        "decode_linear_q4_n128_split_up_silu_m16", "decode_linear_q4_n128_split_up_silu_m24",
        "decode_linear_q4_n128_split_up_silu_m32"};
    // Gate/up at every lane count: a plain gate pass into the gate scratch,
    // then the up pass whose epilogue applies the SiLU gate.
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    if (w.epilogue == LinearEpilogue::GateUp) secondPipeline_ = upSiluNames[lane];
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (four) {
    pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4"
                         : "decode_linear_q4_n128_m24_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up uses the conservative limit shared by both devices. Its many-wave
// threshold follows N256; the four-simdgroup threshold scales from N128. Those
// two extrapolations remain unmeasured.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Paired-A/B tuning (tune-kernels) and the per-shape microprofile
// (benchmark-prefill) on a 32-core Apple9 GPU (M4 Max) measured the
// four-simdgroup N128 tile ahead of N256 on every prefill shape and probed
// row count: +6..10% GPU wherever the margin cleared the tuning threshold,
// never behind. Apple9 GPUs at or below that measured core count therefore
// share the Apple10 prefill rule. Larger Apple9 GPUs (40-core class) keep the
// wide-tile rule below; it was sized for them and remains unremeasured there.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;

// Apple10 and later decode split K across the threadgroups of the Split128
// tile (256 threads) by one rule at every batch width: the largest power of
// two up to LinearConfig::kMaximumSplits whose split grid still fits four
// threadgroups per core (1024 threads, twice the 512-thread occupancy knee),
// with at least one 256-input block per partition. A grid of more than two
// tiles per core keeps one split, the sequential tiles. Measured DRAM-cold on
// a 20-core M5 Pro over the 27B and 35B MLX 4-bit decode projections and their
// drafts at one to four lanes, with 10 to 80 cores emulated by width, and on a
// 12-core M6 (Apple11): grids that only reach the knee leave time (the 20
// tiles of 2560 x 4096 take 0.48-0.60 of the sequential time with four
// splits, 0.60-0.71 with two), and a grid past four per core loses to its
// second wave (6144 x 5120 in two splits is 9% slower at one lane). The rule is
// never slower than the sequential tiles on the M5 Pro or on the M6's own 12
// cores (20 cores emulated on the M6 ran 5120 x 4096 4% slower at one lane).
constexpr uint32_t kSplitGroupsPerCore = 4;

uint32_t apple10Splits(LinearMatrix matrix, uint32_t cores) noexcept {
  const uint64_t grid = matrix.outputSize / 128;
  uint32_t splits = 1;
  while (splits < LinearConfig::kMaximumSplits && grid * 2 * splits <= uint64_t{kSplitGroupsPerCore} * cores &&
         2 * splits <= matrix.inputSize / kInputSumBlock)
    splits *= 2;
  return splits;
}

// Apple10 wide plain projections reduce input re-reads with paired N256
// tiles at one resident wave, measured on 16/20-core GPUs. Apple9's simdgroup
// policy is independent.
constexpr uint32_t kPaired256TilesPerCore = 8;
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  if (w.epilogue == LinearEpilogue::None && tiles256 >= kPaired256TilesPerCore * cores)
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, kPaired256WaveGroupsPerCore * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

} // namespace

Linear::Linear(const DeviceCapabilities &device) noexcept
    : appleGpuFamily_(device.appleGpuFamily),
      gpuCores_(plannedGpuCores(device)) {}

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  return plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, LinearEpilogue::None, shape.layout,
               shape.bits})
      .storageRows();
}

// GPU family selects variants; core count and workload tile counts determine
// parallelism.
LinearConfig Linear::baseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  validate(w);
  if (w.weightLayout == WeightLayout::Block32) return ggufBaseline(w, projections);
  // Q5 only runs the Apple7/8 register-matrix tiles (SimdgroupF32, Mma64):
  // never silently fall back to a Q4 tile on a GPU family without them.
  if (w.bits == 5 && appleGpuFamily_ >= 9)
    throw std::invalid_argument("Q5 unsupported on this GPU family");
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    // Apple7/8 (M1/M2): the register-matrix tile measured 2.2-3.2x faster
    // than every MPP tile on an M1 Max, for all epilogues and row counts.
    if (appleGpuFamily_ < 9) return {LinearTile::Mma64, 0, LinearSimdgroups::Four};
    if (appleGpuFamily_ >= 10 || gpuCores_ <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * gpuCores_;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  // Keep the existing broad-column plain projection path for wider batches:
  // independent row tiles repeat its weight stream. Reuse the existing
  // two-N256-tiles-per-core boundary rather than model-specific dimensions.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * gpuCores_;
  // Apple7/8 (M1/M2) run the register-matrix tile with exact half weights and
  // fp32 activations for every decode projection, wide batches included: on an
  // M1 Max it measured 44-68% faster than every MPP tile, with the same K
  // partitions as Apple9. Covering all lanes in one threadgroup, it shortened
  // two- to four-lane decode cycles a further 1.20-1.28x.
  if (appleGpuFamily_ < 9 || (appleGpuFamily_ == 9 && !widePlain)) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < LinearConfig::kMaximumSplits && uint64_t(grid) * splits < 16ULL * gpuCores_ &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    // The fused gdn input (N 16640, K 5120) measured 3-4% faster with four partitions.
    if (appleGpuFamily_ < 9 && w.matrix.outputSize == 16640 && w.matrix.inputSize == 5120 && groups % 8 == 0)
      splits = 4;
    return {appleGpuFamily_ == 9 ? LinearTile::Simdgroup : LinearTile::SimdgroupF32, 0,
            LinearSimdgroups::Four, splits};
  }
  if (appleGpuFamily_ >= 10) {
    if (const uint32_t splits = apple10Splits(w.matrix, gpuCores_); splits > 1)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
    if (lanes == 1)
      if (const auto config = apple10OneLaneConfig(w, gpuCores_)) return *config;
  }
  // Apple9 reaches here only for wide plain projections of three or four
  // lanes, which keep their one-tile grids: the round-robin policy above was
  // measured on Apple10.
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return appleGpuFamily_ >= 10 ? decodeGroups(tiles, gpuCores_, policy)
                                 : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp)
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // Every M24 projection that gets here runs four SIMD groups.
  if (lanes == 3)
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (widePlain) return {LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  return LinearPlan(workload, baseline(workload));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.weightLayout = p.layout();
  if (w.weightLayout == WeightLayout::Affine64) {
    if (gate && gate->bits != p.bits) throw std::invalid_argument("gate and up projections have different bits");
    w.bits = p.bits;
  }
  const std::array<const Projection *, 2> projections{&p, gate};
  LinearPlan plan(w, baseline(w, projections), p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  return plan;
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue) const {
  return plan({{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue}, p);
}

LinearScratchSize Linear::decodeScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
      LinearWorkload w = decode({shape.outputSize, shape.inputSize}, lanes, epilogue);
      w.weightLayout = shape.layout;
      w.bits = shape.bits;
      bound.include(shape.layout == WeightLayout::Block32 ? ggufDecodeScratchSize(w) : plan(w).scratchSize());
    }
  // Decode plans store at most every lane's rows.
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, kMaximumDecodeTileRows);
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // Prefill plans take split scratch only in chunks of up to a decode batch,
  // which a GGUF projection runs on the staged tile (LinearGguf.cpp).
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout,
                          shape.bits})
                        .scratchSize());
  // Every prefill plan stores at most the token budget.
  static_assert(SPLASH_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, SPLASH_PREFILL_TOKEN_BUDGET);
  return bound;
}


PreparedInput Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Projection &p, const LinearPlan &selected, const Projection *gate) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if (p.layout() != w.weightLayout || (gate && gate->layout() != w.weightLayout))
    throw std::invalid_argument("projection layout does not match execution plan");
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  if (w.weightLayout == WeightLayout::Affine64 && (p.bits != w.bits || (gate && gate->bits != w.bits)))
    throw std::invalid_argument("projection bits do not match execution plan");
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "residual");
  requireBytes(b.sums, selected.sumsBytes(), "sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "partials");
  requireBytes(b.scratch.counters, scratch.counters, "counters");
  if (p.layout() == WeightLayout::Block32) {
    if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "rotated input");
    addGguf(graph, b, p, selected, gate);
    // A rotated projection's plan prepares its table, if any, from the
    // rotated rows, which no other plan reads.
    if (p.rotation) return {};
    // Only quantized segments run the plan's tile: float segments alone
    // leave the scratch table as it was.
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    const bool tiled =
        std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
    return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
  }
  requireAffineProjection(p, w.matrix);
  if (gate) requireAffineProjection(*gate, w.matrix);
  const AffineWeights &weights = p.affine();
  if (selected.usesSimdgroup()) {
    const LinearInput input = selected.input();
    if (b.prepared.layout != input || !b.prepared.source.sameView(b.input))
      graph.add(input == LinearInput::Table64Half ? "decode_linear_q4_prepare_halftable" : "decode_linear_q4_prepare",
                {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / SPLASH_TARGET_VERIFY_ROWS, 1}, {128, 1, 1});
    const AffineWeights &first = gate ? gate->affine() : weights;
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights};
    if (w.bits == 5) bindings.push_back(first.hi);
    bindings.insert(bindings.end(), {first.scales, first.biases, b.output, b.scratch.sums, b.scratch.partials,
                                     b.scratch.counters});
    if (gate) {
      bindings.push_back(weights.weights);
      if (w.bits == 5) bindings.push_back(weights.hi);
      bindings.insert(bindings.end(), {weights.scales, weights.biases});
    } else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    // The bfloat tile runs a threadgroup per lane; the fp32 tile covers every
    // lane in one threadgroup. The sgf decode tiles take their K-split count
    // (and Q5's hi-tile range, the union of the gate/up streams') as groups.
    const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
    const bool f32 = selected.configuration().tile == LinearTile::SimdgroupF32;
    if (f32) {
      Q4PersistentParams params{n, k, selected.configuration().splits, weights.hiTileBegin, weights.hiTileEnd};
      if (gate) {
        params.hi_tile_begin = std::min(first.hiTileBegin, weights.hiTileBegin);
        params.hi_tile_end = std::max(first.hiTileEnd, weights.hiTileEnd);
      }
      graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings), params,
          {selected.groups(), selected.configuration().splits, 1}, {128, 1, 1});
    } else {
      graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings),
          Q4Params{n, k}, {selected.groups(), selected.configuration().splits, lanes}, {128, 1, 1});
    }
    return {b.input, input};
  }
  const auto dispatch = [&](std::string_view name,
      std::vector<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill) {
      const uint32_t tailRows = w.rows % kAffinePrefillTileRows;
      const bool shortMmaTail = selected.configuration().tile == LinearTile::Mma64 &&
          tailRows >= 1 && tailRows <= 16 && w.epilogue != LinearEpilogue::GateUp;
      const Q4Params params{n, k, weights.hiTileBegin, weights.hiTileEnd};
      if (shortMmaTail) {
        const uint32_t fullTiles = w.rows / kAffinePrefillTileRows;
        if (fullTiles)
          graph.add(std::string(name), bindings, params,
              {fullTiles, n / selected.tileColumns(), 1},
              {selected.threadsPerThreadgroup(), 1, 1});
        const uint32_t activeTailRows = tailRows <= 8 ? 8 : 16;
        std::string tailName(name);
        tailName += activeTailRows == 8 ? "_m8" : "_m16";
        graph.add(std::move(tailName), std::move(bindings),
            Q4PrefillTailParams{params, fullTiles},
            {1, n / selected.tileColumns(), 1},
            {activeTailRows == 8 ? 64u : 128u, 1, 1});
      } else {
        graph.add(std::string(name), bindings, params,
            {selected.storageRows() / kAffinePrefillTileRows, n / selected.tileColumns(), 1},
            {selected.threadsPerThreadgroup(), 1, 1});
      }
    } else {
      // Split128 binds its partials and counters after the sequential
      // kernel's buffers; every other decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      if (config.tile == LinearTile::Split128)
        bindings.insert(bindings.end(), {b.scratch.partials, b.scratch.counters});
      const std::string kernel = kernelInstance(name, selected.destination());
      const metal::DispatchSize groups{selected.groups(), config.splits, 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      // The persistent tiles stride over the column tiles by their groups.
      if (persistentTile(config.tile))
        graph.add(kernel, std::move(bindings), Q4PersistentParams{n, k, selected.groups()}, groups, threads);
      else
        graph.add(kernel, std::move(bindings), Q4Params{n, k}, groups, threads);
    }
  };
  const bool prefill = w.phase == LinearPhase::Prefill;
  // The Q5 weight triple (weights, hi, scales, biases): only Mma64 prefill
  // reaches here with bits==5 (SimdgroupF32 decode binds above instead).
  const auto qw = [&](const AffineWeights &aw) {
    std::vector<metal::MetalBuffer> v{aw.weights};
    if (w.bits == 5) v.push_back(aw.hi);
    v.insert(v.end(), {aw.scales, aw.biases});
    return v;
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    const AffineWeights &g = gate->affine();
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases,
                                     b.output, weights.weights, weights.scales, weights.biases});
    else {
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases, b.gateScratch});
      dispatch(selected.secondPipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate) {
    std::vector<metal::MetalBuffer> bindings{b.input};
    auto qwv = qw(weights);
    bindings.insert(bindings.end(), qwv.begin(), qwv.end());
    bindings.insert(bindings.end(), {b.gateScratch, b.output, b.sums, b.downSums});
    dispatch(selected.pipeline(), std::move(bindings));
  } else if (w.epilogue == LinearEpilogue::Residual) {
    std::vector<metal::MetalBuffer> bindings{b.input};
    auto qwv = qw(weights);
    bindings.insert(bindings.end(), qwv.begin(), qwv.end());
    if (prefill) bindings.insert(bindings.end(), {b.residual, b.output, b.sums});
    else bindings.insert(bindings.end(), {b.residual, b.output});
    dispatch(selected.pipeline(), std::move(bindings));
  } else if (prefill) {
    std::vector<metal::MetalBuffer> bindings{b.input};
    auto qwv = qw(weights);
    bindings.insert(bindings.end(), qwv.begin(), qwv.end());
    bindings.insert(bindings.end(), {b.output, b.sums});
    dispatch(selected.pipeline(), std::move(bindings));
  } else {
    std::vector<metal::MetalBuffer> bindings{b.input};
    auto qwv = qw(weights);
    bindings.insert(bindings.end(), qwv.begin(), qwv.end());
    bindings.push_back(b.output);
    dispatch(selected.pipeline(), std::move(bindings));
  }
  return b.prepared;
}

void Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                            const Projection &consumer, uint32_t rows) const {
  validate({{consumer.outputSize, consumer.inputSize}, rows, LinearPhase::Prefill});
  const uint32_t tiles = (rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
  const uint64_t storageRows = uint64_t{tiles} * kAffinePrefillTileRows;
  requireBytes(input, storageRows * consumer.inputSize * 2, "input");
  requireBytes(sums, storageRows * (consumer.inputSize / kQuantGroup) * 4, "sums");
  graph.add("prefill_linear_q4_sums32", {input, sums}, consumer.inputSize, {tiles, 1, 1});
}
void Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                        metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                        LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::None));
}
void Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                                uint32_t rows, LinearScratch scratch) const {
  add(graph, {.input = input, .output = output, .sums = sums, .residual = residual, .scratch = scratch}, p,
      prefillPlan(p, rows, LinearEpilogue::Residual));
}
void Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                                  metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                  metal::MetalBuffer sums, metal::MetalBuffer downSums, uint32_t rows,
                                  LinearScratch scratch) const {
  add(graph,
      {.input = input, .output = output, .sums = sums, .gateScratch = gateScratch, .downSums = downSums,
       .scratch = scratch},
      up, prefillPlan(up, rows, LinearEpilogue::UpWithGate));
}

} // namespace splash::ops
