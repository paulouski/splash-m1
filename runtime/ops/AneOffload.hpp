#pragma once

#include "ane/Ane.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace splash::ops {

// The trailing output columns [firstColumn, firstColumn + columns) of a block projection that the Neural Engine
// computes in prefill while the GPU computes the leading ones (runtime/ane).
struct AneColumns final {
  uint32_t firstColumn = 0;
  uint32_t columns = 0;
};

class AneOffload final {
public:
  // The share `fraction` of the projection's columns, in whole GGUF_TILE_COLUMNS tiles and at most the engine's
  // widest program; nothing unless the projection is quantized, unrotated and bf16, and its inputs fit a program.
  [[nodiscard]] static std::optional<AneColumns> plan(const Projection &projection, double fraction);

  // As Linear::addPrefill of `rows` rows, with the planned columns on the engine: stage their int8 weights, pack
  // the input, signal; the GPU's columns; wait, unpack. `program` is the engine's for the projection's inputs and
  // the planned columns, and runs `rows`.
  static void addPrefill(metal::CommandGraph &graph, const Linear &linear, ane::Runtime &engine,
                         const std::shared_ptr<ane::Program> &program, metal::MetalBuffer input,
                         const Projection &projection, const AneColumns &offload, metal::MetalBuffer output,
                         metal::MetalBuffer sums, uint32_t rows, LinearScratch scratch);
};

} // namespace splash::ops
