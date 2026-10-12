#include "ops/AneOffload.hpp"

#include "metal/abi/Ane.h"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kStageThreads = 256;
constexpr uint32_t kPackThreadsX = 32;
constexpr uint32_t kPackThreadsY = 8;

// The columns of the quantized segments, which lead the float ones (a float segment stays with the GPU).
uint32_t quantizedColumns(const Projection &projection) {
  uint32_t columns = 0;
  for (const QuantizedSegment &s : projection.blocks().segments) {
    if (s.isFloat()) break;
    columns += s.outputSize;
  }
  return columns;
}

} // namespace

std::optional<AneColumns> AneOffload::plan(const Projection &projection, double fraction) {
  if (projection.layout() != WeightLayout::Block32 || projection.rotation ||
      projection.destination != FloatOutput::BFloat16 || projection.inputSize % 32 ||
      projection.inputSize > ane::Program::kMaximumInputSize)
    return std::nullopt;
  // A quantized segment after a float one would need the float path between its columns.
  bool floating = false;
  for (const QuantizedSegment &s : projection.blocks().segments) {
    if (s.isFloat()) floating = true;
    else if (floating) return std::nullopt;
  }
  const uint32_t covered = quantizedColumns(projection);
  const uint32_t tiles = std::min(static_cast<uint32_t>(std::floor(fraction * covered / GGUF_TILE_COLUMNS)),
                                  ane::Program::kMaximumWidth / GGUF_TILE_COLUMNS);
  const uint32_t columns = tiles * GGUF_TILE_COLUMNS;
  if (!columns || columns >= covered) return std::nullopt;
  return AneColumns{covered - columns, columns};
}

void AneOffload::addPrefill(metal::CommandGraph &graph, const Linear &linear, ane::Runtime &engine,
                            const std::shared_ptr<ane::Program> &program, metal::MetalBuffer input,
                            const Projection &projection, const AneColumns &offload, metal::MetalBuffer output,
                            metal::MetalBuffer sums, uint32_t rows, LinearScratch scratch) {
  const uint32_t k = projection.inputSize;
  if (!program->runs(rows) || program->inputSize() != k || program->width() != offload.columns ||
      offload.firstColumn + offload.columns != quantizedColumns(projection))
    throw std::invalid_argument("invalid Neural Engine projection");
  const ane::Runtime::Values values = engine.evaluate(graph, program, rows);
  for (const QuantizedSegment &s : projection.blocks().segments) {
    const uint32_t end = s.columnOffset + s.outputSize;
    if (s.isFloat()) break;
    if (end <= offload.firstColumn) continue;
    const uint32_t first = std::max(offload.firstColumn, s.columnOffset);
    graph.add("ane_stage",
              {s.plane0, s.plane1Slot(), s.meta, program->weights(values.slot), program->scales(values.slot)},
              AneStageParams{k, s.formatId, first - s.columnOffset, end - first, first - offload.firstColumn,
                             program->weightStride()},
              {end - first, 1, 1}, {kStageThreads, 1, 1});
  }
  graph.add("ane_pack_bf16", {input, program->input()}, AnePackParams{rows, k, program->inputStride(), 0, 0},
            {(k + kPackThreadsX - 1) / kPackThreadsX, (rows + kPackThreadsY - 1) / kPackThreadsY, 1},
            {kPackThreadsX, kPackThreadsY, 1});
  graph.signalAfterLast(values.ready);
  linear.addPrefill(graph, input, projection, output, std::move(sums), rows, std::move(scratch), offload.firstColumn);
  graph.waitBeforeNext(values.done);
  graph.add("ane_unpack_bf16", {program->output(), program->scales(values.slot), output, engine.status()},
            AnePackParams{rows, offload.columns, program->outputStride(), projection.outputSize, offload.firstColumn},
            {(offload.columns + kPackThreadsX - 1) / kPackThreadsX, (rows + kPackThreadsY - 1) / kPackThreadsY, 1},
            {kPackThreadsX, kPackThreadsY, 1});
}

} // namespace splash::ops
