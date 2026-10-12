#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <memory>

namespace splash::ane {

// The Neural Engine as a second matmul unit for prefill (opt-in: SPLASH_ANE=1), through the private
// AppleNeuralEngine client the Core ML service uses (the approach of ds4's d50435b). The GPU stages int8 weights
// and fp16 input rows into IOSurfaces and signals the backend's shared event; the engine's evaluation waits on it,
// runs, and signals the next value, which the GPU waits on before it reads the output.

[[nodiscard]] bool requested() noexcept;

// y[rows][width] = x[rows][inputSize] . w[width][inputSize]^T over fp16 rows and int8 weight rows, one function
// per row bucket (512 + 128 k rows, up to maximumRows) over the same surfaces. The weights have two surfaces, one
// per slot, and each slot has the scale of each of its rows (y times scale is the product).
class Program final {
public:
  static constexpr uint32_t kMinimumRows = 512;
  static constexpr uint32_t kRowStep = 128;
  static constexpr uint32_t kMaximumInputSize = 8192;
  static constexpr uint32_t kMaximumWidth = 16384;
  static constexpr uint32_t kSlots = 2;
  ~Program();
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;

  [[nodiscard]] uint32_t inputSize() const noexcept;
  [[nodiscard]] uint32_t width() const noexcept;
  [[nodiscard]] uint32_t maximumRows() const noexcept;
  [[nodiscard]] bool runs(uint32_t rows) const noexcept;
  // Bytes per int8 weight row, halves per input row and per output row.
  [[nodiscard]] uint32_t weightStride() const noexcept;
  [[nodiscard]] uint32_t inputStride() const noexcept;
  [[nodiscard]] uint32_t outputStride() const noexcept;
  [[nodiscard]] const metal::MetalBuffer &input() const noexcept;
  [[nodiscard]] const metal::MetalBuffer &output() const noexcept;
  [[nodiscard]] const metal::MetalBuffer &weights(uint32_t slot) const noexcept;
  [[nodiscard]] const metal::MetalBuffer &scales(uint32_t slot) const noexcept;

  struct Impl;
  explicit Program(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] uint32_t nextSlot() noexcept;
  [[nodiscard]] Impl &state() const noexcept { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

class Runtime final {
public:
  // The process's runtime for `backend`; null unless requested, and after a failure.
  [[nodiscard]] static Runtime *get(metal::MetalBackend &backend);
  // The runtime get created, usable or not.
  [[nodiscard]] static Runtime *existing() noexcept;

  // The program for inputSize x width at up to maximumRows rows (compiled on first use); null once the engine
  // cannot compile it, which stops its use for the process.
  [[nodiscard]] std::shared_ptr<Program> program(uint32_t inputSize, uint32_t width, uint32_t maximumRows);
  // Set to 1 by the unpack when an output row is not finite.
  [[nodiscard]] const metal::MetalBuffer &status() const noexcept;

  // Reserves the event values of one evaluation of `program` over `rows` rows, and the weight slot it reads (they
  // alternate, so the next evaluation's weights can be staged while this one runs): the graph's dispatches signal
  // ready after packing the input and wait for done before the unpack. The evaluation is queued when the graph is
  // submitted (CommandGraph::runHostWork).
  struct Values {
    uint64_t ready;
    uint64_t done;
    uint32_t slot;
  };
  Values evaluate(metal::CommandGraph &graph, const std::shared_ptr<Program> &program, uint32_t rows);

  // After a command that ran evaluations has completed: whether they all finished with finite rows. A failure
  // stops the engine for the process; the caller reruns the command without it.
  [[nodiscard]] bool settle();

  struct Impl;
  explicit Runtime(std::unique_ptr<Impl> impl) noexcept;
  ~Runtime();

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::ane
