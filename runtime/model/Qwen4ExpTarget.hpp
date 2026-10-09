#pragma once

// Qwen3.8-Flash-Next's graphs (model/Qwen4Exp.hpp): hyper-connection mixes
// around every block, the PLE layer's gated n-gram rows and dilated
// convolution, GDN and QSA full-attention mixers, and the sparse MoE with a
// shared expert. QwenTarget builds the other families' hybrid graph and hands
// this family's calls here.

#include "model/Qwen4Exp.hpp"
#include "model/QwenTarget.hpp"

#include <cstdint>
#include <span>

namespace splash::model {

class Qwen4ExpTarget final {
public:
  Qwen4ExpTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                 const ops::ExecutionPlans &operators);

  // The hidden buffer that holds the final mix of every row.
  [[nodiscard]] metal::MetalBuffer addPrefill(metal::CommandGraph &graph, const QwenTargetPrefillBuffers &buffers,
                                              std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
                                              std::span<const SplashKvLayer> kvLayers) const;
  void addVerify(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                 std::span<const SplashKvLayer> kvLayers, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const;
  // The vocabulary projection of `lanes` lanes of final-mix rows, which it
  // reads as they are: the model has no output norm.
  void addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer logits, uint32_t lanes,
               ops::LinearScratch scratch) const;
  // Each lane's PLE history after its retained rows.
  void addPleCommit(metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers, uint32_t lanes) const;

private:
  struct PrefillStep;
  struct VerifyStep;

  [[nodiscard]] ops::HyperShape hyperShape() const noexcept;
  [[nodiscard]] ops::PleShape pleShape() const noexcept;
  const Qwen4ExpWeights &weights_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model
