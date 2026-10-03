#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
#include "ModelDescriptor.hpp"
#include "Qwen3_6Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenVision.hpp"
#include "VisionLoader.hpp"
#include "ops/PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace splash::model {

class QwenStateStorage;

using TargetWeights = std::variant<Qwen3_8Weights, Qwen3_6MoeWeights>;

struct ModelPackage final {
  ModelDescriptor descriptor;
  TargetWeights target;
  DFlashDraftWeights draft;
  QwenVisionWeights vision;
  std::string manifestFingerprintSha256;

  [[nodiscard]] const std::string &name() const noexcept {
    return descriptor.name;
  }
  [[nodiscard]] kv::Layout targetKvLayout(kv::Format format) const noexcept {
    auto layout = descriptor.targetKvLayout;
    layout.format = format;
    return layout;
  }
  [[nodiscard]] CompositeStateLayout stateLayout() const noexcept {
    return descriptor.stateLayout;
  }
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept {
    return descriptor.capabilities.maximumContextTokens;
  }
  [[nodiscard]] uint64_t targetActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) {
      return weights.actualAllocatedBytes;
    }, target);
  }
  [[nodiscard]] const std::string &targetManifestFingerprint() const noexcept {
    return std::visit([](const auto &weights) -> const std::string & {
      return weights.manifestFingerprintSha256;
    }, target);
  }
  [[nodiscard]] std::span<const WeightFileRecord> targetFiles() const noexcept {
    return std::visit([](const auto &weights) ->
                          std::span<const WeightFileRecord> {
      return weights.files;
    }, target);
  }
};

// Model execution resources, which the engine assembles. What a request's
// start allocates is admitted through the state storage.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  const ModelPackage &package;
  kv::PageStorage &kvPages;
  QwenStateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
  // Packed-prefill row cap the shared prefill arena is sized for.
  uint32_t prefillRows = ExecutionLimits::prefillTokenBudget;
  // Concurrent lanes the shared decode arena is sized for.
  uint32_t decodeLanes = ExecutionLimits::maximumBatchWidth;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

// Where a model package's roles are read from: independent directories, not
// necessarily under one shared parent (a local MLX export and its DFlash2
// draft checkpoint have none). vision stays empty when the descriptor names
// no vision role. A legacy installed package's target/draft/vision are one
// root's subdirectories of the same names (ofRoot).
struct ModelPaths final {
  std::filesystem::path target;
  std::filesystem::path draft;
  std::filesystem::path vision;

  [[nodiscard]] static ModelPaths ofRoot(const std::filesystem::path &root) {
    return {root / "target", root / "draft", root / "vision"};
  }
};

[[nodiscard]] uint64_t preparedModelWeightBytes(const std::filesystem::path &root,
                                                 const ModelDescriptor &descriptor);
[[nodiscard]] uint64_t preparedModelWeightBytes(const ModelPaths &paths,
                                                 const ModelDescriptor &descriptor);

// The vision role's upstream source, planned for preparation; null for a
// packed vision file or a model without vision.
[[nodiscard]] std::unique_ptr<VisionLoader>
planVisionLoader(metal::MetalBackend &backend, const std::filesystem::path &root,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);
[[nodiscard]] std::unique_ptr<VisionLoader>
planVisionLoader(metal::MetalBackend &backend, const ModelPaths &paths,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);
// The vision role: prepared by `loader` when there is one, else the packed
// file; empty weights for a model without vision.
[[nodiscard]] QwenVisionWeights
loadVisionWeights(metal::MetalBackend &backend, const std::filesystem::path &root,
                  const ModelDescriptor &descriptor, const VisionLoader *loader);
[[nodiscard]] QwenVisionWeights
loadVisionWeights(metal::MetalBackend &backend, const ModelPaths &paths,
                  const ModelDescriptor &descriptor, const VisionLoader *loader);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const ModelPaths &paths,
                 const ModelDescriptor &descriptor, PreparationCheck admitConversion);

[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const DeviceCapabilities &device,
                     const ModelPackage &package,
                     const ops::ExecutionPlans &operators,
                     kv::Format format = kv::Format::Int8,
                     uint32_t prefillRows = ExecutionLimits::prefillTokenBudget,
                     uint32_t decodeLanes = ExecutionLimits::maximumBatchWidth);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
