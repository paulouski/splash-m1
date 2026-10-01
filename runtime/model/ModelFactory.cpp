#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  if (!package.descriptor.valid() ||
      package.descriptor.draft != package.draft.layout ||
      !std::visit(
          [&](const auto &target) {
            return package.descriptor.target == TargetLayout{target.layout} &&
                   target.layout.vocabularySize ==
                       package.draft.layout.vocabularySize;
          },
          package.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles<Qwen3_8Layout> &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const ModelPaths &paths,
                         ModelDescriptor descriptor, PreparationCheck admitConversion = {}) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  const PreparationCheck check = [&backend] { backend.checkOperation(); };
  // Every prepared file of the model, the vision tower's, the draft's and the
  // target's, is planned before the first is written, so one disk check
  // budgets them all.
  const auto vision = planVisionLoader(backend, paths, result.descriptor, admitConversion);
  std::vector<PreparedWeight> prepared;
  if (vision) prepared.push_back(vision->weight());
  std::optional<DraftCheckpointLoader> draft;
  if (result.descriptor.draftSource == DraftSource::Checkpoint) {
    draft.emplace(backend, paths.draft, result.descriptor.draft, admitConversion);
    prepared.insert(prepared.end(), draft->weights().begin(), draft->weights().end());
  }
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path &directory = paths.target;
        // Every missing file is written before the first is mapped: conversion
        // needs normal memory pressure, which the residency of the files
        // mapped before it would take away on a Mac with little memory.
        const auto read = [&](const QwenTargetFiles<Layout> &files,
                              std::span<const PreparedWeight> target, const auto &prepareTarget) {
          prepared.insert(prepared.end(), target.begin(), target.end());
          if (!prepared.empty()) PreparedWeights().requireSpace(prepared, check);
          if (vision) static_cast<void>(vision->prepare());
          if (draft) draft->prepare();
          prepareTarget();
          return readTarget(backend, layout, files);
        };
        switch (result.descriptor.targetSource) {
        case TargetSource::Packed:
          return read(PackedTargetFiles<Layout>{backend, directory, layout}, {}, [] {});
        case TargetSource::Mlx: {
          AffineTargetLoader loader(backend, directory, layout, admitConversion);
          return read(loader, loader.weights(), [&] { loader.prepare(); });
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, findTargetGguf(directory), ggufTargetGeometry(layout),
                                  admitConversion);
          return read(loader, loader.weights(), [&] { loader.prepare(); });
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  result.draft = loadDFlashDraftWeights(
      backend,
      draft ? DraftFiles(std::ref(*draft))
            : DraftFiles(PackedDraftFiles{backend, paths.draft, result.descriptor.draft}),
      result.descriptor.draft);
  result.vision = loadVisionWeights(backend, paths, result.descriptor, vision.get());

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root) {
  return loadPackage(backend, ModelPaths::ofRoot(root), inspectModelPackage(root));
}

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor, PreparationCheck admitConversion) {
  return loadPackage(backend, ModelPaths::ofRoot(root), descriptor, std::move(admitConversion));
}

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const ModelPaths &paths,
                              const ModelDescriptor &descriptor, PreparationCheck admitConversion) {
  return loadPackage(backend, paths, descriptor, std::move(admitConversion));
}

std::unique_ptr<VisionLoader> planVisionLoader(metal::MetalBackend &backend, const ModelPaths &paths,
                                               const ModelDescriptor &descriptor,
                                               PreparationCheck admitConversion) {
  if (descriptor.visionSource != VisionSource::Mlx && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(paths.vision, descriptor.visionSource, descriptor.vision,
                                        [&backend] { backend.checkOperation(); },
                                        std::move(admitConversion));
}

std::unique_ptr<VisionLoader> planVisionLoader(metal::MetalBackend &backend, const std::filesystem::path &root,
                                               const ModelDescriptor &descriptor,
                                               PreparationCheck admitConversion) {
  return planVisionLoader(backend, ModelPaths::ofRoot(root), descriptor, std::move(admitConversion));
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, const ModelPaths &paths,
                                    const ModelDescriptor &descriptor, const VisionLoader *loader) {
  if (loader) return loadQwenVisionWeights(backend, *loader);
  if (descriptor.visionSource == VisionSource::Packed)
    return loadQwenVisionWeights(backend, paths.vision, descriptor.vision);
  return {};
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, const std::filesystem::path &root,
                                    const ModelDescriptor &descriptor, const VisionLoader *loader) {
  return loadVisionWeights(backend, ModelPaths::ofRoot(root), descriptor, loader);
}

uint64_t preparedModelWeightBytes(const ModelPaths &paths, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(paths.target));
    const GgufFile file(source);
    for (const gguf::Image &image :
         std::visit([&](const auto &layout) { return gguf::planImages(file, ggufTargetGeometry(layout)); },
                    descriptor.target))
      bytes += image.bytes;
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    // The checkpoint's real per-tensor bits (a Q5 override adds 8 B per 64
    // values over the default-4-bit estimate): metadata only, no backend.
    const SafetensorsCheckpoint source(paths.target);
    bytes = std::visit([&source](const auto &layout) { return preparedAffineBytes(layout, source); },
                       descriptor.target);
  }
  if (descriptor.draftSource == DraftSource::Checkpoint) bytes += preparedDraftBytes(descriptor.draft);
  if (descriptor.visionSource == VisionSource::Mlx || descriptor.visionSource == VisionSource::Gguf)
    bytes += preparedVisionBytes(descriptor.vision);
  for (const auto &[directory, packed] :
       {std::pair{&paths.target, descriptor.targetSource == TargetSource::Packed},
        std::pair{&paths.draft, descriptor.draftSource == DraftSource::Packed},
        std::pair{&paths.vision, descriptor.visionSource == VisionSource::Packed}}) {
    if (!packed) continue;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(*directory)) {
      if (!entry.is_regular_file()) continue;
      const uint64_t size = entry.file_size();
      if (size > std::numeric_limits<uint64_t>::max() - bytes) throw std::overflow_error("model weight size overflows");
      bytes += size;
    }
  }
  if (!bytes) throw std::invalid_argument("model package contains no regular files");
  return bytes;
}

uint64_t preparedModelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  return preparedModelWeightBytes(ModelPaths::ofRoot(root), descriptor);
}

} // namespace splash::model
