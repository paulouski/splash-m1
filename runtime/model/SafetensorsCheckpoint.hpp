#pragma once

#include "model/PreparedWeights.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// Checkpoint configuration and safetensors index. Opening parses only
// metadata; tensor data is hashed when a prepared file's identity first
// needs it and read in bounded slices, without loading the MLX runtime or
// allocating tensors.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory,
                        const PreparationCheck &check = {});
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> names() const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  // The bits of an affine projection's per-tensor quantization entry, or the
  // checkpoint's default entry when it has none: the group_size/mode/dtype
  // checks requireQuantization performs still run when that bits value is
  // later bound (AffinePlan.hpp's bind), this only plans the storage.
  [[nodiscard]] uint32_t quantizationBits(std::string_view projection) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  void requireConfigString(std::string_view key, std::string_view expected) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
