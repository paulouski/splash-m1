#pragma once

// Editing this file re-prepares every affine model.
// Plans the affine images (AffinePreparation.hpp) of a safetensors checkpoint:
// a header block, then 16 KiB-aligned sections, each bound to the checkpoint
// tensors it reads. AffineTarget.cpp plans an MLX target with it,
// DraftCheckpoint.cpp a DFlash2 draft.

#include "Checked.hpp"
#include "model/AffinePreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model::affine {

// An image of a header block; append places each section after it.
[[nodiscard]] inline Image image(std::string name, std::string_view magic, uint32_t layer, uint32_t type) {
  return {std::move(name), std::string(magic), layer, type, kWeightFileAlignment, {}, {}};
}

inline void append(Image &image, Section section) {
  section.offset = image.bytes;
  image.bytes = alignWeightOffset(image.bytes + section.bytes);
  image.sections.push_back(std::move(section));
}

// U32 packed codes (dtype "U32") are a raw copy. Every other tensor -- a
// norm, a GDN's conv1d.weight or dt_bias -- accepts BF16 or F16 but is
// canonicalized to BF16 (AffinePreparation.cpp's writeCopy): every kernel
// that reads a Copy section's bytes reads them as bfloat, never half or F32,
// so an F16 source (this checkpoint's norms) is converted, not left as
// stored. Never use this for a scale or bias field: use halfCopy.
inline void copy(Image &image, const std::string &name, std::vector<uint64_t> shape,
                 const std::string &dtype = "BF16") {
  Section section;
  section.bytes = dtype == "U32" ? 4 : kBFloat16Bytes;
  for (uint64_t dimension : shape)
    section.bytes = checkedMultiply<WeightStoreError>(section.bytes, dimension, "affine tensor");
  section.input = {name, dtype == "BF16" ? std::vector<std::string>{"BF16", "F16"} : std::vector<std::string>{dtype},
                   std::move(shape)};
  append(image, std::move(section));
}

// A scale or bias tensor: the canonical prepared dtype is half, so an F16
// source is copied as stored and a BF16 source is converted
// (AffinePreparation.cpp's writeHalfCopy).
inline void halfCopy(Image &image, const std::string &name, std::vector<uint64_t> shape) {
  Section section;
  section.kind = SectionKind::HalfCopy;
  section.bytes = kBFloat16Bytes;
  for (uint64_t dimension : shape)
    section.bytes = checkedMultiply<WeightStoreError>(section.bytes, dimension, "affine tensor");
  section.input = {name, {"BF16", "F16"}, std::move(shape)};
  append(image, std::move(section));
}

// Binds every input of image to its checkpoint tensor, which must have one of
// the input's dtypes and its shape, once the checkpoint states the
// quantization of every affine module the image reads.
inline void bind(Image &image, const SafetensorsCheckpoint &source) {
  const auto bindInput = [&](Input &input) {
    const SourceTensor &tensor = source.require(input.name);
    if (std::find(input.dtypes.begin(), input.dtypes.end(), tensor.dtype) == input.dtypes.end() ||
        tensor.shape != input.shape)
      throw WeightStoreError("source tensor type or shape does not match: " + input.name);
    input.tensor = &tensor;
  };
  for (const auto &[module, bits] : image.quantized) source.requireQuantization(module, bits);
  for (Section &section : image.sections) {
    if (section.parts.empty()) bindInput(section.input);
    for (ProjectionPart &part : section.parts)
      for (Input &field : part.fields) bindInput(field);
  }
}

} // namespace splash::model::affine
