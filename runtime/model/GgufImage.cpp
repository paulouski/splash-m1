#include "model/GgufImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/GgufImageLayout.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>

namespace splash::model::gguf {
namespace {

static_assert([] {
  for (const QuantFormat &format : kQuantFormats) {
    const GgmlTypeTraits *type = ggmlTypeTraits(format.ggml_type);
    if (!type || type->blockElements != format.block_elements || type->blockBytes != format.block_bytes)
      return false;
  }
  return true;
}(), "format table types are the GGUF types, block for block");
// The repack and the reference find meta unit u in native block u.
static_assert([] {
  for (const QuantFormat &format : kQuantFormats)
    if (format.meta_groups * 32 != format.block_elements) return false;
  return true;
}(), "a meta unit is one native block");

bool quantizedType(uint32_t type) { return gguf_format_of(type) != GGUF_FMT_COUNT; }
bool floatType(uint32_t type) { return type == ggml::kF32; }
// F32 or a type widened to F32 exactly.
bool wideFloatType(uint32_t type) { return type == ggml::kF32 || type == ggml::kBF16 || type == ggml::kF16; }
// The token rows the embedding kernel gathers.
bool embeddingType(uint32_t type) { return gguf_embedding_format(gguf_format_of(type)); }
// alpha/beta run as one segment of their shared type: any format of the
// projections (one repacked tensor) or F32 (one float tensor), which BF16
// becomes exactly.
bool alphaBetaType(uint32_t type) { return quantizedType(type) || type == ggml::kF32 || type == ggml::kBF16; }

// Plans one image. A missing tensor or one of a type this build cannot load
// is added to `problems` and left out of the image, so the planner can name
// every such tensor at once; a tensor of the wrong shape throws.
class Builder {
public:
  Builder(const GgufFile &file, const TargetGeometry &geometry, std::vector<std::string> &problems,
          std::string name, uint32_t layer, uint32_t type)
      : file_(file), geometry_(geometry), problems_(problems) {
    image_.name = std::move(name);
    image_.magic = kGgufImageMagic;
    image_.layer = layer;
    image_.type = type;
    const auto header = weightFileHeader(image_.magic, layer, type);
    image_.fills.push_back({0, {header.begin(), header.end()}});
    cursor_ = header.size();
  }

  // A norm as F32, which the norm kernels read unrounded, as llama.cpp does
  // (ops::NormWeights): as stored, or the F32 values BF16 or F16 equal exactly.
  void floatNorm(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, wideFloatType);
    if (!tensor) return;
    if (tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    copy(tensorRows(*tensor, 1, tensor->bytes), tensor->type == ggml::kF32 ? Conversion::None : Conversion::WidenToFloat32);
  }

  // Quantized rows [N, K] repacked into planes, rows in `order`. With `q50`,
  // Q5_0 rows are accepted and repacked as the Q8_0 rows they equal exactly,
  // and BF16 rows as Q8_0 rows rounded from them.
  void quantized(const std::string &name, uint64_t rows, uint64_t columns, RowOrder order = {}, bool q50 = false) {
    const GgufTensor *tensor = q50 ? find(name, [](uint32_t type) {
                                       return quantizedType(type) || type == ggml::kQ5_0 || type == ggml::kBF16;
                                     })
                                   : find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns) throw GgufError("unexpected shape for " + name);
    const bool fromQ50 = tensor->type == ggml::kQ5_0, fromBf16 = tensor->type == ggml::kBF16;
    const uint32_t format = fromQ50 || fromBf16 ? GGUF_FMT_Q80 : gguf_format_of(tensor->type);
    Repack repack = planes(format, rows, columns, name);
    repack.sources.push_back(
        tensorRows(*tensor, rows,
                   fromQ50    ? columns / 32 * kQ50BlockBytes
                   : fromBf16 ? columns * 2
                              : ggufRowBytes(kQuantFormats[format], columns),
                   order));
    image_.repacks.push_back(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows), rows in grouped head
  // order: one 256-row tensor of their format padded with zero rows, or F32 as
  // one float tensor, which BF16 is widened to exactly.
  void alphaBeta(const std::string &betaName, const std::string &alphaName) {
    const GgufTensor *beta = file_.find(betaName), *alpha = file_.find(alphaName);
    if (!beta || !alpha || beta->type != alpha->type || !alphaBetaType(beta->type)) {
      reject(betaName, beta);
      reject(alphaName, alpha);
      return;
    }
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {beta, alpha})
      if (t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
    if (beta->type == ggml::kF32 || beta->type == ggml::kBF16) {
      const uint64_t widening = beta->type == ggml::kBF16 ? 2 : 1;
      const uint64_t bytes = (beta->bytes + alpha->bytes) * widening;
      descriptor(ggml::kF32, 2ull * heads, hidden, {}, {bytes, 0, 0}, betaName);
      uint64_t destination = section(bytes);
      for (const GgufTensor *t : {beta, alpha}) {
        image_.copies.push_back({destination, tensorRows(*t, heads, t->bytes / heads, grouped(0, 1)),
                                 widening == 2 ? Conversion::WidenToFloat32 : Conversion::None});
        destination += t->bytes * widening;
      }
      return;
    }
    if (2 * heads > QUANT_TILE_ROWS) throw GgufError("alpha/beta rows exceed one 256-row tile");
    const uint32_t format = gguf_format_of(beta->type);
    const uint64_t rowBytes = ggufRowBytes(kQuantFormats[format], hidden);
    Repack repack = planes(format, QUANT_TILE_ROWS, hidden, alphaName);
    for (const GgufTensor *t : {beta, alpha}) repack.sources.push_back(tensorRows(*t, heads, rowBytes, grouped(0, 1)));
    image_.repacks.push_back(std::move(repack));
  }

  // The convolution taps of every channel, q and k channels as stored, then
  // value channels in grouped head order, as the exact bf16 values the GDN
  // kernels read.
  void convolution(const std::string &name, uint64_t keyRows) {
    const uint32_t channels = geometry_.convolutionDimension;
    if (const GgufTensor *tensor = floatVector(name, uint64_t{channels} * kGdnConvolutionTaps))
      copy(tensorRows(*tensor, channels, tensor->bytes / channels, grouped(keyRows, geometry_.gdnHeadDimension)),
           Conversion::NarrowToBfloat16);
  }

  // A per value head F32 vector in grouped head order: as stored, or as the
  // exact bf16 values the kernels read.
  void headVector(const std::string &name, Conversion conversion) {
    const uint32_t heads = geometry_.gdnValueHeads;
    if (const GgufTensor *tensor = floatVector(name, heads))
      copy(tensorRows(*tensor, heads, tensor->bytes / heads, grouped(0, 1)), conversion);
  }

  // Native token rows, gathered by the embedding kernel.
  void embeddingRows(const std::string &name) {
    const GgufTensor *tensor = find(name, embeddingType);
    if (!tensor) return;
    if (tensor->rows() != geometry_.vocabularySize || tensor->columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor);
  }

  // An F32 tensor [rows, columns] as stored, or widened from BF16 (the MoE
  // router and the shared-expert scalar gate, which llama.cpp keeps unquantized).
  void floatTensor(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, wideFloatType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    if (tensor->type == ggml::kF32) return copiedRows(*tensor);
    const uint64_t bytes = rows * columns * sizeof(float);
    descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, name);
    copy(tensorRows(*tensor, rows, tensor->bytes / rows), Conversion::WidenToFloat32);
  }

  // An F32, BF16, Q8_0 or Q4_K tensor [rows, columns] as F32 rows: as stored, or the
  // F32 values BF16, Q8_0 (d * q) or Q4_K equal exactly. With `withDescriptor` it is
  // a float tensor a projection runs (the QSA indexer's projections, which
  // are BF16), else its rows alone (the hyper-connections' injection
  // weights; the MTP head's are Q8_0).
  void floatRows(const std::string &name, uint64_t rows, uint64_t columns, bool withDescriptor) {
    const GgufTensor *tensor = find(name, [](uint32_t type) {
      return type == ggml::kF32 || type == ggml::kBF16 || type == ggml::kQ8_0 || type == ggml::kQ4_K;
    });
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32, BF16, Q8_0 or Q4_K [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    const uint64_t bytes = rows * columns * sizeof(float);
    if (withDescriptor) descriptor(ggml::kF32, rows, columns, {}, {bytes, 0, 0}, name);
    image_.copies.push_back({section(bytes), tensorRows(*tensor, rows, tensor->bytes / rows),
                             tensor->type == ggml::kBF16   ? Conversion::WidenToFloat32
                             : tensor->type == ggml::kQ8_0 ? Conversion::DequantizeQ80
                             : tensor->type == ggml::kQ4_K ? Conversion::DequantizeQ4K
                                                           : Conversion::None});
  }

  // Value heads of headRows rows from row `from` on, in grouped order.
  RowOrder grouped(uint64_t from, uint32_t headRows) const {
    if (file_.valueRowsGrouped()) return {};
    return {from, headRows, geometry_.gdnKeyHeads, geometry_.gdnValueHeads / geometry_.gdnKeyHeads};
  }

  Image finish() {
    image_.bytes = alignWeightOffset(cursor_);
    return std::move(image_);
  }

private:
  void reject(const std::string &name, const GgufTensor *tensor) {
    problems_.push_back(name + (tensor ? " (" + ggmlTypeName(tensor->type) + ")" : " (missing)"));
  }

  // The tensor `name` when it is present and of an accepted type.
  const GgufTensor *find(const std::string &name, bool (*accepted)(uint32_t type)) {
    const GgufTensor *tensor = file_.find(name);
    if (tensor && accepted(tensor->type)) return tensor;
    reject(name, tensor);
    return nullptr;
  }

  const GgufTensor *floatVector(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, floatType);
    if (tensor && tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    return tensor;
  }

  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty image section in " + image_.name);
    const uint64_t start = alignWeightOffset(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  static TensorRows tensorRows(const GgufTensor &tensor, uint64_t count, uint64_t rowBytes, RowOrder order = {}) {
    if (!count || count * rowBytes != tensor.bytes) throw GgufError("unexpected size for " + tensor.name);
    return {tensor.name, tensor.type, tensor.offset, count, rowBytes, order, tensor.file};
  }

  // Rows written as stored, or narrowed to bf16 or widened to F32, into their own section.
  void copy(TensorRows source, Conversion conversion = Conversion::None) {
    uint64_t bytes = source.rows * source.rowBytes;
    if (conversion == Conversion::NarrowToBfloat16) bytes /= 2;
    if (conversion == Conversion::WidenToFloat32) bytes *= 2;
    image_.copies.push_back({section(bytes), std::move(source), conversion});
  }

  // A tensor's rows as stored, after their descriptor.
  void copiedRows(const GgufTensor &tensor) {
    descriptor(tensor.type, tensor.rows(), tensor.columns(), {}, {tensor.bytes, 0, 0}, tensor.name);
    copy(tensorRows(tensor, tensor.rows(), tensor.bytes / tensor.rows()));
  }

  // The descriptor and planes of a [rows, columns] quantized tensor: rows of
  // whole 64-column tiles, held in whole plane tiles (the rows past the
  // tensor zero), columns of the format's unit.
  Repack planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
    const QuantFormat &layout = kQuantFormats[format];
    if (rows % GGUF_TILE_COLUMNS || columns % ggufColumnUnit(layout)) throw GgufError("tensor is not tile aligned: " + name);
    const uint64_t stored = ggufStorageRows(rows);
    const GgufPlaneBytes bytes = ggufPlaneBytes(layout, stored, columns);
    descriptor(layout.ggml_type, rows, columns, layout, bytes, name);
    Repack repack;
    repack.format = format;
    repack.rows = stored;
    repack.columns = columns;
    repack.plane0 = section(bytes.plane0);
    repack.plane1 = bytes.plane1 ? section(bytes.plane1) : 0;
    repack.meta = section(bytes.meta);
    return repack;
  }

  // The descriptor of a tensor of `type`, quantized in `format` (a float
  // tensor has neither per-group nor meta bytes).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, const QuantFormat &format,
                  const GgufPlaneBytes &bytes, const std::string &name) {
    if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
      throw GgufError("tensor is too large for its descriptor: " + name);
    GgufTensorDescriptor d{};
    d.type = type;
    d.outputSize = static_cast<uint32_t>(rows);
    d.inputSize = static_cast<uint32_t>(columns);
    d.p0 = format.plane0_bytes;
    d.p1 = format.plane1_bytes;
    d.metaBytes = format.meta_bytes;
    d.metaGroups = format.meta_groups;
    d.plane0Bytes = bytes.plane0;
    d.plane1Bytes = bytes.plane1;
    d.metaTotalBytes = bytes.meta;
    std::vector<uint8_t> encoded(sizeof d);
    std::memcpy(encoded.data(), &d, sizeof d);
    image_.fills.push_back({section(encoded.size()), std::move(encoded)});
  }

  const GgufFile &file_;
  const TargetGeometry &geometry_;
  std::vector<std::string> &problems_;
  Image image_;
  uint64_t cursor_ = 0;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

// The target geometry the metadata declares, with the rotary embedding and
// norms the kernels compute: the RoPE base and rotated dimensions, the RMS
// epsilon and no RoPE scaling. One error names every mismatch.
void requireMetadata(const GgufFile &file, const TargetGeometry &geometry) {
  const std::string arch = geometry.architecture();
  if (file.architecture() != arch)
    throw GgufError("GGUF architecture is " + file.architecture() + ", but the package's target is " + arch);
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  // A float equal to value up to its F32 rounding.
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("rope.dimension_count", 2ull * geometry.rotaryPairs);
  expectFloat("rope.freq_base", geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", 1e-6);
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  expect("full_attention_interval", geometry.fullAttentionPeriod);
  expect("ssm.conv_kernel", kGdnConvolutionTaps);
  expect("ssm.group_count", geometry.gdnKeyHeads);
  expect("ssm.time_step_rank", geometry.gdnValueHeads);
  expect("ssm.state_size", geometry.gdnHeadDimension);
  expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  if (geometry.qwen4()) {
    expect("hyper_connection.count", geometry.hyperConnections);
    expect("hyper_connection.low_rank", geometry.hyperRank);
    expect("attention.indexer.head_count", geometry.indexerHeads);
    expect("attention.indexer.key_length", geometry.indexerHeadDimension);
    expect("attention.indexer.top_k", uint64_t{geometry.indexerTopBlocks} * geometry.indexerBlockTokens);
    if (geometry.hasPle()) {
      expect("ple.ngram_size", geometry.pleNgram);
      expect("ple.heads_per_ngram", geometry.pleHeadsPerNgram);
      expect("ple.conv_kernel", geometry.pleConvolutionTaps);
      expect("embedding_length_per_layer_input", geometry.pleHeadDimension);
    }
  }
  if (geometry.sparseMoe()) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length", geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

// One hyper-connection mix (model/Qwen4Exp.hpp Qwen4HyperWeights): its norm,
// down and up projections and, when it injects, its injection weights.
void hyperImage(Builder &b, const TargetGeometry &g, const std::string &stem, bool inject) {
  const uint64_t streams = g.streamWidth();
  b.floatNorm(stem + "_norm.weight", streams);
  b.quantized(stem + "_down.weight", g.hyperRank, streams, {}, true);
  b.quantized(stem + "_up.weight", streams, g.hyperRank, {}, true);
  if (inject) b.floatRows(stem + "_inject.weight", g.hyperConnections, streams, false);
}

void indexerImage(Builder &b, const TargetGeometry &g, const std::string &p) {
  b.floatRows(p + "indexer.q_proj.weight", uint64_t{g.indexerHeads} * g.indexerHeadDimension, g.hiddenSize, true);
  b.floatRows(p + "indexer.k_proj.weight", g.indexerHeadDimension, g.hiddenSize, true);
  b.floatNorm(p + "indexer.q_norm.weight", g.indexerHeadDimension);
  b.floatNorm(p + "indexer.k_norm.weight", g.indexerHeadDimension);
}

void mixerImage(Builder &b, const TargetGeometry &g, const std::string &p, bool full) {
  if (full) {
    b.quantized(p + "attn_q.weight", 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension),
                g.hiddenSize);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(p + "attn_k.weight", kvRows, g.hiddenSize);
    b.quantized(p + "attn_v.weight", kvRows, g.hiddenSize);
    b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
    b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
    b.quantized(p + "attn_output.weight", g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows; // q and k
    b.quantized(p + "attn_qkv.weight", g.convolutionDimension, g.hiddenSize, b.grouped(keyRows, g.gdnHeadDimension));
    b.quantized(p + "attn_gate.weight", valueRows, g.hiddenSize, b.grouped(0, g.gdnHeadDimension));
    b.alphaBeta(p + "ssm_beta.weight", p + "ssm_alpha.weight");
    b.convolution(p + "ssm_conv1d.weight", keyRows);
    b.headVector(p + "ssm_a", Conversion::None);
    b.headVector(p + "ssm_dt.bias", Conversion::NarrowToBfloat16);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(p + "ssm_out.weight", g.hiddenSize, valueRows);
  }
}

void ffnImage(Builder &b, const TargetGeometry &g, const std::string &p) {
  if (g.sparseMoe()) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, g.hiddenSize);
    b.quantized(p + "ffn_gate_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_up_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_down_exps.weight", routed * g.hiddenSize, width);
    b.quantized(p + "ffn_gate_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_up_shexp.weight", width, g.hiddenSize);
    b.quantized(p + "ffn_down_shexp.weight", g.hiddenSize, width);
    b.floatTensor(p + "ffn_gate_inp_shexp.weight", 1, g.hiddenSize);
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_down.weight", g.hiddenSize, g.intermediateSize);
  }
}

// The PLE layer's projections, norms and dilated convolution taps.
void pleImage(Builder &b, const TargetGeometry &g, const std::string &p) {
  const uint64_t width = uint64_t{g.pleNgram - 1} * g.pleHeadsPerNgram * g.pleHeadDimension;
  b.quantized(p + "ple_key.weight", g.streamWidth(), width, {}, true);
  b.quantized(p + "ple_value.weight", g.hiddenSize, width, {}, true);
  b.floatNorm(p + "ple_norm_key.weight", g.streamWidth());
  b.floatNorm(p + "ple_norm_query.weight", g.streamWidth());
  b.floatNorm(p + "ple_norm_conv.weight", g.streamWidth());
  b.floatNorm(p + "ple_conv1d.weight", g.streamWidth() * g.pleConvolutionTaps);
}

Image qwen4LayerImage(const GgufFile &file, const TargetGeometry &g, std::vector<std::string> &problems,
                      uint32_t index) {
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  hyperImage(b, g, p + "hc_attn", true);
  mixerImage(b, g, p, full);
  if (full) indexerImage(b, g, p);
  hyperImage(b, g, p + "hc_ffn", true);
  ffnImage(b, g, p);
  if (g.hasPle() && index == g.pleLayer) pleImage(b, g, p);
  return b.finish();
}

// The arrays a qwen4exp GGUF declares, which requireMetadata's scalars leave
// out: each layer's QSA compress ratio (the target's layers, and an MTP
// file's block when it carries one), the multimodal RoPE sections the kernels
// hardcode and the PLE layer. One error names every mismatch.
void requireQwen4Arrays(const GgufFile &file, const TargetGeometry &g) {
  const std::string arch = g.architecture();
  std::string mismatched;
  const auto ratios = file.numericArray(arch + ".attention.compress_ratios");
  bool ratiosMatch = ratios && ratios->size() >= g.layers;
  for (uint32_t layer = 0; ratiosMatch && layer < g.layers; ++layer)
    ratiosMatch = (*ratios)[layer] == (g.isFullAttentionLayer(layer) ? g.indexerBlockTokens : 0);
  if (!ratiosMatch) mismatched += "attention.compress_ratios";
  const auto sections = file.numericArray(arch + ".rope.dimension_sections");
  if (!sections || sections->size() != 4 || (*sections)[0] != 11 || (*sections)[1] != 11 || (*sections)[2] != 10 ||
      (*sections)[3] != 0)
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.dimension_sections (expected 11, 11, 10, 0)");
  if (g.hasPle()) {
    const auto layers = file.numericArray(arch + ".ple.layers");
    if (!layers || layers->size() != 1 || (*layers)[0] != g.pleLayer)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string("ple.layers");
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

Image layerImage(const GgufFile &file, const TargetGeometry &g, std::vector<std::string> &problems,
                 uint32_t index) {
  if (g.qwen4()) return qwen4LayerImage(file, g, problems, index);
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
  mixerImage(b, g, p, full);
  b.floatNorm(p + "post_attention_norm.weight", g.hiddenSize);
  ffnImage(b, g, p);
  return b.finish();
}

// A rotated GGUF (GgufRotation) must name exactly what the loader rotates:
// every tensor the images of a dense target repack, whose segments read
// H (D x) (ops::InputRotation) while float segments (F32 or BF16 alpha/beta)
// read x as it is, and the token table, which the rotated gather decodes from
// PQ2_0 rows, with the GDN value heads of the rotated inputs grouped.
void requireRotation(const GgufFile &file, const TargetGeometry &g, const std::vector<Image> &images) {
  const GgufRotation &rotation = *file.rotation();
  if (g.sparseMoe()) throw GgufError("rotated weights are supported for dense targets only");
  if (!rotation.valueHeadsGrouped)
    throw GgufError("rotated GDN inputs must keep their value heads grouped (prism.hadamard.gdn_v_grouped)");
  std::set<std::string, std::less<>> repacked;
  for (const Image &image : images)
    for (const Repack &repack : image.repacks)
      for (const TensorRows &source : repack.sources) repacked.insert(source.name);
  if (rotation.weights != repacked)
    throw GgufError("the rotation must name every quantized tensor of the target and nothing else");
  if (rotation.tables != std::set<std::string, std::less<>>{"token_embd.weight"} ||
      file.require("token_embd.weight").type != ggml::kPQ2_0)
    throw GgufError("the rotation's one token table must be token_embd.weight in PQ2_0");
}

} // namespace

std::vector<Image> planImages(const GgufFile &file, const TargetGeometry &geometry) {
  requireMetadata(file, geometry);
  if (geometry.qwen4()) requireQwen4Arrays(file, geometry);
  std::vector<std::string> problems;
  std::vector<Image> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer)
    images.push_back(layerImage(file, geometry, problems, layer));
  Builder head(file, geometry, problems, "head.bin", geometry.layers, 2);
  if (geometry.qwen4()) hyperImage(head, geometry, "output_hc", false);
  else head.floatNorm("output_norm.weight", geometry.hiddenSize);
  head.quantized("output.weight", geometry.vocabularySize, geometry.hiddenSize);
  images.push_back(head.finish());
  Builder embedding(file, geometry, problems, "embedding.bin", geometry.vocabularySize, geometry.hiddenSize);
  embedding.embeddingRows("token_embd.weight");
  images.push_back(embedding.finish());
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems) names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("GGUF tensors this build cannot load: " + names);
  }
  if (file.rotation()) requireRotation(file, geometry, images);
  return images;
}

Image planMtpImage(const GgufFile &file, const TargetGeometry &target) {
  if (!target.qwen4()) throw GgufError("only a qwen4exp target has an MTP head");
  TargetGeometry g = target;
  g.pleNgram = 0; // the MTP file carries no PLE metadata
  requireMetadata(file, g);
  requireQwen4Arrays(file, g);
  if (file.unsignedValue(g.architecture() + std::string(".nextn_predict_layers")) != 1)
    throw GgufError("the MTP GGUF must hold one nextn layer");
  std::vector<std::string> problems;
  const std::string p = prefix(g.layers);
  Builder b(file, g, problems, "mtp.bin", g.layers, 4);
  hyperImage(b, g, p + "hc_attn", true);
  mixerImage(b, g, p, true);
  indexerImage(b, g, p);
  hyperImage(b, g, p + "hc_ffn", true);
  ffnImage(b, g, p);
  b.quantized(p + "nextn.eh_proj.weight", g.hiddenSize, 2ull * g.hiddenSize);
  b.floatNorm(p + "nextn.enorm.weight", g.hiddenSize);
  b.floatNorm(p + "nextn.hnorm.weight", g.streamWidth());
  hyperImage(b, g, p + "nextn.hc_head", false);
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems) names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("MTP GGUF tensors this build cannot load: " + names);
  }
  return b.finish();
}

std::optional<PleTable> readPleTable(const GgufFile &file, const TargetGeometry &g, const GgufFile *external) {
  if (!g.hasPle()) return std::nullopt;
  const std::string arch = g.architecture();
  const uint32_t heads = (g.pleNgram - 1) * g.pleHeadsPerNgram;
  if (g.pleNgram > 4 || heads > pleMaximumHeads) throw GgufError("unsupported PLE n-gram shape");
  const auto array = [&](const char *key, size_t count) {
    const auto values = file.numericArray(arch + ".ple." + key);
    if (!values || values->size() < count) throw GgufError("GGUF PLE metadata is missing: ple." + std::string(key));
    return *values;
  };
  // Every value is an integer below 2^53, which the parser's doubles hold exactly.
  const auto exact = [](double value, double limit) {
    if (!(value >= 0 && value <= limit) || value != std::floor(value))
      throw GgufError("GGUF PLE metadata holds an invalid value");
    return static_cast<uint64_t>(value);
  };
  const GgufTensor *modelTable = external ? file.find("per_layer_token_embd.weight") : &file.require("per_layer_token_embd.weight");
  const GgufFile &tableFile = external ? *external : file;
  const GgufTensor &table = tableFile.require("per_layer_token_embd.weight");
  // The 32-wide formats the embedding gather reads natively.
  uint32_t format = GGUF_FMT_COUNT;
  for (uint32_t i = 0; i < GGUF_FMT_COUNT; ++i)
    if (kQuantFormats[i].ggml_type == table.type && kQuantFormats[i].block_elements == 32 && gguf_embedding_format(i))
      format = i;
  if (format == GGUF_FMT_COUNT || table.dims.size() != 2 || table.columns() != g.pleHeadDimension ||
      table.columns() % 32 != 0)
    throw GgufError("per_layer_token_embd must be 2-D rows of " + std::to_string(g.pleHeadDimension) +
                    " values in a 32-wide format the embedding gather reads (IQ4_NL, Q4_0, Q4_1, Q8_0)");
  const uint32_t rowBytes = static_cast<uint32_t>(table.columns() / 32 * kQuantFormats[format].block_bytes);
  if (rowBytes > pleMaxRowBytes || table.bytes != table.rows() * rowBytes)
    throw GgufError("per_layer_token_embd has an unexpected row size");
  if (modelTable && external && (modelTable->rows() != table.rows() || modelTable->columns() != table.columns()))
    throw GgufError("the external PLE table's shape differs from the model's per_layer_token_embd");
  PleTable result;
  result.rowBytes = rowBytes;
  result.format = format;
  result.rows = table.rows();
  result.hash.ngram = g.pleNgram;
  result.hash.headsPerNgram = g.pleHeadsPerNgram;
  const auto multipliers = array("layer_multipliers", g.pleNgram);
  for (uint32_t i = 0; i < g.pleNgram; ++i) result.hash.multipliers[i] = exact(multipliers[i], 9007199254740992.0);
  const auto offsets = array("head_offsets", heads), sizes = array("head_vocab_sizes", heads);
  for (uint32_t h = 0; h < heads; ++h) {
    const uint64_t offset = exact(offsets[h], 4294967295.0), size = exact(sizes[h], 4294967295.0);
    if (!size || offset + size > result.rows) throw GgufError("a PLE head's rows lie outside per_layer_token_embd");
    result.hash.offsets.push_back(static_cast<uint32_t>(offset));
    result.hash.vocabularies.push_back(static_cast<uint32_t>(size));
  }
  result.hash.eos = static_cast<uint32_t>(file.unsignedValue(arch + ".ple.eos_token_id").value_or(0));
  if (!result.hash.eos) throw GgufError("GGUF PLE metadata is missing: ple.eos_token_id");
  if (result.rows > UINT32_MAX) throw GgufError("per_layer_token_embd has more rows than a row id holds");
  const WeightSource &source = tableFile.source(table.file);
  result.path = source.path();
  result.dataOffset = source.dataOffset() + table.offset;
  return result;
}

} // namespace splash::model::gguf
