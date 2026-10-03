// Editing this file re-prepares every Prism MLX model.
#include "WeightPreparationIdentity.hpp"
#include "model/PrismMlx.hpp"

#include "model/SafetensorsCheckpoint.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>

namespace splash::model {
namespace {

static_assert(std::endian::native == std::endian::little, "packed codes are copied as little-endian bytes");

constexpr std::string_view kModelType = "prism_hadamard_qwen35";
constexpr std::string_view kRotationPrefix = "prism.hadamard.";
constexpr uint64_t kGroup = 128;                       // values per scale and per PQ2_0 block
constexpr uint64_t kBlockBytes = 2 + kGroup / 4;       // F16 scale, 2-bit codes
constexpr uint64_t kRowBatch = 64;                     // rows synthesized per read piece
constexpr uint64_t kAlignment = 32;

[[noreturn]] void fail(const std::string &message) { throw GgufError("Prism MLX checkpoint: " + message); }

NSData *readFile(const std::filesystem::path &path) {
  if (std::filesystem::file_size(path) > 8 * 1024 * 1024) fail(path.filename().string() + " exceeds its bound");
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  if (!data) fail("cannot read " + path.string());
  return data;
}

NSDictionary *parseObject(NSData *data, const std::filesystem::path &path) {
  id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  if (![parsed isKindOfClass:[NSDictionary class]]) fail(path.filename().string() + " is not a JSON object");
  return parsed;
}

NSDictionary *readConfig(const std::filesystem::path &directory) {
  const auto path = directory / "config.json";
  return parseObject(readFile(path), path);
}

std::string text(id value) {
  if (![value isKindOfClass:[NSString class]]) return {};
  return [value UTF8String];
}

double number(id value, std::string_view what) {
  if (![value isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    fail(std::string(what) + " must be a number");
  return [value doubleValue];
}

uint64_t integer(id value, std::string_view what) {
  const double real = number(value, what);
  if (real < 0 || real > 9007199254740991.0 || std::floor(real) != real) fail(std::string(what) + " must be an integer");
  return uint64_t(real);
}

NSDictionary *object(NSDictionary *parent, NSString *key) {
  id value = parent[key];
  if (![value isKindOfClass:[NSDictionary class]]) fail(std::string([key UTF8String]) + " must be an object");
  return value;
}

// The synthesized tensors, found by their offset in the GGUF's tensor data.
struct Synthesized {
  enum Kind { Pq20, Copy, NegExp } kind = Copy;
  std::string name;
  uint64_t offset = 0, bytes = 0, rowBytes = 0;
  const SourceTensor *weight = nullptr, *scales = nullptr, *biases = nullptr;
};

class Synthesizer final {
public:
  Synthesizer(std::unique_ptr<SafetensorsCheckpoint> checkpoint, std::vector<Synthesized> tensors)
      : checkpoint_(std::move(checkpoint)), tensors_(std::move(tensors)) {}

  void read(uint64_t offset, std::span<uint8_t> out) const {
    auto found = std::upper_bound(tensors_.begin(), tensors_.end(), offset,
                                  [](uint64_t at, const Synthesized &t) { return at < t.offset; });
    if (found == tensors_.begin()) fail("read below the first tensor");
    const Synthesized &tensor = *--found;
    const uint64_t local = offset - tensor.offset;
    if (local > tensor.bytes || out.size() > tensor.bytes - local) fail("read past " + tensor.name);
    if (out.empty()) return;
    switch (tensor.kind) {
    case Synthesized::Copy: tensor.weight->read(local, out); return;
    case Synthesized::NegExp: negativeExponential(tensor, local, out); return;
    case Synthesized::Pq20: break;
    }
    const uint64_t first = local / tensor.rowBytes, last = (local + out.size() - 1) / tensor.rowBytes;
    for (uint64_t row = first; row <= last; row += kRowBatch) {
      const uint64_t count = std::min(kRowBatch, last + 1 - row);
      std::vector<uint8_t> rows(count * tensor.rowBytes);
      blocks(tensor, row, rows);
      const uint64_t begin = std::max(local, row * tensor.rowBytes),
                     end = std::min(local + out.size(), (row + count) * tensor.rowBytes);
      std::memcpy(out.data() + (begin - local), rows.data() + (begin - row * tensor.rowBytes), end - begin);
    }
  }

private:
  // ssm_a is -exp(A_log), rounded once from double.
  static void negativeExponential(const Synthesized &tensor, uint64_t local, std::span<uint8_t> out) {
    std::vector<float> values(tensor.bytes / 4);
    tensor.weight->read(0, {reinterpret_cast<uint8_t *>(values.data()), tensor.bytes});
    for (float &value : values) value = float(-std::exp(double(value)));
    std::memcpy(out.data(), reinterpret_cast<const uint8_t *>(values.data()) + local, out.size());
  }

  // The PQ2_0 blocks of whole rows [first, first + rows.size() / rowBytes).
  static void blocks(const Synthesized &tensor, uint64_t first, std::span<uint8_t> rows) {
    const uint64_t count = rows.size() / tensor.rowBytes, groups = tensor.rowBytes / kBlockBytes;
    const uint64_t codeBytes = groups * (kBlockBytes - 2), scaleBytes = groups * 2;
    std::vector<uint8_t> codes(count * codeBytes), scales(count * scaleBytes), biases(count * scaleBytes);
    tensor.weight->read(first * codeBytes, codes);
    tensor.scales->read(first * scaleBytes, scales);
    tensor.biases->read(first * scaleBytes, biases);
    for (uint64_t row = 0; row < count; ++row)
      for (uint64_t group = 0; group < groups; ++group) {
        const uint8_t *code = codes.data() + row * codeBytes + group * (kBlockBytes - 2);
        const uint8_t *scale = scales.data() + row * scaleBytes + group * 2;
        const uint8_t *bias = biases.data() + row * scaleBytes + group * 2;
        // Codes 0, 1, 2 are -s, 0, +s: bias == -scale and no code 3.
        if (bias[0] != scale[0] || bias[1] != (scale[1] ^ 0x80))
          fail(tensor.name + " row " + std::to_string(first + row) + " has biases other than -scales");
        for (uint64_t word = 0; word < (kBlockBytes - 2) / 4; ++word) {
          uint32_t packed;
          std::memcpy(&packed, code + 4 * word, 4);
          if ((packed & (packed >> 1)) & 0x55555555u)
            fail(tensor.name + " row " + std::to_string(first + row) + " holds the unused code 3");
        }
        uint8_t *block = rows.data() + (row * groups + group) * kBlockBytes;
        std::memcpy(block, scale, 2);
        std::memcpy(block + 2, code, kBlockBytes - 2);
      }
  }

  std::unique_ptr<SafetensorsCheckpoint> checkpoint_; // owns the SourceTensors read through
  std::vector<Synthesized> tensors_;
};

// Builds the GGUF tensor table of the checkpoint's tensors.
class Binder final {
public:
  Binder(const SafetensorsCheckpoint &checkpoint, const std::filesystem::path &file,
         const std::map<uint32_t, std::vector<int8_t>> &signs)
      : checkpoint_(checkpoint), file_(file), signs_(signs) {}

  // A rotated PQ2_0 projection [rows, columns] from the module `prefix`.
  void projection(const std::string &name, const std::string &prefix, uint64_t rows, uint64_t columns) {
    Synthesized tensor;
    tensor.kind = Synthesized::Pq20;
    tensor.weight = &require(prefix + ".weight", "U32", {rows, columns / 16});
    tensor.scales = &require(prefix + ".scales", "F16", {rows, columns / kGroup});
    tensor.biases = &require(prefix + ".biases", "F16", {rows, columns / kGroup});
    const SourceTensor &signs = require(prefix + ".signs", "F32", {columns});
    const auto expected = signs_.find(uint32_t(columns));
    if (expected == signs_.end()) fail("the rotation has no signs of width " + std::to_string(columns));
    std::vector<float> values(columns);
    signs.read(0, {reinterpret_cast<uint8_t *>(values.data()), columns * 4});
    for (uint64_t i = 0; i < columns; ++i)
      if (values[i] != float(expected->second[i])) fail(prefix + ".signs differs from the rotation's signs");
    tensor.rowBytes = columns / kGroup * kBlockBytes;
    const uint64_t bytes = rows * tensor.rowBytes;
    names_[prefix + ".weight"] = name;
    add(std::move(tensor), name, ggml::kPQ2_0, {columns, rows}, bytes);
  }

  // An F32 tensor, as stored or as -exp.
  void floats(const std::string &name, const std::string &source, std::vector<uint64_t> shape,
              std::vector<uint64_t> dims, Synthesized::Kind kind = Synthesized::Copy) {
    Synthesized tensor;
    tensor.kind = kind;
    tensor.weight = &require(source, "F32", shape);
    uint64_t bytes = 4;
    for (uint64_t dim : dims) bytes *= dim;
    add(std::move(tensor), name, ggml::kF32, std::move(dims), bytes);
  }

  // Every language model tensor must have been taken; the vision tower is not read.
  void requireNoOthers() const {
    for (const std::string &name : checkpoint_.names())
      if (!name.starts_with("vision_tower.") && !used_.contains(name)) fail("unexpected tensor " + name);
  }

  [[nodiscard]] const std::map<std::string, std::string> &weightNames() const { return names_; }
  GgufMemoryHeader header;
  std::vector<Synthesized> tensors;

private:
  const SourceTensor &require(const std::string &name, std::string_view dtype, std::vector<uint64_t> shape) {
    const SourceTensor *tensor = checkpoint_.find(name);
    if (!tensor) fail("missing tensor " + name);
    if (tensor->dtype != dtype || tensor->shape != shape) fail("unexpected dtype or shape of " + name);
    if (tensor->file->path() != file_) fail(name + " is not in " + file_.filename().string());
    used_.insert(name);
    return *tensor;
  }

  void add(Synthesized tensor, const std::string &name, uint32_t type, std::vector<uint64_t> dims, uint64_t bytes) {
    cursor_ = (cursor_ + kAlignment - 1) / kAlignment * kAlignment;
    tensor.name = name;
    tensor.offset = cursor_;
    tensor.bytes = bytes;
    cursor_ += bytes;
    GgufTensor entry;
    entry.name = name;
    entry.type = type;
    entry.dims = std::move(dims);
    entry.offset = tensor.offset;
    header.tensors.push_back(std::move(entry));
    tensors.push_back(std::move(tensor));
  }

  const SafetensorsCheckpoint &checkpoint_;
  const std::filesystem::path &file_;
  const std::map<uint32_t, std::vector<int8_t>> &signs_;
  std::set<std::string> used_;
  std::map<std::string, std::string> names_;
  uint64_t cursor_ = 0;
};

// The prism.hadamard.* metadata of hadamard.json, with its tensor names as
// the GGUF's; the signs by width.
std::map<uint32_t, std::vector<int8_t>> readSigns(NSDictionary *hadamard) {
  NSArray *widths = hadamard[@"prism.hadamard.sign_widths"], *values = hadamard[@"prism.hadamard.sign_values"];
  if (![widths isKindOfClass:[NSArray class]] || ![values isKindOfClass:[NSArray class]])
    fail("hadamard.json has no signs");
  std::map<uint32_t, std::vector<int8_t>> signs;
  NSUInteger at = 0;
  for (id width in widths) {
    const uint64_t count = integer(width, "sign width");
    if (!count || count > values.count - at) fail("invalid sign width");
    auto &into = signs[uint32_t(count)];
    if (!into.empty()) fail("duplicate sign width");
    for (uint64_t i = 0; i < count; ++i, ++at) {
      const double sign = number(values[at], "sign");
      if (sign != 1 && sign != -1) fail("signs are +1 or -1");
      into.push_back(int8_t(sign));
    }
  }
  if (at != values.count) fail("signs do not match their widths");
  return signs;
}

void readRotation(NSDictionary *hadamard, const std::map<std::string, std::string> &weightNames,
                  GgufMemoryHeader &header) {
  for (NSString *key in hadamard) {
    const std::string name = [key UTF8String];
    if (!name.starts_with(kRotationPrefix)) fail("hadamard.json holds a key outside prism.hadamard.: " + name);
    id value = hadamard[key];
    if ([value isKindOfClass:[NSString class]]) {
      header.stringValues[name] = text(value);
    } else if ([value isKindOfClass:[NSNumber class]]) {
      // A JSON true is the 1 a GGUF boolean holds.
      header.unsignedValues[name] = CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()
          ? uint64_t([value boolValue]) : integer(value, name);
    } else if ([value isKindOfClass:[NSArray class]]) {
      NSArray *array = value;
      const bool strings = array.count && [array[0] isKindOfClass:[NSString class]];
      if (strings) {
        auto &into = header.names[name];
        for (id entry : array) {
          const std::string mlx = text(entry);
          const auto found = weightNames.find(mlx);
          if (found == weightNames.end()) fail("hadamard.json names an unknown tensor: " + mlx);
          into.push_back(found->second);
        }
      } else {
        auto &into = header.arrays[name];
        for (id entry : array) into.push_back(number(entry, name));
      }
    } else {
      fail("unsupported hadamard.json value: " + name);
    }
  }
}

// The qwen35.* metadata of the checkpoint's own configuration, which the
// planner checks against the target geometry.
void readMetadata(NSDictionary *config, GgufMemoryHeader &header) {
  NSDictionary *t = object(config, @"text_config");
  if (text(t[@"model_type"]) != "qwen3_5_text") fail("text_config.model_type is not qwen3_5_text");
  if (![t[@"tie_word_embeddings"] isEqual:@NO] || ![t[@"attn_output_gate"] isEqual:@YES])
    fail("text_config must untie the embeddings and gate attention");
  NSDictionary *rope = object(t, @"rope_parameters");
  if (text(rope[@"rope_type"]) != "default") fail("rope_parameters.rope_type is not default");
  const double headDimension = number(t[@"head_dim"], "head_dim");
  const double rotated = headDimension * number(rope[@"partial_rotary_factor"], "partial_rotary_factor");
  if (rotated != std::floor(rotated)) fail("the rotated dimensions are not whole");
  const uint64_t valueHeads = integer(t[@"linear_num_value_heads"], "linear_num_value_heads");
  const uint64_t keyDimension = integer(t[@"linear_key_head_dim"], "linear_key_head_dim");
  const uint64_t valueDimension = integer(t[@"linear_value_head_dim"], "linear_value_head_dim");
  if (keyDimension != valueDimension) fail("linear key and value head dimensions differ");
  auto &u = header.unsignedValues;
  u["qwen35.block_count"] = integer(t[@"num_hidden_layers"], "num_hidden_layers");
  u["qwen35.embedding_length"] = integer(t[@"hidden_size"], "hidden_size");
  u["qwen35.feed_forward_length"] = integer(t[@"intermediate_size"], "intermediate_size");
  u["qwen35.attention.head_count"] = integer(t[@"num_attention_heads"], "num_attention_heads");
  u["qwen35.attention.head_count_kv"] = integer(t[@"num_key_value_heads"], "num_key_value_heads");
  u["qwen35.attention.key_length"] = uint64_t(headDimension);
  u["qwen35.attention.value_length"] = uint64_t(headDimension);
  u["qwen35.rope.dimension_count"] = uint64_t(rotated);
  u["qwen35.full_attention_interval"] = integer(t[@"full_attention_interval"], "full_attention_interval");
  u["qwen35.ssm.conv_kernel"] = integer(t[@"linear_conv_kernel_dim"], "linear_conv_kernel_dim");
  u["qwen35.ssm.group_count"] = integer(t[@"linear_num_key_heads"], "linear_num_key_heads");
  u["qwen35.ssm.time_step_rank"] = valueHeads;
  u["qwen35.ssm.state_size"] = keyDimension;
  u["qwen35.ssm.inner_size"] = valueHeads * valueDimension;
  header.floatValues["qwen35.rope.freq_base"] = number(rope[@"rope_theta"], "rope_theta");
  header.floatValues["qwen35.attention.layer_norm_rms_epsilon"] = number(t[@"rms_norm_eps"], "rms_norm_eps");
  header.stringValues["general.architecture"] = "qwen35";
}

} // namespace

bool isPrismMlx(const std::filesystem::path &directory) {
  @autoreleasepool {
    if (!std::filesystem::is_regular_file(directory / "config.json")) return false;
    return text(readConfig(directory)[@"model_type"]) == kModelType;
  }
}

void requirePrismMlxConfig(const std::filesystem::path &directory) {
  @autoreleasepool {
    NSDictionary *config = readConfig(directory);
    if (text(config[@"model_type"]) != kModelType) fail("config.json is not a Prism Hadamard checkpoint");
    NSDictionary *quantization = object(config, @"quantization");
    id mode = quantization[@"mode"];
    if (integer(config[@"schema_version"], "schema_version") != 2 || integer(quantization[@"bits"], "bits") != 2 ||
        integer(quantization[@"group_size"], "group_size") != kGroup || (mode && text(mode) != "affine"))
      fail("requires schema 2 and MLX affine 2-bit/group-128 quantization");
    if (!std::filesystem::is_regular_file(directory / "hadamard.json")) fail("hadamard.json is missing");
  }
}

std::filesystem::path prismMlxFile(const std::filesystem::path &directory) {
  return directory / "model.safetensors";
}

PrismMlxView bindPrismMlx(WeightSource &source, const std::filesystem::path &directory,
                          const gguf::TargetGeometry &g) {
  @autoreleasepool {
    if (g.sparseMoe()) fail("only dense targets are supported");
    requirePrismMlxConfig(directory);
    PrismMlxView view;
    readMetadata(readConfig(directory), view.header);
    NSData *hadamardBytes = readFile(directory / "hadamard.json");
    NSDictionary *hadamard = parseObject(hadamardBytes, "hadamard.json");
    const auto signs = readSigns(hadamard);

    auto checkpoint = std::make_unique<SafetensorsCheckpoint>(directory, PreparationCheck{});
    checkpoint->requireLayerTypes(g.layers, g.fullAttentionPeriod);
    const std::string language = "language_model.";
    Binder binder(*checkpoint, source.path(), signs);
    const uint64_t valueRows = uint64_t{g.gdnValueHeads} * g.gdnHeadDimension;
    binder.projection("output.weight", language + "lm_head", g.vocabularySize, g.hiddenSize);
    binder.floats("output_norm.weight", language + "model.norm.weight", {g.hiddenSize}, {g.hiddenSize});
    binder.projection("token_embd.weight", language + "model.embed_tokens", g.vocabularySize, g.hiddenSize);
    for (uint32_t layer = 0; layer < g.layers; ++layer) {
      const std::string b = "blk." + std::to_string(layer) + ".";
      const std::string p = language + "model.layers." + std::to_string(layer) + ".";
      binder.floats(b + "attn_norm.weight", p + "input_layernorm.weight", {g.hiddenSize}, {g.hiddenSize});
      binder.floats(b + "post_attention_norm.weight", p + "post_attention_layernorm.weight", {g.hiddenSize},
                    {g.hiddenSize});
      binder.projection(b + "ffn_gate.weight", p + "mlp.gate_proj", g.intermediateSize, g.hiddenSize);
      binder.projection(b + "ffn_up.weight", p + "mlp.up_proj", g.intermediateSize, g.hiddenSize);
      binder.projection(b + "ffn_down.weight", p + "mlp.down_proj", g.hiddenSize, g.intermediateSize);
      if (g.isFullAttentionLayer(layer)) {
        const uint64_t kv = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
        binder.projection(b + "attn_q.weight", p + "self_attn.q_proj", 2ull * g.attentionWidth, g.hiddenSize);
        binder.projection(b + "attn_k.weight", p + "self_attn.k_proj", kv, g.hiddenSize);
        binder.projection(b + "attn_v.weight", p + "self_attn.v_proj", kv, g.hiddenSize);
        binder.projection(b + "attn_output.weight", p + "self_attn.o_proj", g.hiddenSize, g.attentionWidth);
        binder.floats(b + "attn_q_norm.weight", p + "self_attn.q_norm.weight", {g.attentionHeadDimension},
                      {g.attentionHeadDimension});
        binder.floats(b + "attn_k_norm.weight", p + "self_attn.k_norm.weight", {g.attentionHeadDimension},
                      {g.attentionHeadDimension});
      } else {
        const std::string a = p + "linear_attn.";
        binder.projection(b + "attn_qkv.weight", a + "in_proj_qkv", g.convolutionDimension, g.hiddenSize);
        binder.projection(b + "attn_gate.weight", a + "in_proj_z", valueRows, g.hiddenSize);
        binder.projection(b + "ssm_out.weight", a + "out_proj", g.hiddenSize, valueRows);
        // beta and alpha are bf16-exact F32, which the GGUF stores as BF16.
        binder.floats(b + "ssm_beta.weight", a + "in_proj_b.weight", {g.gdnValueHeads, g.hiddenSize},
                      {g.hiddenSize, g.gdnValueHeads});
        binder.floats(b + "ssm_alpha.weight", a + "in_proj_a.weight", {g.gdnValueHeads, g.hiddenSize},
                      {g.hiddenSize, g.gdnValueHeads});
        binder.floats(b + "ssm_conv1d.weight", a + "conv1d.weight", {g.convolutionDimension, 4, 1},
                      {4, g.convolutionDimension});
        binder.floats(b + "ssm_a", a + "A_log", {g.gdnValueHeads}, {g.gdnValueHeads}, Synthesized::NegExp);
        binder.floats(b + "ssm_dt.bias", a + "dt_bias", {g.gdnValueHeads}, {g.gdnValueHeads});
        binder.floats(b + "ssm_norm.weight", a + "norm.weight", {g.gdnHeadDimension}, {g.gdnHeadDimension});
      }
    }
    binder.requireNoOthers();

    view.header.tensors = std::move(binder.header.tensors);
    view.header.valueRowsGrouped = true;
    readRotation(hadamard, binder.weightNames(), view.header);

    // The header and hadamard.json are read once; the tensor data is hashed
    // by the source.
    uint64_t headerBytes = 0;
    readWeightBytes(source.descriptor(), 0, {reinterpret_cast<uint8_t *>(&headerBytes), 8});
    if (headerBytes > source.bytes() - 8) fail("safetensors header exceeds the file");
    std::vector<uint8_t> header(8 + headerBytes);
    readWeightBytes(source.descriptor(), 0, header);
    view.identity = weightDigest(header) + weightDigest({static_cast<const uint8_t *>(hadamardBytes.bytes),
                                                         static_cast<size_t>(hadamardBytes.length)});

    const SourceTensor &first = checkpoint->require(language + "lm_head.weight");
    source.setDataOffset(first.file->dataOffset());
    auto synthesizer = std::make_shared<Synthesizer>(std::move(checkpoint), std::move(binder.tensors));
    source.setReader([synthesizer](uint64_t offset, std::span<uint8_t> out) { synthesizer->read(offset, out); });
    return view;
  }
}

std::string prismMlxKey(std::string_view plannedKey, std::string_view identity) {
  return weightDigest(std::string(plannedKey) + " splash-prism-mlx-v1 " SPLASH_PRISM_PREPARATION_ID " " +
                      std::string(identity));
}

} // namespace splash::model
