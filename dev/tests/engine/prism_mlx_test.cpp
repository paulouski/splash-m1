// A Prism ML Hadamard MLX checkpoint presented as its PQ2_0 GGUF: the
// synthesized blocks against an independent unpacking of the codes, the
// floats, the metadata and tensor checks, and the cache identity.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/PrismMlx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;

namespace {

constexpr uint32_t kHidden = 1024, kVocab = 4, kLayers = 4, kConv = 1536, kValueHeads = 8;
const std::string kLanguage = "language_model.";

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};

// One checkpoint's tensors and the independent record of its codes.
struct Fixture {
  std::map<std::string, Tensor> tensors;
  std::map<std::string, std::vector<uint8_t>> codes;    // module -> one code per value
  std::map<std::string, std::vector<uint16_t>> scales;  // module -> F16 bits
  std::vector<std::string> modules;
  std::map<std::string, std::vector<float>> floats;
  std::vector<int> signs = std::vector<int>(kHidden);
  uint32_t state = 12345;

  uint32_t next() { return state = state * 1664525u + 1013904223u; }

  template <class T> static std::vector<uint8_t> raw(const std::vector<T> &values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
  }

  void floatTensor(const std::string &name, std::vector<uint64_t> shape, bool negativeA = false) {
    uint64_t count = 1;
    for (uint64_t dim : shape) count *= dim;
    std::vector<float> values(count);
    for (float &value : values) value = (negativeA ? 0.5f : 0.0f) + float(next() % 100000) / 7919.0f;
    floats[name] = values;
    tensors[name] = {"F32", std::move(shape), raw(values)};
  }

  void module(const std::string &prefix, uint64_t rows, uint64_t columns) {
    modules.push_back(prefix);
    std::vector<uint8_t> code(rows * columns);
    for (uint8_t &c : code) c = uint8_t(next() >> 16) % 3;
    std::vector<uint32_t> words(rows * columns / 16);
    for (size_t i = 0; i < code.size(); ++i) words[i / 16] |= uint32_t(code[i]) << (2 * (i % 16));
    std::vector<uint16_t> scale(rows * columns / 128), bias;
    static constexpr uint16_t kScales[] = {0x3C00, 0x3800, 0x2E66, 0x4200, 0x1400};
    for (uint16_t &s : scale) s = kScales[(next() >> 16) % 5];
    for (uint16_t s : scale) bias.push_back(s ^ 0x8000);
    std::vector<float> sign(columns, 1.0f);
    for (uint64_t i = 0; i < columns; ++i) sign[i] = float(signs[i % kHidden]);
    codes[prefix] = code;
    scales[prefix] = scale;
    tensors[prefix + ".weight"] = {"U32", {rows, columns / 16}, raw(words)};
    tensors[prefix + ".scales"] = {"F16", {rows, columns / 128}, raw(scale)};
    tensors[prefix + ".biases"] = {"F16", {rows, columns / 128}, raw(bias)};
    tensors[prefix + ".signs"] = {"F32", {columns}, raw(sign)};
  }

  Fixture() {
    for (int &sign : signs) sign = (next() >> 20) & 1 ? 1 : -1;
    const std::string m = kLanguage + "model.";
    module(kLanguage + "lm_head", kVocab, kHidden);
    module(m + "embed_tokens", kVocab, kHidden);
    floatTensor(m + "norm.weight", {kHidden});
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      const std::string p = m + "layers." + std::to_string(layer) + ".";
      floatTensor(p + "input_layernorm.weight", {kHidden});
      floatTensor(p + "post_attention_layernorm.weight", {kHidden});
      module(p + "mlp.gate_proj", kHidden, kHidden);
      module(p + "mlp.up_proj", kHidden, kHidden);
      module(p + "mlp.down_proj", kHidden, kHidden);
      if (layer == kLayers - 1) {
        module(p + "self_attn.q_proj", 2 * kHidden, kHidden);
        module(p + "self_attn.k_proj", 128, kHidden);
        module(p + "self_attn.v_proj", 128, kHidden);
        module(p + "self_attn.o_proj", kHidden, kHidden);
        floatTensor(p + "self_attn.q_norm.weight", {128});
        floatTensor(p + "self_attn.k_norm.weight", {128});
        continue;
      }
      const std::string a = p + "linear_attn.";
      module(a + "in_proj_qkv", kConv, kHidden);
      module(a + "in_proj_z", 1024, kHidden);
      module(a + "out_proj", kHidden, 1024);
      floatTensor(a + "in_proj_b.weight", {kValueHeads, kHidden});
      floatTensor(a + "in_proj_a.weight", {kValueHeads, kHidden});
      floatTensor(a + "conv1d.weight", {kConv, 4, 1});
      floatTensor(a + "A_log", {kValueHeads}, true);
      floatTensor(a + "dt_bias", {kValueHeads});
      floatTensor(a + "norm.weight", {128});
    }
  }

  std::string config(int bits = 2, int group = 128, int schema = 2) const {
    std::string layers;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      layers += std::string(layer ? "," : "") + (layer == kLayers - 1 ? "\"full_attention\"" : "\"linear_attention\"");
    return "{\"schema_version\":" + std::to_string(schema) + ",\"model_type\":\"prism_hadamard_qwen35\","
           "\"quantization\":{\"bits\":" + std::to_string(bits) + ",\"group_size\":" + std::to_string(group) +
           ",\"mode\":\"affine\"},\"text_config\":{\"model_type\":\"qwen3_5_text\",\"tie_word_embeddings\":false,"
           "\"attn_output_gate\":true,\"head_dim\":128,\"hidden_size\":1024,\"intermediate_size\":1024,"
           "\"num_hidden_layers\":4,\"num_attention_heads\":8,\"num_key_value_heads\":1,\"full_attention_interval\":4,"
           "\"linear_conv_kernel_dim\":4,\"linear_key_head_dim\":128,\"linear_value_head_dim\":128,"
           "\"linear_num_key_heads\":2,\"linear_num_value_heads\":8,\"rms_norm_eps\":1e-06,"
           "\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25},"
           "\"layer_types\":[" + layers + "]}}";
  }

  std::string hadamard(const std::string &padding = "") const {
    std::string names, values;
    for (const std::string &name : modules)
      if (name != kLanguage + "model.embed_tokens") names += (names.empty() ? "" : ",") + ("\"" + name + ".weight\"");
    for (uint32_t i = 0; i < kHidden; ++i) values += std::string(i ? "," : "") + std::to_string(signs[i]);
    return "{\"prism.hadamard.version\":1,\"prism.hadamard.block_size\":1024,"
           "\"prism.hadamard.transform\":\"normalized-sylvester-walsh-hadamard\","
           "\"prism.hadamard.axis\":\"input-last-dimension\",\"prism.hadamard.sign_mode\":\"explicit\","
           "\"prism.hadamard.weight_names\":[" + names + "],"
           "\"prism.hadamard.inverse_weight_names\":[\"" + kLanguage + "model.embed_tokens.weight\"],"
           "\"prism.hadamard.sign_widths\":[1024],\"prism.hadamard.sign_values\":[" + values + "],"
           "\"prism.hadamard.gdn_v_grouped\":true}" + padding;
  }

  void write(const std::filesystem::path &directory, const std::string &configText,
             const std::string &hadamardText) const {
    std::filesystem::create_directories(directory);
    splash::test::writeFile(directory / "config.json", configText);
    splash::test::writeFile(directory / "hadamard.json", hadamardText);
    std::string header = "{";
    std::vector<uint8_t> data;
    for (const auto &[name, tensor] : tensors) {
      header += std::string(header.size() > 1 ? "," : "") + "\"" + name + "\":{\"dtype\":\"" + tensor.dtype +
                "\",\"shape\":[";
      for (size_t i = 0; i < tensor.shape.size(); ++i)
        header += std::string(i ? "," : "") + std::to_string(tensor.shape[i]);
      header += "],\"data_offsets\":[" + std::to_string(data.size()) + "," +
                std::to_string(data.size() + tensor.bytes.size()) + "]}";
      data.insert(data.end(), tensor.bytes.begin(), tensor.bytes.end());
    }
    header += "}";
    const uint64_t length = header.size();
    std::vector<uint8_t> file(8);
    std::memcpy(file.data(), &length, 8);
    file.insert(file.end(), header.begin(), header.end());
    file.insert(file.end(), data.begin(), data.end());
    splash::test::writeFile(directory / "model.safetensors", file);
  }

  void write(const std::filesystem::path &directory) const { write(directory, config(), hadamard()); }

  // The PQ2_0 blocks of a module's rows, from its codes alone: the group's F16
  // scale, then each byte holding four codes, the first in its low bits.
  std::vector<uint8_t> blocks(const std::string &prefix) const {
    const auto &code = codes.at(prefix);
    const auto &scale = scales.at(prefix);
    std::vector<uint8_t> out;
    for (size_t group = 0; group < scale.size(); ++group) {
      out.push_back(uint8_t(scale[group]));
      out.push_back(uint8_t(scale[group] >> 8));
      for (size_t byte = 0; byte < 32; ++byte) {
        uint8_t packed = 0;
        for (size_t i = 0; i < 4; ++i) packed |= code[group * 128 + byte * 4 + i] << (2 * i);
        out.push_back(packed);
      }
    }
    return out;
  }
};

splash::model::gguf::TargetGeometry geometry() {
  splash::model::gguf::TargetGeometry g;
  g.layers = kLayers;
  g.hiddenSize = kHidden;
  g.vocabularySize = kVocab;
  g.intermediateSize = 1024;
  g.gdnKeyHeads = 2;
  g.gdnValueHeads = kValueHeads;
  g.gdnHeadDimension = 128;
  g.convolutionDimension = kConv;
  g.attentionWidth = 1024;
  g.attentionKvHeads = 1;
  g.attentionHeadDimension = 128;
  g.rotaryPairs = 16;
  g.fullAttentionPeriod = 4;
  return g;
}

std::vector<uint8_t> readTensor(const WeightSource &source, const GgufFile &file, const std::string &name,
                                uint64_t from = 0, uint64_t count = UINT64_MAX) {
  const GgufTensor &tensor = file.require(name);
  count = std::min(count, tensor.bytes - from);
  std::vector<uint8_t> bytes(count);
  source.readData(tensor.offset + from, bytes);
  return bytes;
}

template <class F> void bound(const std::filesystem::path &directory, F use) {
  WeightSource source(prismMlxFile(directory));
  PrismMlxView view = bindPrismMlx(source, directory, geometry());
  const GgufFile file(source, std::move(view.header));
  use(source, file, view.identity);
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory scratch("splash-prism-mlx");
    const std::filesystem::path &root = scratch.path();
    setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
    Fixture fixture;
    const auto directory = root / "good";
    fixture.write(directory);
    require(isPrismMlx(directory), "a Prism config is not detected");

    bound(directory, [&](WeightSource &source, const GgufFile &file, const std::string &) {
      const auto &rotation = file.rotation();
      require(rotation && rotation->valueHeadsGrouped && file.valueRowsGrouped(), "rotation or grouped rows lost");
      require(rotation->weights.size() == fixture.modules.size() - 1 &&
                  rotation->tables == std::set<std::string, std::less<>>{"token_embd.weight"},
              "rotation names differ");
      require(file.require("token_embd.weight").type == ggml::kPQ2_0 &&
                  file.require("token_embd.weight").dims == std::vector<uint64_t>({kHidden, kVocab}),
              "token table type or shape differs");
      // Every projection's blocks equal the independent unpacking of its codes.
      const std::pair<const char *, std::string> mapped[] = {
          {"output.weight", kLanguage + "lm_head"},
          {"token_embd.weight", kLanguage + "model.embed_tokens"},
          {"blk.0.ffn_gate.weight", kLanguage + "model.layers.0.mlp.gate_proj"},
          {"blk.1.attn_qkv.weight", kLanguage + "model.layers.1.linear_attn.in_proj_qkv"},
          {"blk.1.attn_gate.weight", kLanguage + "model.layers.1.linear_attn.in_proj_z"},
          {"blk.2.ssm_out.weight", kLanguage + "model.layers.2.linear_attn.out_proj"},
          {"blk.3.attn_q.weight", kLanguage + "model.layers.3.self_attn.q_proj"},
          {"blk.3.attn_k.weight", kLanguage + "model.layers.3.self_attn.k_proj"},
          {"blk.3.attn_output.weight", kLanguage + "model.layers.3.self_attn.o_proj"}};
      for (const auto &[gguf, module] : mapped) {
        const auto expected = fixture.blocks(module);
        require(readTensor(source, file, gguf) == expected, std::string("synthesized blocks differ: ") + gguf);
        const uint64_t at = 17, count = expected.size() / 2;
        require(readTensor(source, file, gguf, at, count) ==
                    std::vector<uint8_t>(expected.begin() + at, expected.begin() + at + count),
                std::string("a partial read differs: ") + gguf);
      }
      // Floats keep their bits; ssm_a is -exp(A_log).
      const auto floats = [&](const std::string &gguf, const std::string &mlx) {
        const auto bytes = readTensor(source, file, gguf);
        require(bytes == Fixture::raw(fixture.floats.at(mlx)), "float tensor changed: " + gguf);
      };
      const std::string p = kLanguage + "model.layers.1.";
      floats("blk.1.attn_norm.weight", p + "input_layernorm.weight");
      floats("blk.1.post_attention_norm.weight", p + "post_attention_layernorm.weight");
      floats("blk.1.ssm_beta.weight", p + "linear_attn.in_proj_b.weight");
      floats("blk.1.ssm_alpha.weight", p + "linear_attn.in_proj_a.weight");
      floats("blk.1.ssm_conv1d.weight", p + "linear_attn.conv1d.weight");
      floats("blk.1.ssm_dt.bias", p + "linear_attn.dt_bias");
      floats("blk.1.ssm_norm.weight", p + "linear_attn.norm.weight");
      floats("output_norm.weight", kLanguage + "model.norm.weight");
      floats("blk.3.attn_q_norm.weight", kLanguage + "model.layers.3.self_attn.q_norm.weight");
      const auto decay = readTensor(source, file, "blk.1.ssm_a");
      std::vector<float> expected;
      for (float a : fixture.floats.at(p + "linear_attn.A_log")) expected.push_back(float(-std::exp(double(a))));
      require(decay == Fixture::raw(expected), "ssm_a is not -exp(A_log)");
      std::vector<uint8_t> past(4 * kValueHeads + 4);
      rejects([&] { source.readData(file.require("blk.1.ssm_a").offset, past); }, "read past",
              "a read past a tensor accepted");
    });

    // Metadata and tensor checks.
    const auto bad = [&](const std::string &name, const std::string &configText, const std::string &hadamardText,
                         const Fixture &source, std::string_view error) {
      source.write(root / name, configText, hadamardText);
      rejects([&] { bound(root / name, [](auto &, auto &, auto &) {}); }, error, name + " accepted");
    };
    bad("bits", fixture.config(4), fixture.hadamard(), fixture, "2-bit/group-128");
    bad("group", fixture.config(2, 64), fixture.hadamard(), fixture, "2-bit/group-128");
    bad("schema", fixture.config(2, 128, 1), fixture.hadamard(), fixture, "schema 2");
    {
      Fixture f = fixture;
      f.write(root / "nohadamard");
      std::filesystem::remove(root / "nohadamard" / "hadamard.json");
      rejects([&] { bound(root / "nohadamard", [](auto &, auto &, auto &) {}); }, "hadamard.json is missing",
              "a missing hadamard.json accepted");
    }
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.layers.0.mlp.up_proj.signs"].bytes[3] ^= 0x80;
      bad("signs", f.config(), f.hadamard(), f, ".signs differs from the rotation's signs");
    }
    {
      Fixture f = fixture;
      f.tensors.erase(kLanguage + "model.layers.0.mlp.up_proj.weight");
      bad("missing", f.config(), f.hadamard(), f, "missing tensor " + kLanguage + "model.layers.0.mlp.up_proj.weight");
    }
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.extra.weight"] = {"F32", {1}, std::vector<uint8_t>(4)};
      bad("extra", f.config(), f.hadamard(), f, "unexpected tensor " + kLanguage + "model.extra.weight");
    }
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.layers.0.mlp.up_proj.scales"].dtype = "BF16";
      bad("dtype", f.config(), f.hadamard(), f, "unexpected dtype or shape");
    }
    {
      // The vision tower is not read.
      Fixture f = fixture;
      f.tensors["vision_tower.patch_embed.weight"] = {"F32", {1}, std::vector<uint8_t>(4)};
      f.write(root / "vision");
      bound(root / "vision", [](auto &, auto &, auto &) {});
    }
    const auto failsWhenRead = [&](const std::string &name, Fixture &f, std::string_view error) {
      f.write(root / name);
      rejects([&] {
        bound(root / name, [&](WeightSource &source, const GgufFile &file, const std::string &) {
          (void)readTensor(source, file, "blk.0.ffn_up.weight");
        });
      }, error, name + " accepted");
    };
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.layers.0.mlp.up_proj.biases"].bytes[2] ^= 1;
      failsWhenRead("bias", f, "biases other than -scales");
    }
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.layers.0.mlp.up_proj.weight"].bytes[5] |= 0xC0;
      failsWhenRead("code3", f, "unused code 3");
    }

    // The cache identity follows every source byte and hadamard.json.
    const auto identity = [&](const std::filesystem::path &path) {
      std::string key;
      bound(path, [&](WeightSource &source, const GgufFile &, const std::string &id) {
        key = prismMlxKey(source.digest(), id);
      });
      return key;
    };
    const std::string original = identity(directory);
    require(identity(directory) == original, "the identity is not stable");
    {
      Fixture f = fixture;
      f.tensors[kLanguage + "model.layers.1.mlp.down_proj.weight"].bytes[3] ^= 1;
      f.write(root / "byte");
      require(identity(root / "byte") != original, "a changed source byte kept the identity");
    }
    {
      Fixture f = fixture;
      f.write(root / "padded", f.config(), f.hadamard(" "));
      require(identity(root / "padded") != original, "a changed hadamard.json kept the identity");
    }
    {
      Fixture f = fixture;
      f.signs[0] = -f.signs[0];
      for (auto &[name, tensor] : f.tensors)
        if (name.ends_with(".signs")) tensor.bytes[3] ^= 0x80; // first value's sign bit, little endian
      f.write(root / "sign");
      require(identity(root / "sign") != original, "a changed hadamard sign kept the identity");
    }
    std::cout << "prism-mlx ok\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "prism-mlx FAILED: " << error.what() << '\n';
    return 1;
  }
}
