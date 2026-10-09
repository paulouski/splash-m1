// The GGUF image planner on the small dense and MoE targets and a rotated
// dense one: where each tensor goes and in which row order, which F32 tensors
// it narrows to bf16, and the sources it refuses.
//   gguf-planner
#include "GgufFixtures.hpp"
#include "metal/abi/Gguf.h"
#include "model/GgufTarget.hpp"
#include "model/Qwen4Exp.hpp"

#include <deque>
#include <functional>
#include <map>
#include <optional>

using namespace gguf_fixtures;

namespace {

// The images of the GGUF at path, planned for geometry; the planner's error
// when it refuses the file.
struct Plan {
  std::vector<model::gguf::Image> images;
  std::optional<std::string> error;
};

Plan plan(const std::filesystem::path &path, const model::gguf::TargetGeometry &geometry) {
  try {
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    return {model::gguf::planImages(gguf, geometry), std::nullopt};
  } catch (const model::GgufError &error) {
    return {{}, error.what()};
  }
}

// Whether a refusal names every part.
bool names(const Plan &result, std::initializer_list<std::string_view> parts) {
  return result.error && std::all_of(parts.begin(), parts.end(), [&](std::string_view part) {
           return result.error->find(part) != std::string::npos;
         });
}

// The plan's copy of the named tensor in any image, or nullptr.
const model::gguf::Copy *copyIn(const Plan &result, const std::string &name) {
  for (const model::gguf::Image &image : result.images)
    if (const auto *copy = copyOf(image, name)) return copy;
  return nullptr;
}

// Value heads regrouped from llama.cpp's tiled order from row `from` on.
bool grouped(const model::gguf::RowOrder &order, uint64_t from, uint32_t headRows,
             const model::gguf::TargetGeometry &g) {
  return order.from == from && order.headRows == headRows && order.keyHeads == g.gdnKeyHeads &&
         order.valueHeadsPerKey == g.gdnValueHeads / g.gdnKeyHeads;
}

// A copy of the tensor's rows as the GGUF stores them.
bool asStored(const model::gguf::Copy *copy, const test_gguf::Bytes &data) {
  return copy && copy->conversion == model::gguf::Conversion::None && copy->source.order.from == UINT64_MAX &&
         copy->source.rows * copy->source.rowBytes == data.size();
}

void checkDense(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(false);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const Plan dense = plan(path, g);
  check(!dense.error, "planner plans the dense target" + (dense.error ? ": " + *dense.error : ""));
  if (dense.error) return;
  const model::gguf::Image &gdn = dense.images[0];

  for (const Tensor &tensor : target.tensors)
    if (isNorm(tensor.name))
      check(asStored(copyIn(dense, tensor.name), tensor.data), "planner keeps F32 values as stored: " + tensor.name);

  const uint32_t keyRows = g.convolutionDimension - g.gdnValueHeads * g.gdnHeadDimension;
  const auto *convolution = copyOf(gdn, "blk.0.ssm_conv1d.weight");
  check(convolution && convolution->conversion == model::gguf::Conversion::NarrowToBfloat16 &&
            convolution->source.rows == g.convolutionDimension &&
            grouped(convolution->source.order, keyRows, g.gdnHeadDimension, g),
        "planner narrows the convolution to bf16 and groups its value heads");
  const auto *decay = copyOf(gdn, "blk.0.ssm_a");
  check(decay && decay->conversion == model::gguf::Conversion::None && grouped(decay->source.order, 0, 1, g),
        "planner keeps the decay F32 and groups its value heads");
  const auto *bias = copyOf(gdn, "blk.0.ssm_dt.bias");
  check(bias && bias->conversion == model::gguf::Conversion::NarrowToBfloat16 && grouped(bias->source.order, 0, 1, g),
        "planner narrows the time bias to bf16 and groups its value heads");

  // A key or value projection of other than attentionKvHeads heads.
  for (const std::string name : {"blk.1.attn_k.weight", "blk.1.attn_v.weight"}) {
    for (uint64_t rows : {uint64_t{g.attentionHeadDimension}, uint64_t{3} * g.attentionHeadDimension}) {
      std::vector<Tensor> malformed = target.tensors;
      Tensor &t = tensorNamed(malformed, name);
      t.dims[1] = rows;
      t.data.resize(rows * gguf_reference::rowBytes(Fmt(gguf_format_of(t.type)), g.hiddenSize));
      writeGguf(path, malformed, g);
      check(names(plan(path, g), {"shape", name}),
            "planner refuses mismatched KV rows by name: " + name + " rows=" + std::to_string(rows));
    }
  }
}

// alpha/beta of any one quantized format, on either architecture: one 256-row
// tensor of that format, the beta then the alpha rows at its native row bytes
// in grouped head order; a pair of two types is refused by both names.
void checkQuantizedAlphaBeta(const std::filesystem::path &directory) {
  for (bool moe : {false, true}) {
    const SmallTarget target = smallTarget(moe);
    const model::gguf::TargetGeometry &g = target.geometry;
    const auto path = directory / "alpha-beta.gguf";
    const auto tensorsWith = [&](uint32_t betaType, uint32_t alphaType) {
      std::vector<Tensor> tensors = target.tensors;
      for (const char *name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        Tensor &tensor = tensorNamed(tensors, name);
        tensor.type = tensor.name.ends_with("_beta.weight") ? betaType : alphaType;
        const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(tensor.type);
        tensor.data.resize(uint64_t{g.gdnValueHeads} * g.hiddenSize / traits.blockElements * traits.blockBytes);
      }
      return tensors;
    };
    for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
      const uint32_t type = kQuantFormats[format].ggml_type;
      writeGguf(path, tensorsWith(type, type), g);
      const Plan result = plan(path, g);
      const auto *pair = result.error ? nullptr : repackOf(result.images[0], "blk.0.ssm_beta.weight");
      const auto heads = [&](const model::gguf::TensorRows &rows, const char *name) {
        return rows.name == name && rows.type == type && rows.rows == g.gdnValueHeads &&
               rows.rowBytes == gguf_reference::rowBytes(Fmt(format), g.hiddenSize) && grouped(rows.order, 0, 1, g);
      };
      check(pair && pair->format == format && pair->rows == QUANT_TILE_ROWS && pair->columns == g.hiddenSize &&
                pair->sources.size() == 2 && heads(pair->sources[0], "blk.0.ssm_beta.weight") &&
                heads(pair->sources[1], "blk.0.ssm_alpha.weight"),
            "planner alpha/beta: one 256-row " + model::ggmlTypeName(type) + " tensor of beta then alpha rows (" +
                g.architecture() + ")" + (result.error ? ": " + *result.error : ""));
    }
    for (const auto &[beta, alpha] : {std::pair{kIQ4_XS, kQ8_0}, std::pair{kQ4_K, model::ggml::kF32}}) {
      writeGguf(path, tensorsWith(beta, alpha), g);
      const std::string betaName = "blk.0.ssm_beta.weight (" + model::ggmlTypeName(beta) + ")";
      const std::string alphaName = "blk.0.ssm_alpha.weight (" + model::ggmlTypeName(alpha) + ")";
      check(names(plan(path, g), {betaName, alphaName}),
            "planner refuses alpha/beta of two types by both names: " + betaName + ", " + alphaName);
    }
  }
}

void checkMoe(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "moe.gguf";

  model::gguf::TargetGeometry wrongExperts = g;
  wrongExperts.experts = 5;
  writeGguf(path, target.tensors, wrongExperts);
  check(names(plan(path, g), {"expert_count"}), "planner names a metadata mismatch");
  model::gguf::TargetGeometry dense = g;
  dense.experts = 0;
  dense.intermediateSize = g.expertIntermediateSize;
  writeGguf(path, target.tensors, dense);
  check(names(plan(path, g), {"architecture", dense.architecture(), g.architecture()}),
        "planner checks the architecture against the package");

  writeGguf(path, target.tensors, g);
  const Plan moe = plan(path, g);
  check(!moe.error, "planner plans the MoE target" + (moe.error ? ": " + *moe.error : ""));
  if (moe.error) return;
  const model::gguf::Image &layer = moe.images[0];
  const auto *beta = copyOf(layer, "blk.0.ssm_beta.weight"), *alpha = copyOf(layer, "blk.0.ssm_alpha.weight");
  check(beta && alpha && alpha->destination == beta->destination + target.data(beta->source.name).size() &&
            grouped(beta->source.order, 0, 1, g) && grouped(alpha->source.order, 0, 1, g) &&
            beta->conversion == model::gguf::Conversion::None && alpha->conversion == model::gguf::Conversion::None,
        "planner F32 alpha/beta tensor: beta then alpha rows in grouped order");
  for (const std::string name : {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight"})
    check(asStored(copyOf(layer, name), target.data(name)), "planner copies the F32 tensor as stored: " + name);

  // Each quantized tensor is repacked on its own, a 3-D expert tensor as one
  // tensor of experts * N rows.
  size_t quantized = 0;
  for (const Tensor &tensor : target.tensors) {
    if (!tensor.name.starts_with("blk.0.") || gguf_format_of(tensor.type) == GGUF_FMT_COUNT) continue;
    ++quantized;
    const uint64_t rows = tensor.dims[1] * (tensor.dims.size() > 2 ? tensor.dims[2] : 1);
    const auto *repack = repackOf(layer, tensor.name);
    check(repack && repack->sources.size() == 1 && repack->rows == rows && repack->columns == tensor.dims[0] &&
              repack->sources[0].rows == rows,
          "planner repacks " + tensor.name + " as " + std::to_string(rows) + " rows of " +
              std::to_string(tensor.dims[0]));
  }
  check(layer.repacks.size() == quantized, "planner repacks only the layer's quantized tensors");
}

// The rotary embedding and norms the kernels compute: a GGUF declares their
// RoPE base, rotated dimensions and RMS epsilon, and no RoPE scaling.
void checkRotary(const std::filesystem::path &directory) {
  const SmallTarget target = smallTarget(false);
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "rotary.gguf";
  const auto planned = [&](const std::vector<test_gguf::Key> &keys) {
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    return plan(path, g);
  };
  const std::string scaling = g.architecture() + std::string(".rope.scaling.type");
  std::vector<test_gguf::Key> keys = metadata(g);
  keys.push_back(test_gguf::stringKey(scaling, "none"));
  check(!planned(keys).error, "planner accepts a GGUF that declares no RoPE scaling");
  keys.back() = test_gguf::stringKey(scaling, "yarn");
  check(names(planned(keys), {"rope.scaling.type yarn (expected none)"}), "planner refuses RoPE scaling");

  model::gguf::TargetGeometry other = g;
  other.rotaryPairs = 64;
  other.rotaryTheta = 1e6F;
  keys = metadata(other);
  for (test_gguf::Key &key : keys)
    if (key.name.ends_with(".attention.layer_norm_rms_epsilon")) key = test_gguf::float32Key(key.name, 1e-5F);
  check(names(planned(keys), {"rope.dimension_count 128 (expected 64)", "rope.freq_base 1e+06 (expected 1e+07)",
                              "attention.layer_norm_rms_epsilon 1e-05 (expected 1e-06)"}),
        "planner names another rotated width, RoPE base and RMS epsilon");

  keys = metadata(g);
  std::erase_if(keys, [](const test_gguf::Key &key) {
    return key.name.find(".rope.") != std::string::npos || key.name.ends_with(".layer_norm_rms_epsilon");
  });
  check(names(planned(keys), {"rope.dimension_count missing", "rope.freq_base missing",
                              "attention.layer_norm_rms_epsilon missing"}),
        "planner requires the rotary and norm metadata");
}

// A dense target stored for rotated inputs, every rotated input one rotation
// block wide: the rotation must name exactly the tensors the planner repacks,
// so quantized alpha/beta (Q8_0 and IQ4_XS here), whose segment reads the
// rotated input, and not F32 ones.
void checkRotation(const std::filesystem::path &directory) {
  model::gguf::TargetGeometry g;
  g.layers = 2;
  g.hiddenSize = GGUF_ROTATION_BLOCK;
  g.vocabularySize = 256;
  g.intermediateSize = GGUF_ROTATION_BLOCK;
  g.gdnKeyHeads = 4;
  g.gdnValueHeads = 16;
  g.gdnHeadDimension = 64;
  g.convolutionDimension = 1536; // q and k of 4 heads, v of 16
  g.attentionWidth = GGUF_ROTATION_BLOCK; // four query heads of 256
  g.attentionKvHeads = 1;
  g.attentionHeadDimension = 256;
  g.rotaryPairs = 32;
  g.rotaryTheta = 1e7F;
  g.fullAttentionPeriod = 2; // layer 1
  const auto path = directory / "rotated.gguf";
  // PQ2_0 projections and token table, alpha/beta of type `alphaBeta`, and a
  // rotation that names every quantized tensor but the table, alpha/beta only
  // when `namesAlphaBeta`.
  const auto planned = [&](uint32_t alphaBeta, bool namesAlphaBeta) {
    TensorTypes types{{"ssm_beta.weight", alphaBeta}, {"ssm_alpha.weight", alphaBeta}};
    for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                             "attn_gate.weight", "ssm_out.weight", "ffn_gate.weight", "ffn_up.weight",
                             "ffn_down.weight", "output.weight", "token_embd.weight"})
      types[name] = model::ggml::kPQ2_0;
    const std::vector<Tensor> tensors = targetTensors(g, types);
    std::vector<std::string> weights;
    for (const Tensor &tensor : tensors)
      if (gguf_format_of(tensor.type) != GGUF_FMT_COUNT && tensor.name != "token_embd.weight" &&
          (namesAlphaBeta || !(tensor.name.ends_with("_alpha.weight") || tensor.name.ends_with("_beta.weight"))))
        weights.push_back(tensor.name);
    std::vector<test_gguf::Key> keys = metadata(g);
    const auto key = [](const char *name) { return std::string("prism.hadamard.") + name; };
    keys.push_back(test_gguf::uint32Key(key("version"), 1));
    keys.push_back(test_gguf::uint32Key(key("block_size"), GGUF_ROTATION_BLOCK));
    keys.push_back(test_gguf::stringKey(key("transform"), "normalized-sylvester-walsh-hadamard"));
    keys.push_back(test_gguf::stringKey(key("axis"), "input-last-dimension"));
    keys.push_back(test_gguf::stringKey(key("sign_mode"), "explicit"));
    keys.push_back(test_gguf::boolKey(key("gdn_v_grouped"), true));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_widths"), {GGUF_ROTATION_BLOCK}));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_values"), std::vector<int32_t>(GGUF_ROTATION_BLOCK, 1)));
    keys.push_back(test_gguf::stringArrayKey(key("weight_names"), weights));
    keys.push_back(test_gguf::stringArrayKey(key("inverse_weight_names"), {"token_embd.weight"}));
    splash::test::writeFile(path, test_gguf::file(keys, tensors));
    return plan(path, g);
  };
  const Plan floats = planned(model::ggml::kF32, false);
  check(!floats.error, "planner plans a rotation of F32 alpha/beta" + (floats.error ? ": " + *floats.error : ""));
  for (uint32_t type : {kQ8_0, kIQ4_XS}) {
    const std::string gates = model::ggmlTypeName(type) + " alpha/beta";
    const Plan named = planned(type, true);
    check(!named.error, "planner plans a rotation that names " + gates + (named.error ? ": " + *named.error : ""));
    check(names(planned(type, false), {"the rotation must name every quantized tensor"}),
          "planner refuses a rotation that leaves out " + gates);
  }
}

// Qwen3.8-Flash-Next (qwen4exp): widths that are no whole plane tile (640-row
// shared expert, 320-row hyper-connection gate) or superblock (K 320, 640), a
// PLE table that gets no image, F32/BF16/Q8_0 float tensors, and a split GGUF.
void checkQwen4(const std::filesystem::path &directory) {
  SmallTarget target = smallQwen4Target();
  const model::gguf::TargetGeometry &g = target.geometry;
  const auto path = directory / "qwen4.gguf";
  writeGguf(path, target.tensors, g);
  const Plan plan4 = plan(path, g);
  check(!plan4.error, "planner plans the qwen4exp target" + (plan4.error ? ": " + *plan4.error : ""));
  if (plan4.error) return;
  check(plan4.images.size() == g.layers + 2, "planner plans the layers, head and embedding: the PLE table has no image");
  check(copyIn(plan4, "per_layer_token_embd.weight") == nullptr, "planner copies no PLE table rows");
  const auto &layer0 = plan4.images[0];
  // Rows to whole plane tiles, the source keeping its own rows.
  const auto *shared = repackOf(layer0, "blk.0.ffn_gate_shexp.weight");
  check(shared && shared->rows == 768 && shared->columns == g.hiddenSize && shared->sources.size() == 1 &&
            shared->sources[0].rows == 640 && shared->format == GGUF_FMT_Q5K,
        "planner pads the 640-row shared expert to 768 image rows");
  const auto *down = repackOf(layer0, "blk.0.ffn_down_shexp.weight");
  check(down && down->rows == QUANT_TILE_ROWS && down->columns == 640 && down->format == GGUF_FMT_Q80,
        "planner takes K 640 of Q8_0 as a 64-column unit");
  const auto *gate = repackOf(layer0, "blk.0.hc_attn_down.weight");
  const auto *up = repackOf(layer0, "blk.0.hc_attn_up.weight");
  check(gate && gate->rows == 512 && gate->sources[0].rows == 320 && gate->columns == g.streamWidth(),
        "planner pads the 320-row hyper-connection gate to 512 image rows");
  check(up && up->rows == g.streamWidth() && up->columns == 320, "planner takes K 320 of Q8_0 as a 64-column unit");
  const auto *experts = repackOf(layer0, "blk.0.ffn_down_exps.weight");
  check(experts && experts->rows == uint64_t{g.hiddenSize} * g.experts && experts->columns == 640 &&
            experts->format == GGUF_FMT_IQ4NL,
        "planner keeps the experts' image rows whole (640-wide experts)");
  const auto *inject = copyIn(plan4, "blk.0.hc_attn_inject.weight");
  check(inject && inject->conversion == model::gguf::Conversion::None && inject->source.rows == g.hyperConnections,
        "planner keeps the F32 injection weights as stored");
  const auto *indexer = copyOf(plan4.images[g.layers - 1], "blk.3.indexer.q_proj.weight");
  check(indexer && indexer->conversion == model::gguf::Conversion::WidenToFloat32 &&
            indexer->source.rows == uint64_t{g.indexerHeads} * g.indexerHeadDimension,
        "planner widens the BF16 indexer projection to F32");
  const auto &layer1 = plan4.images[1];
  check(repackOf(layer1, "blk.1.ple_key.weight") && copyOf(layer1, "blk.1.ple_conv1d.weight") &&
            !repackOf(plan4.images[0], "blk.0.ple_key.weight"),
        "planner puts the PLE layer's projections and taps in that layer");
  check(plan4.images[g.layers].name == "head.bin" && repackOf(plan4.images[g.layers], "output_hc_up.weight"),
        "planner builds the head from the final hyper-connection mix, with no output norm");

  // The PLE table is read in place: where its rows are and the hash that picks them.
  model::WeightSource source(path);
  const model::GgufFile file(source);
  const auto table = model::gguf::readPleTable(file, g);
  check(table && table->path == path && table->dataOffset == source.dataOffset() + file.require("per_layer_token_embd.weight").offset &&
            table->rows == 420 && table->hash.ngram == 3 && table->hash.headsPerNgram == 2 && table->hash.eos == 248044 &&
            table->hash.multipliers[0] == 23703573157769ull && table->hash.multipliers[2] == 8052911324071ull &&
            table->hash.heads() == 4 && table->hash.vocabularies[3] == 109 && table->hash.offsets[3] == 311,
        "planner reads the PLE table's place and hash");
  check(!model::gguf::readPleTable(file, smallTarget(false).geometry), "a target without PLE has no PLE table");

  // Metadata a qwen4exp GGUF must carry.
  const auto refuses = [&](const char *what, const std::function<void(std::vector<test_gguf::Key> &)> &edit,
                           std::initializer_list<std::string_view> parts) {
    std::vector<test_gguf::Key> keys = metadata(g);
    qwen4Metadata(keys, g);
    edit(keys);
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    check(names(plan(path, g), parts), std::string("planner refuses qwen4exp metadata: ") + what);
  };
  const auto replace = [&](std::vector<test_gguf::Key> &keys, const std::string &name, test_gguf::Key key) {
    for (test_gguf::Key &k : keys)
      if (k.name == "qwen4exp." + name) k = std::move(key);
  };
  refuses("hyper rank", [&](auto &keys) { replace(keys, "hyper_connection.low_rank", test_gguf::uint32Key("qwen4exp.hyper_connection.low_rank", 64)); },
          {"hyper_connection.low_rank 64", "expected 320"});
  refuses("compress ratios", [&](auto &keys) { replace(keys, "attention.compress_ratios", test_gguf::int32ArrayKey("qwen4exp.attention.compress_ratios", {0, 0, 4, 0})); },
          {"attention.compress_ratios"});
  refuses("rope sections", [&](auto &keys) { replace(keys, "rope.dimension_sections", test_gguf::int32ArrayKey("qwen4exp.rope.dimension_sections", {16, 8, 8, 0})); },
          {"rope.dimension_sections"});
  refuses("PLE layer", [&](auto &keys) { replace(keys, "ple.layers", test_gguf::int32ArrayKey("qwen4exp.ple.layers", {2})); },
          {"ple.layers"});
  refuses("indexer top k", [&](auto &keys) { replace(keys, "attention.indexer.top_k", test_gguf::uint32Key("qwen4exp.attention.indexer.top_k", 4)); },
          {"indexer.top_k 4"});
  {
    std::vector<test_gguf::Key> keys = metadata(g);
    qwen4Metadata(keys, g);
    replace(keys, "ple.head_vocab_sizes", test_gguf::uint64ArrayKey("qwen4exp.ple.head_vocab_sizes", {101, 103, 107, 1000}));
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    model::WeightSource bad(path);
    std::string error;
    try {
      static_cast<void>(model::gguf::readPleTable(model::GgufFile(bad), g));
    } catch (const model::GgufError &e) {
      error = e.what();
    }
    check(error.find("rows lie outside") != std::string::npos,
          "planner refuses a PLE head whose rows leave the table");
  }

  // The same target as a two-file split GGUF plans the same images, its second
  // file's tensors read from it.
  std::vector<Tensor> first(target.tensors.begin(), target.tensors.begin() + 40), second(target.tensors.begin() + 40, target.tensors.end());
  std::vector<test_gguf::Key> keys = metadata(g);
  qwen4Metadata(keys, g);
  const auto split = [&](uint32_t no) {
    std::vector<test_gguf::Key> k = no == 0 ? keys : std::vector<test_gguf::Key>{};
    k.push_back(test_gguf::uint32Key("split.no", no));
    k.push_back(test_gguf::uint32Key("split.count", 2));
    k.push_back(test_gguf::uint32Key("split.tensors.count", uint32_t(target.tensors.size())));
    return k;
  };
  const auto part0 = directory / "qwen4-00001-of-00002.gguf", part1 = directory / "qwen4-00002-of-00002.gguf";
  splash::test::writeFile(part0, test_gguf::file(split(0), first));
  splash::test::writeFile(part1, test_gguf::file(split(1), second));
  model::WeightSource s0(part0), s1(part1);
  model::WeightSource *parts[] = {&s0, &s1};
  const model::GgufFile splitFile(parts);
  const std::vector<model::gguf::Image> images = model::gguf::planImages(splitFile, g);
  bool same = images.size() == plan4.images.size();
  for (size_t i = 0; same && i < images.size(); ++i)
    same = images[i].bytes == plan4.images[i].bytes && images[i].repacks.size() == plan4.images[i].repacks.size() &&
           images[i].copies.size() == plan4.images[i].copies.size();
  check(same, "planner plans a split GGUF like the whole one");
  const auto *fromSecond = repackOf(images[3], "blk.3.ffn_down_exps.weight");
  const auto *fromFirst = repackOf(images[0], "blk.0.ffn_down_exps.weight");
  check(fromFirst && fromSecond && fromFirst->sources[0].file == 0 && fromSecond->sources[0].file == 1,
        "planner names the file each tensor is read from");
  const auto pleSplit = model::gguf::readPleTable(splitFile, g);
  check(pleSplit && pleSplit->path == part1, "planner finds the PLE table in its shard of a split GGUF");
}

// With SPLASH_FLASH_NEXT_GGUF_DIR naming a Qwen3.8-Flash-Next GGUF directory (its split parts, and mtp/*.gguf for
// the MTP head): plans every image of the real shards from their headers alone, reads no weight, and prints the
// planned bytes against the Metal working set.
void checkRealShards() {
  const char *root = std::getenv("SPLASH_FLASH_NEXT_GGUF_DIR");
  if (!root) return;
  const model::Qwen4ExpLayout layout;
  const model::gguf::TargetGeometry g = model::ggufTargetGeometry(layout);
  std::deque<model::WeightSource> sources;
  std::vector<model::WeightSource *> parts;
  const std::vector<std::filesystem::path> paths = model::findTargetGgufs(root);
  for (const auto &path : paths) parts.push_back(&sources.emplace_back(path));
  const model::GgufFile file(parts);
  const std::vector<model::gguf::Image> images = model::gguf::planImages(file, g);
  check(images.size() == layout.layers + 2, "real shards: planned " + std::to_string(paths.size()) + " files, " +
                                                std::to_string(file.tensors().size()) + " tensors, " +
                                                std::to_string(images.size()) + " images");
  const auto table = model::gguf::readPleTable(file, g);
  check(table && table->rows == 320001536 && table->hash.heads() == 16 && table->hash.eos == layout.pleEosToken,
        "real shards: the PLE table's place and hash");
  std::map<std::string, uint64_t> planes; // planned plane bytes by ggml type
  uint64_t layersBytes = 0, padding = 0, total = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    total += images[i].bytes;
    if (i < layout.layers) layersBytes += images[i].bytes;
    for (const auto &repack : images[i].repacks) {
      const model::GgufPlaneBytes bytes = model::ggufPlaneBytes(kQuantFormats[repack.format], repack.rows, repack.columns);
      const uint64_t all = bytes.plane0 + bytes.plane1 + bytes.meta;
      planes[model::ggmlTypeName(kQuantFormats[repack.format].ggml_type)] += all;
      uint64_t sourceRows = 0;
      for (const auto &source : repack.sources) sourceRows += source.rows;
      padding += all / repack.rows * (repack.rows - sourceRows);
    }
  }
  const auto gib = [](uint64_t bytes) { return double(bytes) / (1ull << 30); };
  std::printf("  %-40s %9s\n", "planned image bytes", "GiB");
  std::printf("  %-40s %9.3f\n", "layers 0-47", gib(layersBytes));
  std::printf("  %-40s %9.3f\n", "head (lm head and final mix)", gib(images[layout.layers].bytes));
  std::printf("  %-40s %9.3f\n", "embedding (native Q4_K rows)", gib(images[layout.layers + 1].bytes));
  std::printf("  %-40s %9.3f\n", "GPU-resident total, without PLE", gib(total));
  for (const auto &[type, bytes] : planes) std::printf("    plane bytes of %-22s %9.3f\n", type.c_str(), gib(bytes));
  std::printf("    zero rows padding the image tiles      %9.3f\n", gib(padding));
  std::printf("  %-40s %9.3f  (read in place from its shard)\n", "PLE table",
              gib(file.require("per_layer_token_embd.weight").bytes));
  std::printf("  %-40s %9.3f  (48 GiB Metal working set of the M1 Max reference)\n", "headroom", 48.0 - gib(total));
  check(total < (48ull << 30), "real shards: the planned images fit the 48 GiB Metal working set");
  const std::filesystem::path mtpDirectory = std::filesystem::path(root) / "mtp";
  if (!std::filesystem::exists(mtpDirectory)) return;
  model::WeightSource mtpSource(model::findTargetGgufs(mtpDirectory).front());
  const model::GgufFile mtpFile(mtpSource);
  const model::gguf::Image mtp = model::gguf::planMtpImage(mtpFile, g);
  std::printf("  %-40s %9.3f  (extra image, loaded with ProposalSource Mtp)\n", "MTP head", gib(mtp.bytes));
  check(total + mtp.bytes < (48ull << 30), "real shards: the target and the MTP head fit the 48 GiB Metal working set");
}

} // namespace

int main() {
  @autoreleasepool {
    const splash::test::TemporaryDirectory directory("splash-gguf-planner");
    guarded("planner on the dense target", [&] { checkDense(directory.path()); });
    guarded("planner on the MoE target", [&] { checkMoe(directory.path()); });
    guarded("planner on quantized alpha/beta", [&] { checkQuantizedAlphaBeta(directory.path()); });
    guarded("planner on the rotary metadata", [&] { checkRotary(directory.path()); });
    guarded("planner on the qwen4exp target", [&] { checkQwen4(directory.path()); });
    guarded("planner on the real Flash-Next shards", [&] { checkRealShards(); });
    guarded("planner on a rotated target", [&] { checkRotation(directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF planner tests FAILED" : "GGUF planner tests passed", failures);
    return failures ? 1 : 0;
  }
}
