// The dispatches ops::Linear::add encodes for a fixed set of projections, compared byte for byte with
// dev/tests/fixtures/linear-graph-dump.txt: pipeline names, grids, buffer bindings (label, view offset, size), parameter
// bytes, dispatch counters and the returned prepared input. Nothing is submitted: the graph is only read. The Linear
// sees hand-built device descriptions, so the dump does not depend on the Mac that runs it (it does depend on the
// build's compile-time device floor, so the fixture is the default macOS 15 build's).
//
// A change that moves the encoding on purpose regenerates the fixture: linear-graph-dump <metallib> <fixture> --write.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

namespace {

constexpr uint32_t kFloat = QuantizedSegment::kFloat32;
constexpr uint32_t kLane = 8;

// A harness bug (an unlabeled binding, a full arena), never a recorded outcome.
struct Unlabeled : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Every buffer of a case under a label, so a binding prints as "label+offset:size".
class Labels {
public:
  Labels(MetalBackend &backend) : backend_(backend) {}
  // A buffer of its own.
  MetalBuffer own(const std::string &label, uint64_t bytes) {
    if (!bytes) return {};
    MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    entries_.push_back({label, buffer, 0});
    return buffer;
  }
  // A view of the case's weights arena.
  MetalBuffer weights(const std::string &label, uint64_t bytes) {
    if (!bytes) return {};
    if (!arena_) arena_ = backend_.allocateBuffer(kArenaBytes, metal::BufferStorage::Shared, "weights-arena");
    const uint64_t offset = used_;
    used_ += (bytes + 255) / 256 * 256;
    if (used_ > kArenaBytes) throw Unlabeled("weights arena is full");
    MetalBuffer view = backend_.view(arena_, offset, bytes);
    entries_.push_back({label, view, offset});
    return view;
  }
  [[nodiscard]] std::string name(const MetalBuffer &buffer) const {
    if (!buffer) return "-";
    for (const auto &[label, view, offset] : entries_)
      if (view.sameView(buffer))
        return label + "+" + std::to_string(offset) + ":" + std::to_string(buffer.sizeBytes());
    throw Unlabeled("a dispatch binds a buffer the case did not label");
  }

private:
  static constexpr uint64_t kArenaBytes = uint64_t{256} << 20;
  MetalBackend &backend_;
  MetalBuffer arena_;
  uint64_t used_ = 0;
  std::vector<std::tuple<std::string, MetalBuffer, uint64_t>> entries_;
};

std::string hex(const void *data, uint64_t size) {
  static const char digits[] = "0123456789abcdef";
  std::string text;
  for (uint64_t i = 0; i < size; ++i) {
    const uint8_t byte = static_cast<const uint8_t *>(data)[i];
    text += digits[byte >> 4];
    text += digits[byte & 15];
  }
  return text;
}

void dumpGraph(std::ostream &out, const CommandGraph &graph, const Labels &labels) {
  int index = 0;
  for (const metal::ComputeDispatch &d : graph.dispatches()) {
    out << " d" << index++ << ' ' << d.pipelineName << " g=" << d.threadgroups.x << ',' << d.threadgroups.y << ','
        << d.threadgroups.z << " t=" << d.threadsPerThreadgroup.x << ',' << d.threadsPerThreadgroup.y << ','
        << d.threadsPerThreadgroup.z;
    for (const metal::BufferBinding &b : d.buffers) out << " b" << b.index << '=' << labels.name(b.buffer);
    for (const metal::BytesBinding &b : d.bytes) out << " p" << b.index << '=' << hex(b.data, b.sizeBytes);
    out << '\n';
  }
}

struct Seg {
  uint32_t format;
  uint32_t columns;
};

// A block (GGUF) case: the projection's segments over `k` inputs in order, an optional gate of the up's columns.
struct Block {
  std::string name;
  uint32_t family = 7, cores = 24;
  std::vector<Seg> segments;
  uint32_t k = 5120;
  uint32_t rows = kLane;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  std::optional<uint32_t> gate;
  FloatOutput destination = FloatOutput::BFloat16;
  bool rotated = false;
  // The one-row GEMV scratch: 0 none, 1 sized to fit, 2 a staged row too small, 3 partials too small.
  int gemv = 0;
  bool table16 = false;
  std::optional<LinearConfig> forced;
};

struct Affine {
  std::string name;
  uint32_t family = 7, cores = 24;
  uint32_t n = 5120, k = 5120, bits = 4;
  uint32_t rows = kLane;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
};

DeviceCapabilities device(uint32_t family, uint32_t cores) {
  DeviceCapabilities d;
  d.appleGpuFamily = family;
  d.gpuCoreCount = cores;
  return d;
}

void header(std::ostream &out, const std::string &name, uint32_t family, uint32_t cores) {
  out << "== " << name << " family=" << family << " cores=" << cores << '\n';
}

void footer(std::ostream &out, const PreparedInput &prepared, const Labels &labels) {
  out << " prepared=" << static_cast<int>(prepared.layout) << ' ' << labels.name(prepared.source) << '\n';
}

void describe(std::ostream &out, const LinearPlan &plan) {
  const LinearConfig c = plan.configuration();
  out << " plan tile=" << static_cast<int>(c.tile) << " groups=" << c.groups << " sg=" << static_cast<int>(c.simdgroups)
      << " splits=" << c.splits << " rows=" << plan.storageRows() << " input=" << static_cast<int>(plan.input())
      << " pipeline=" << plan.pipeline() << '/' << plan.secondPipeline() << '\n';
}

QuantizedSegment segment(Labels &labels, const std::string &name, const Seg &s, uint32_t k) {
  if (s.format == kFloat)
    return QuantizedSegment::floats(s.columns, k, labels.weights(name + ".f32", uint64_t{s.columns} * k * 4));
  const QuantFormat &f = kQuantFormats[s.format];
  return QuantizedSegment::planes(s.format, s.columns, k, labels.weights(name + ".p0", 4096),
                                  f.plane1_bytes ? labels.weights(name + ".p1", 4096) : MetalBuffer{},
                                  labels.weights(name + ".meta", 4096));
}

void runBlock(std::ostream &out, MetalBackend &backend, const Block &c) {
  header(out, c.name, c.family, c.cores);
  try {
    Labels labels(backend);
    const Linear linear(device(c.family, c.cores));
    BlockWeights blocks;
    uint32_t columns = 0;
    for (size_t i = 0; i < c.segments.size(); ++i) {
      blocks.segments.push_back(segment(labels, "w" + std::to_string(i), c.segments[i], c.k));
      blocks.segments.back().columnOffset = columns;
      columns += c.segments[i].columns;
    }
    // The columns past the last segment are padding (the matrix takes whole 256-column tiles).
    columns = (columns + 255) / 256 * 256;
    Projection projection(columns, c.k, std::move(blocks));
    projection.destination = c.destination;
    std::optional<Projection> gate;
    if (c.gate) {
      BlockWeights g;
      g.segments.push_back(segment(labels, "g", {*c.gate, c.segments.front().columns}, c.k));
      gate.emplace(columns, c.k, std::move(g));
    }
    if (c.rotated) {
      projection.rotation.signs = labels.weights("signs", c.k);
      if (gate) gate->rotation = projection.rotation;
    }
    const LinearWorkload w{{columns, c.k}, c.rows, c.phase, c.epilogue, WeightLayout::Block32};
    const LinearPlan plan = c.forced ? Linear::plan(w, *c.forced, c.destination)
                                     : linear.plan(w, projection, gate ? &*gate : nullptr);
    describe(out, plan);
    const uint64_t rows = plan.storageRows();
    const LinearScratchSize size = plan.scratchSize();
    LinearBuffers b;
    b.input = labels.own("in", rows * c.k * 2);
    b.output = labels.own("out", rows * columns * elementBytes(plan.destination()));
    if (c.epilogue == LinearEpilogue::Residual) b.residual = labels.own("res", rows * columns * 2);
    b.gateScratch = labels.own("gate", plan.gateScratchBytes());
    b.scratch.input = labels.own("s.in", size.input);
    b.scratch.sums = labels.own("s.sums", size.sums);
    b.scratch.partials = labels.own("s.part", size.partials);
    b.scratch.counters = labels.own("s.cnt", size.counters);
    if (c.rotated) b.scratch.rotated = labels.own("s.rot", rotatedBytes(c.k, rows));
    if (c.gemv) {
      uint64_t partials = 0;
      for (const Seg &s : c.segments) partials = std::max(partials, gemvPartialBytes(s.columns, c.k, c.gate.has_value()));
      b.scratch.gemvStaged = labels.own("s.gst", c.gemv == 2 ? 64 : gemvStageBytes(c.k));
      b.scratch.gemvPartials = labels.own("s.gpart", c.gemv == 3 ? 16 : std::max<uint64_t>(partials, 16));
    }
    if (c.table16) b.prepared = {b.input, LinearInput::Table16};
    CommandGraph graph;
    const PreparedInput prepared = linear.add(graph, b, projection, plan, gate ? &*gate : nullptr);
    dumpGraph(out, graph, labels);
    footer(out, prepared, labels);
  } catch (const Unlabeled &) {
    throw;
  } catch (const std::exception &error) {
    out << " THROW " << error.what() << '\n';
  }
}

void runAffine(std::ostream &out, MetalBackend &backend, const Affine &c) {
  header(out, c.name, c.family, c.cores);
  try {
    Labels labels(backend);
    const Linear linear(device(c.family, c.cores));
    const auto weights = [&](const std::string &name) {
      AffineWeights a;
      a.weights = labels.weights(name + ".w", uint64_t{c.n} * c.k / 2);
      const uint64_t groups = uint64_t{c.n} * (c.k / 64) * 2;
      a.scales = labels.weights(name + ".sc", groups);
      a.biases = labels.weights(name + ".bi", groups);
      if (c.bits == 5) a.hi = labels.weights(name + ".hi", uint64_t{c.n} * c.k / 8);
      return a;
    };
    Projection projection(c.n, c.k, weights("w"));
    projection.bits = c.bits;
    std::optional<Projection> gate;
    if (c.epilogue == LinearEpilogue::GateUp) {
      gate.emplace(c.n, c.k, weights("g"));
      gate->bits = c.bits;
    }
    const LinearWorkload w{{c.n, c.k}, c.rows, c.phase, c.epilogue, WeightLayout::Affine64, c.bits};
    const LinearPlan plan = linear.plan(w, projection, gate ? &*gate : nullptr);
    describe(out, plan);
    const uint64_t rows = plan.storageRows();
    const LinearScratchSize size = plan.scratchSize();
    LinearBuffers b;
    b.input = labels.own("in", rows * c.k * 2);
    b.output = labels.own("out", rows * c.n * 2);
    if (c.epilogue == LinearEpilogue::Residual) b.residual = labels.own("res", rows * c.n * 2);
    b.sums = labels.own("sums", plan.sumsBytes());
    b.gateScratch = labels.own("gate", plan.gateScratchBytes());
    b.downSums = labels.own("dsums", plan.downSumsBytes());
    b.scratch.input = labels.own("s.in", size.input);
    b.scratch.sums = labels.own("s.sums", size.sums);
    b.scratch.partials = labels.own("s.part", size.partials);
    b.scratch.counters = labels.own("s.cnt", size.counters);
    CommandGraph graph;
    PreparedInput prepared = linear.add(graph, b, projection, plan, gate ? &*gate : nullptr);
    if (c.phase == LinearPhase::Prefill && c.epilogue == LinearEpilogue::None)
      linear.addPrefillSums(graph, b.input, b.sums, projection, c.rows);
    dumpGraph(out, graph, labels);
    footer(out, prepared, labels);
  } catch (const Unlabeled &) {
    throw;
  } catch (const std::exception &error) {
    out << " THROW " << error.what() << '\n';
  }
}

struct Family {
  uint32_t id, cores;
};
// 7: simdgroup MMA tiles; 9: the register tile and non-MMA staged tiles; 10: the float neural accelerator.
constexpr Family kFamilies[] = {{7, 24}, {9, 40}, {10, 20}};

void blockCases(std::ostream &out, MetalBackend &backend) {
  using E = LinearEpilogue;
  const uint32_t pq20 = GGUF_FMT_PQ20, q4k = GGUF_FMT_Q4K, q6k = GGUF_FMT_Q6K, q80 = GGUF_FMT_Q80,
                 iq2 = GGUF_FMT_IQ2XXS, q2k = GGUF_FMT_Q2K;
  const auto name = [](const char *what, const Family &f, auto... parts) {
    std::string text = std::string(what) + "/f" + std::to_string(f.id);
    ((text += "/" + std::to_string(parts)), ...);
    return text;
  };
  for (const Family &f : kFamilies) {
    const auto base = [&](const std::string &label) {
      Block c;
      c.name = label;
      c.family = f.id;
      c.cores = f.cores;
      return c;
    };
    // Single tensors: the 27B down (out) and gate/up shapes, every epilogue the decode graph uses.
    for (const uint32_t format : {pq20, q4k, q6k, iq2, q80, q2k}) {
      const bool wide = format == pq20 || format == q4k;
      for (const uint32_t lanes : wide ? std::vector<uint32_t>{1, 2, 3, 4} : std::vector<uint32_t>{1, 3}) {
        for (const E e : {E::None, E::Residual}) {
          Block c = base(name("down", f, format, lanes, static_cast<int>(e)));
          c.segments = {{format, 5120}};
          c.k = 17408;
          c.rows = lanes * kLane;
          c.epilogue = e;
          runBlock(out, backend, c);
        }
      }
      for (const uint32_t lanes : {1u, 2u, 4u}) {
        Block c = base(name("gateup", f, format, lanes));
        c.segments = {{format, 17408}};
        c.gate = format;
        c.k = 5120;
        c.rows = lanes * kLane;
        c.epilogue = E::GateUp;
        runBlock(out, backend, c);
      }
    }
    for (const uint32_t lanes : {1u, 4u}) {
      Block c = base(name("gateup-mixed", f, lanes));
      c.segments = {{pq20, 17408}};
      c.gate = q4k;
      c.rows = lanes * kLane;
      c.epilogue = E::GateUp;
      runBlock(out, backend, c);
    }
    // Fused projections: gdn (qkv|z|alpha-beta), attention qkv, mixed formats in the sort's order.
    const std::vector<std::pair<const char *, std::vector<Seg>>> fused{
        {"fused-pq20", {{pq20, 10240}, {pq20, 6144}}},
        {"fused-pq20-3", {{pq20, 12288}, {pq20, 1024}, {pq20, 1024}}},
        {"fused-q4k", {{q4k, 12288}, {q4k, 1024}, {q4k, 1024}}},
        {"fused-mixed", {{q80, 1024}, {pq20, 12288}, {q4k, 1024}}},
        {"fused-float", {{pq20, 10240}, {pq20, 6144}, {kFloat, 96}}},
        {"fused-float-q4k", {{q4k, 10240}, {q80, 6144}, {kFloat, 96}}},
    };
    for (const auto &[label, segments] : fused)
      for (const uint32_t lanes : {1u, 2u, 4u}) {
        Block c = base(name(label, f, lanes));
        c.segments = segments;
        c.rows = lanes * kLane;
        runBlock(out, backend, c);
      }
    // The vocabulary head: bf16 and the fp32 logits.
    for (const uint32_t format : {pq20, q4k})
      for (const uint32_t lanes : {1u, 4u})
        for (const FloatOutput d : {FloatOutput::BFloat16, FloatOutput::Float32}) {
          Block c = base(name("head", f, format, lanes, static_cast<int>(d)));
          c.segments = {{format, 248320}};
          c.rows = lanes * kLane;
          c.destination = d;
          runBlock(out, backend, c);
        }
    // Prefill chunks: the decode tiles (up to 32 rows), a tail (77), the 128-row tile (200), a float router.
    for (const uint32_t format : {pq20, q4k})
      for (const uint32_t rows : {20u, 32u, 77u, 200u})
        for (const E e : {E::None, E::Residual, E::UpWithGate}) {
          Block c = base(name("prefill", f, format, rows, static_cast<int>(e)));
          c.segments = {{format, 5120}};
          c.k = 5120;
          c.rows = rows;
          c.phase = LinearPhase::Prefill;
          c.epilogue = e;
          runBlock(out, backend, c);
        }
    for (const uint32_t rows : {16u, 200u}) {
      Block c = base(name("prefill-float", f, rows));
      c.segments = {{pq20, 1024}, {kFloat, 256}};
      c.k = 2048;
      c.rows = rows;
      c.phase = LinearPhase::Prefill;
      runBlock(out, backend, c);
    }
    // The one-row GEMV: every epilogue and destination, a vocabulary head, a gate/up pair, the float strip, and
    // each condition that sends a projection back to its tile.
    for (const int variant : {1, 2, 3}) {
      for (const E e : {E::None, E::Residual}) {
        Block c = base(name("gemv", f, variant, static_cast<int>(e)));
        c.segments = {{pq20, 5120}};
        c.k = 17408;
        c.epilogue = e;
        c.gemv = variant;
        runBlock(out, backend, c);
      }
    }
    {
      Block c = base(name("gemv-gateup", f));
      c.segments = {{pq20, 17408}};
      c.gate = pq20;
      c.epilogue = E::GateUp;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-head", f));
      c.segments = {{pq20, 248320}};
      c.destination = FloatOutput::Float32;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-wide", f));
      c.segments = {{pq20, 65536}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-fused", f));
      c.segments = {{pq20, 10240}, {pq20, 6144}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-float", f));
      c.segments = {{pq20, 10240}, {pq20, 6144}, {kFloat, 96}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-float-one", f));
      c.segments = {{pq20, 10240}, {kFloat, 96}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-fused-residual", f));
      c.segments = {{pq20, 10240}, {pq20, 6144}};
      c.epilogue = E::Residual;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-two-lanes", f));
      c.segments = {{pq20, 5120}};
      c.rows = 2 * kLane;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-q4k", f));
      c.segments = {{q4k, 5120}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-ragged", f));
      c.segments = {{pq20, 5056}};
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("gemv-mixed-formats", f));
      c.segments = {{pq20, 5120}, {q4k, 1024}};
      c.gemv = 1;
      runBlock(out, backend, c);
    }
    // Rotated inputs (Prism): rotate then the consumer, with the policy's splits and forced ones,
    // a gate/up pair, fused segments, another format, the GEMV and a prefill chunk.
    for (const uint32_t format : {pq20, q4k})
      for (const uint32_t lanes : {1u, 2u, 4u})
        for (const E e : {E::None, E::Residual}) {
          Block c = base(name("rot", f, format, lanes, static_cast<int>(e)));
          c.segments = {{format, 5120}};
          c.k = 17408;
          c.rows = lanes * kLane;
          c.epilogue = e;
          c.rotated = true;
          runBlock(out, backend, c);
        }
    for (const uint32_t lanes : {1u, 4u}) {
      Block c = base(name("rot-gateup", f, lanes));
      c.segments = {{pq20, 17408}};
      c.gate = pq20;
      c.rows = lanes * kLane;
      c.epilogue = E::GateUp;
      c.rotated = true;
      runBlock(out, backend, c);
      c = base(name("rot-gateup-mixed", f, lanes));
      c.segments = {{pq20, 17408}};
      c.gate = q4k;
      c.rows = lanes * kLane;
      c.epilogue = E::GateUp;
      c.rotated = true;
      runBlock(out, backend, c);
      c = base(name("rot-fused", f, lanes));
      c.segments = {{pq20, 10240}, {pq20, 6144}};
      c.rows = lanes * kLane;
      c.rotated = true;
      runBlock(out, backend, c);
      c = base(name("rot-fused-mixed", f, lanes));
      c.segments = {{pq20, 10240}, {q4k, 6144}};
      c.rows = lanes * kLane;
      c.rotated = true;
      runBlock(out, backend, c);
      c = base(name("rot-fused-float", f, lanes));
      c.segments = {{pq20, 10240}, {pq20, 6144}, {kFloat, 96}};
      c.rows = lanes * kLane;
      c.rotated = true;
      runBlock(out, backend, c);
    }
    for (const uint32_t splits : {1u, 2u, 4u, 8u})
      for (const uint32_t k : {5120u, 17408u}) {
        Block c = base(name("rot-forced", f, splits, k));
        c.segments = {{pq20, 5120}};
        c.k = k;
        c.rotated = true;
        c.forced = LinearConfig{.tile = LinearTile::GgufStaged, .splits = splits};
        runBlock(out, backend, c);
      }
    {
      Block c = base(name("rot-gemv", f));
      c.segments = {{pq20, 5120}};
      c.rotated = true;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("rot-gemv-float", f));
      c.segments = {{pq20, 5120}, {kFloat, 96}};
      c.rotated = true;
      c.gemv = 1;
      runBlock(out, backend, c);
      c = base(name("rot-prefill", f));
      c.segments = {{pq20, 5120}};
      c.rows = 20;
      c.phase = LinearPhase::Prefill;
      c.rotated = true;
      runBlock(out, backend, c);
      c = base(name("rot-prefill-tile", f));
      c.segments = {{pq20, 5120}};
      c.rows = 200;
      c.phase = LinearPhase::Prefill;
      c.rotated = true;
      runBlock(out, backend, c);
    }
    // Forced configurations the policy does not pick: every staged split on each epilogue, the register tile with
    // and without its table already prepared.
    for (const uint32_t splits : {1u, 4u})
      for (const E e : {E::None, E::Residual})
        for (const uint32_t format : {pq20, q4k}) {
          Block c = base(name("staged-forced", f, format, splits, static_cast<int>(e)));
          c.segments = {{format, 5120}};
          c.k = 5120;
          c.epilogue = e;
          c.forced = LinearConfig{.tile = LinearTile::GgufStaged, .splits = splits};
          runBlock(out, backend, c);
        }
    for (const bool prepared : {false, true})
      for (const uint32_t lanes : {1u, 4u}) {
        Block c = base(name("register-forced", f, lanes, prepared));
        c.segments = {{pq20, 5120}};
        c.rows = lanes * kLane;
        c.table16 = prepared;
        c.forced = LinearConfig{.tile = LinearTile::GgufRegister, .splits = 2};
        runBlock(out, backend, c);
        c = base(name("register-forced-fused", f, lanes, prepared));
        c.segments = {{pq20, 10240}, {q4k, 6144}};
        c.rows = lanes * kLane;
        c.table16 = prepared;
        c.forced = LinearConfig{.tile = LinearTile::GgufRegister, .splits = 2};
        runBlock(out, backend, c);
      }
  }
}

void affineCases(std::ostream &out, MetalBackend &backend) {
  using E = LinearEpilogue;
  for (const Family &f : kFamilies)
    for (const uint32_t bits : {4u, 5u}) {
      const auto base = [&](const std::string &label) {
        Affine c;
        c.name = label + "/f" + std::to_string(f.id) + "/q" + std::to_string(bits);
        c.family = f.id;
        c.cores = f.cores;
        c.bits = bits;
        return c;
      };
      for (const uint32_t lanes : {1u, 2u, 4u})
        for (const E e : {E::None, E::Residual, E::GateUp}) {
          Affine c = base("affine-decode/" + std::to_string(lanes) + "/" + std::to_string(static_cast<int>(e)));
          c.n = e == E::GateUp ? 17408 : 5120;
          c.k = e == E::Residual ? 17408 : 5120;
          c.rows = lanes * kLane;
          c.epilogue = e;
          runAffine(out, backend, c);
        }
      for (const uint32_t rows : {20u, 77u, 200u})
        for (const E e : {E::None, E::Residual, E::UpWithGate}) {
          Affine c = base("affine-prefill/" + std::to_string(rows) + "/" + std::to_string(static_cast<int>(e)));
          c.n = 5120;
          c.k = 5120;
          c.rows = rows;
          c.phase = LinearPhase::Prefill;
          c.epilogue = e;
          runAffine(out, backend, c);
        }
    }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) {
      std::cerr << "usage: linear-graph-dump <any metallib> <fixture> [--write]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      std::ostringstream dump;
      blockCases(dump, backend);
      affineCases(dump, backend);
      const std::string text = dump.str();
      if (argc > 3 && std::string(argv[3]) == "--write") {
        std::ofstream(argv[2], std::ios::binary) << text;
        std::cout << "wrote " << argv[2] << " (" << text.size() << " bytes)\n";
        return 0;
      }
      std::ifstream file(argv[2], std::ios::binary);
      std::stringstream golden;
      golden << file.rdbuf();
      if (!file || golden.str() != text) {
        std::istringstream a(golden.str()), b(text);
        std::string x, y;
        for (int line = 1; true; ++line) {
          const bool gx = static_cast<bool>(std::getline(a, x)), gy = static_cast<bool>(std::getline(b, y));
          if (!gx && !gy) break;
          if (gx != gy || x != y) {
            std::cout << "linear graph dump differs at line " << line << "\n  golden: " << (gx ? x : "<end>")
                      << "\n  actual: " << (gy ? y : "<end>") << '\n';
            break;
          }
        }
        std::cout << "linear graph dump FAIL\n";
        return 1;
      }
      std::cout << "linear graph dump ok (" << text.size() << " bytes)\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "linear graph dump failed: " << error.what() << '\n';
      return 1;
    }
  }
}
