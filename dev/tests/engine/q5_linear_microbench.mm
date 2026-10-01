// Q5 vs Q4 microbench for the Apple7/8 register-matrix tiles (SimdgroupF32
// decode, Mma64 prefill). Not wired into dev/native.mk: a one-off
// measurement, run directly against the already-built production metallib.
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/Q5Pack.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace splash;
using namespace splash::ops;

namespace {
uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; v *= 0x846ca68b; return v ^ (v >> 16); }
uint16_t floatToHalf(float v) { _Float16 h = _Float16(v); uint16_t b; std::memcpy(&b, &h, 2); return b; }
uint16_t floatToBf16(float v) { uint32_t b; std::memcpy(&b, &v, 4); return uint16_t(b >> 16); }

Projection makeQ4(metal::MetalBackend &backend, LinearMatrix m, uint32_t seed) {
  const uint64_t params = uint64_t(m.outputSize) * (m.inputSize / 64);
  Projection p(m.outputSize, m.inputSize,
               AffineWeights{backend.allocateBuffer(params * 32), backend.allocateBuffer(params * 2),
                             backend.allocateBuffer(params * 2)});
  auto *w = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *sc = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *bi = static_cast<uint16_t *>(p.affine().biases.contents());
  for (uint64_t i = 0; i < params * 32; ++i) w[i] = uint8_t(hash(uint32_t(i) + seed));
  for (uint64_t i = 0; i < params; ++i) {
    const float scale = 0.004f + float(hash(uint32_t(i) + seed + 3) % 17) * 0.0001f;
    sc[i] = floatToHalf(scale); bi[i] = floatToHalf(-7.5f * scale);
  }
  return p;
}
Projection makeQ5(metal::MetalBackend &backend, LinearMatrix m, uint32_t seed) {
  const uint64_t groups = uint64_t(m.outputSize) * (m.inputSize / model::q5::kGroupElements);
  const uint64_t lo4Bytes = groups * model::q5::kLo4Bytes, hiBytes = groups * model::q5::kHiBytes;
  const uint64_t params = groups;
  Projection p(m.outputSize, m.inputSize,
               AffineWeights{backend.allocateBuffer(lo4Bytes), backend.allocateBuffer(params * 2),
                             backend.allocateBuffer(params * 2), backend.allocateBuffer(hiBytes)});
  p.bits = 5;
  auto *w = static_cast<uint8_t *>(p.affine().weights.contents());
  auto *hi = static_cast<uint8_t *>(p.affine().hi.contents());
  auto *sc = static_cast<uint16_t *>(p.affine().scales.contents());
  auto *bi = static_cast<uint16_t *>(p.affine().biases.contents());
  for (uint64_t i = 0; i < lo4Bytes; ++i) w[i] = uint8_t(hash(uint32_t(i) + seed));
  for (uint64_t i = 0; i < hiBytes; ++i) hi[i] = uint8_t(hash(uint32_t(i) + seed + 991));
  for (uint64_t i = 0; i < params; ++i) {
    const float scale = 0.004f + float(hash(uint32_t(i) + seed + 3) % 17) * 0.0001f;
    sc[i] = floatToHalf(scale); bi[i] = floatToHalf(-7.5f * scale);
  }
  return p;
}
metal::MetalBuffer bf16Input(metal::MetalBackend &backend, uint64_t elements, uint32_t seed) {
  metal::MetalBuffer buf = backend.allocateBuffer(elements * 2);
  auto *p = static_cast<uint16_t *>(buf.contents());
  for (uint64_t i = 0; i < elements; ++i)
    p[i] = floatToBf16(float(int(hash(uint32_t(i) + seed) % 257) - 128) / 32);
  return buf;
}

double benchDecode(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t bits, int repeats) {
  DeviceCapabilities device; device.appleGpuFamily = 7; device.gpuCoreCount = 16;
  Linear linear(device);
  const LinearMatrix m{n, k};
  Projection p = bits == 5 ? makeQ5(backend, m, 41) : makeQ4(backend, m, 41);
  LinearWorkload w{m, SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode};
  w.bits = bits;
  const auto plan = linear.plan(w, p);
  metal::MetalBuffer input = bf16Input(backend, uint64_t(SPLASH_TARGET_VERIFY_ROWS) * k, 7);
  metal::MetalBuffer output = backend.allocateBuffer(2ULL * SPLASH_TARGET_VERIFY_ROWS * n);
  const auto scratchSize = plan.scratchSize();
  metal::MetalBuffer table = backend.allocateBuffer(std::max<uint64_t>(scratchSize.input, 1));
  metal::MetalBuffer sums = backend.allocateBuffer(std::max<uint64_t>(scratchSize.sums, 1));
  metal::MetalBuffer partials = backend.allocateBuffer(std::max<uint64_t>(scratchSize.partials, 1));
  metal::MetalBuffer counters = backend.allocateBuffer(std::max<uint64_t>(scratchSize.counters, 1));
  std::memset(counters.contents(), 0, counters.sizeBytes());
  LinearScratch scratch{table, sums, partials, counters};
  double total = 0;
  for (int r = 0; r < repeats; ++r) {
    std::memset(counters.contents(), 0, counters.sizeBytes());
    metal::CommandGraph graph;
    linear.add(graph, {input, output, {}, {}, {}, {}, scratch}, p, plan);
    total += backend.submitCommand(graph.dispatches()).wallSeconds;
  }
  return total / repeats;
}

double benchPrefill(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits, int repeats) {
  DeviceCapabilities device; device.appleGpuFamily = 7; device.gpuCoreCount = 16;
  Linear linear(device);
  const LinearMatrix m{n, k};
  Projection p = bits == 5 ? makeQ5(backend, m, 61) : makeQ4(backend, m, 61);
  LinearWorkload w{m, rows, LinearPhase::Prefill};
  w.bits = bits;
  const auto plan = linear.plan(w, p);
  metal::MetalBuffer input = bf16Input(backend, uint64_t(rows) * k, 11);
  metal::MetalBuffer output = backend.allocateBuffer(2ULL * rows * n);
  metal::MetalBuffer sums = backend.allocateBuffer(4ULL * rows * (k / 64));
  std::memset(sums.contents(), 0, sums.sizeBytes());
  double total = 0;
  for (int r = 0; r < repeats; ++r) {
    metal::CommandGraph graph;
    linear.add(graph, {input, output, sums, {}, {}, {}, {}}, p, plan);
    total += backend.submitCommand(graph.dispatches()).wallSeconds;
  }
  return total / repeats;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) { std::cerr << "usage: q5-linear-microbench <metallib>\n"; return 1; }
  metal::MetalBackend backend(argv[1]);
  const std::vector<std::pair<uint32_t, uint32_t>> shapes{{5120, 17408}, {6144, 5120}, {5120, 6144}, {1024, 5120}};
  constexpr int kRepeats = 20, kWarmup = 5;
  for (const auto &[n, k] : shapes) {
    for (int i = 0; i < kWarmup; ++i) benchDecode(backend, n, k, 4, 1);
    const double q4 = benchDecode(backend, n, k, 4, kRepeats) * 1000;
    for (int i = 0; i < kWarmup; ++i) benchDecode(backend, n, k, 5, 1);
    const double q5 = benchDecode(backend, n, k, 5, kRepeats) * 1000;
    std::cout << "decode  N=" << n << " K=" << k << " rows=8   Q4=" << q4 << "ms Q5=" << q5
              << "ms ratio=" << (q5 / q4) << '\n';
  }
  for (const auto &[n, k] : shapes) {
    for (int i = 0; i < kWarmup; ++i) benchPrefill(backend, n, k, 256, 4, 1);
    const double q4 = benchPrefill(backend, n, k, 256, 4, kRepeats) * 1000;
    for (int i = 0; i < kWarmup; ++i) benchPrefill(backend, n, k, 256, 5, 1);
    const double q5 = benchPrefill(backend, n, k, 256, 5, kRepeats) * 1000;
    std::cout << "prefill N=" << n << " K=" << k << " rows=256 Q4=" << q4 << "ms Q5=" << q5
              << "ms ratio=" << (q5 / q4) << '\n';
  }
  return 0;
}
