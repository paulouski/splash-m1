// Correctness test for the Q5 (Q5Pack.hpp) SimdgroupF32 decode and Mma64
// prefill tiles (Apple7/8 only). Two independent checks per tile:
//   1. A Q4 projection promoted losslessly into Q5 (lo4 unchanged, hi=0,
//      Q5Pack.hpp's promoteQ4Group) must produce a BYTE-IDENTICAL output to
//      the same weights run through the existing Q4 kernel: this pins down
//      the host-side buffer bindings/pipeline routing (Linear.cpp) and the
//      kernel's hi=0 path in one exact comparison, no error bound needed.
//   2. Random hi bits, compared to a CPU fp64 reference built directly from
//      the lo4/hi planes (v = lo4 | hi<<4, dequant = scale*v + bias), within
//      a generous fixed tolerance -- this test's job is to catch a
//      transposed/mis-scaled hi bit, not to derive a tight rounding bound.
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/Q5Pack.hpp"
#include "ops/Linear.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;

namespace {

void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}

// Native half, independent of the bf16 helpers elsewhere: the SimdgroupF32
// and Mma64 kernels' scale/bias buffers are `half`.
uint16_t floatToHalf(float v) {
  _Float16 h = _Float16(v);
  uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}
float halfToFloat(uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return float(h);
}
uint16_t floatToBf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return uint16_t(bits >> 16);
}
float bf16ToFloat(uint16_t bits) {
  uint32_t v = uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}
uint32_t hash(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; v *= 0x846ca68b; return v ^ (v >> 16);
}

// A Q4 projection (lo4 nibbles in the StorageN=256 tile layout) and its Q5
// promotion (same lo4, hi=0) or Q5 randomization (hash-derived hi bits).
struct QProjections {
  Projection q4, q5;
  std::vector<uint8_t> lo4;
  std::vector<uint8_t> hi;
};
QProjections buildProjections(metal::MetalBackend &backend, LinearMatrix m, uint32_t seed, bool randomHi) {
  const uint64_t groups = uint64_t(m.outputSize) * (m.inputSize / model::q5::kGroupElements);
  const uint64_t lo4Bytes = groups * model::q5::kLo4Bytes;
  const uint64_t hiBytes = groups * model::q5::kHiBytes;
  const uint64_t paramBytes = groups * 2;
  std::vector<uint8_t> lo4(lo4Bytes), hi(hiBytes, 0);
  for (uint64_t i = 0; i < lo4Bytes; ++i) lo4[i] = uint8_t(hash(uint32_t(i) + seed));
  if (randomHi)
    for (uint64_t i = 0; i < hiBytes; ++i) hi[i] = uint8_t(hash(uint32_t(i) + seed + 991));

  QProjections out{Projection(m.outputSize, m.inputSize,
                              AffineWeights{backend.allocateBuffer(lo4Bytes), backend.allocateBuffer(paramBytes),
                                            backend.allocateBuffer(paramBytes)}),
                   Projection(m.outputSize, m.inputSize,
                              AffineWeights{backend.allocateBuffer(lo4Bytes), backend.allocateBuffer(paramBytes),
                                            backend.allocateBuffer(paramBytes), backend.allocateBuffer(hiBytes)}),
                   std::move(lo4), std::move(hi)};
  out.q5.bits = 5;
  std::memcpy(out.q4.affine().weights.contents(), out.lo4.data(), lo4Bytes);
  std::memcpy(out.q5.affine().weights.contents(), out.lo4.data(), lo4Bytes);
  std::memcpy(out.q5.affine().hi.contents(), out.hi.data(), hiBytes);
  auto *sc4 = static_cast<uint16_t *>(out.q4.affine().scales.contents());
  auto *bi4 = static_cast<uint16_t *>(out.q4.affine().biases.contents());
  auto *sc5 = static_cast<uint16_t *>(out.q5.affine().scales.contents());
  auto *bi5 = static_cast<uint16_t *>(out.q5.affine().biases.contents());
  for (uint64_t i = 0; i < groups; ++i) {
    const float scale = 0.004f + float(hash(uint32_t(i) + seed + 3) % 17) * 0.0001f;
    const float bias = -7.5f * scale;
    // Both the Q4 and Q5 SimdgroupF32/Mma64 kernels read scale/bias as
    // native half (not bf16, unlike the older N128/N256 tiles).
    sc4[i] = floatToHalf(scale); bi4[i] = floatToHalf(bias);
    sc5[i] = floatToHalf(scale); bi5[i] = floatToHalf(bias);
  }
  return out;
}

// value(col, k) of the promoted/randomized projection, decoded straight from
// the lo4/hi planes in StorageN=256 order (independent of any kernel code).
uint32_t rawValue(const QProjections &p, LinearMatrix m, uint32_t col, uint32_t k) {
  const uint32_t groupsPerTile = m.inputSize / model::q5::kGroupElements;
  const uint32_t tile = col / 256, localCol = col % 256, group = k / model::q5::kGroupElements;
  const uint32_t local = k % model::q5::kGroupElements;
  const uint64_t g = (uint64_t(tile) * groupsPerTile + group) * 256 + localCol;
  const uint8_t *lo4 = p.lo4.data() + g * model::q5::kLo4Bytes;
  const uint8_t *hi = p.hi.data() + g * model::q5::kHiBytes;
  return model::q5::valueFromSplit(lo4, hi, local);
}
double exactValue(const QProjections &p, LinearMatrix m, const std::vector<float> &x, uint32_t row, uint32_t col) {
  const auto *sc = static_cast<const uint16_t *>(p.q5.affine().scales.contents());
  const auto *bi = static_cast<const uint16_t *>(p.q5.affine().biases.contents());
  double value = 0;
  const uint32_t groupsPerTile = m.inputSize / model::q5::kGroupElements;
  for (uint32_t group = 0; group < groupsPerTile; ++group) {
    const uint64_t at = (uint64_t(col / 256) * groupsPerTile + group) * 256 + col % 256;
    double dot = 0, sum = 0;
    for (uint32_t local = 0; local < model::q5::kGroupElements; ++local) {
      const uint32_t k = group * model::q5::kGroupElements + local;
      const double xv = x[uint64_t(row) * m.inputSize + k];
      dot += xv * rawValue(p, m, col, k);
      sum += xv;
    }
    value += halfToFloat(sc[at]) * dot + halfToFloat(bi[at]) * sum;
  }
  return value;
}

std::vector<float> randomInput(uint64_t rows, uint32_t width, uint32_t seed) {
  std::vector<float> x(rows * width);
  for (uint64_t i = 0; i < x.size(); ++i)
    x[i] = float(int(hash(uint32_t(i) + seed) % 257) - 128) / 32;
  return x;
}
metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, const std::vector<float> &values) {
  metal::MetalBuffer buffer = backend.allocateBuffer(values.size() * 2);
  auto *p = static_cast<uint16_t *>(buffer.contents());
  for (size_t i = 0; i < values.size(); ++i) p[i] = floatToBf16(values[i]);
  return buffer;
}

void runDecode(metal::MetalBackend &backend, uint32_t n, uint32_t k, bool randomHi) {
  DeviceCapabilities device;
  device.appleGpuFamily = 7;
  device.gpuCoreCount = 16;
  Linear linear(device);
  const LinearMatrix m{n, k};
  const uint32_t rows = SPLASH_TARGET_VERIFY_ROWS;
  const auto q = buildProjections(backend, m, 41, randomHi);
  const auto x = randomInput(rows, k, 7);
  metal::MetalBuffer input = bf16Buffer(backend, x);
  metal::MetalBuffer outQ4 = backend.allocateBuffer(2ULL * rows * n);
  metal::MetalBuffer outQ5 = backend.allocateBuffer(2ULL * rows * n);
  std::memset(outQ4.contents(), 0, 2ULL * rows * n);
  std::memset(outQ5.contents(), 0, 2ULL * rows * n);

  const LinearWorkload w4{m, rows, LinearPhase::Decode};
  LinearWorkload w5 = w4; w5.bits = 5;
  const auto plan4 = linear.plan(w4, q.q4);
  const auto plan5 = linear.plan(w5, q.q5);
  require(plan4.configuration().tile == LinearTile::SimdgroupF32, "Q4 baseline is not SimdgroupF32 on Apple7");
  require(plan5.configuration().tile == LinearTile::SimdgroupF32, "Q5 baseline is not SimdgroupF32 on Apple7");
  const auto scratchSize = plan5.scratchSize().include(plan4.scratchSize());
  metal::MetalBuffer table = backend.allocateBuffer(std::max<uint64_t>(scratchSize.input, 1));
  metal::MetalBuffer sums = backend.allocateBuffer(std::max<uint64_t>(scratchSize.sums, 1));
  metal::MetalBuffer partials = backend.allocateBuffer(std::max<uint64_t>(scratchSize.partials, 1));
  metal::MetalBuffer counters = backend.allocateBuffer(std::max<uint64_t>(scratchSize.counters, 1));
  std::memset(counters.contents(), 0, counters.sizeBytes());
  LinearScratch scratch{table, sums, partials, counters};

  metal::CommandGraph graph;
  linear.add(graph, {input, outQ4, {}, {}, {}, {}, scratch}, q.q4, plan4);
  std::memset(counters.contents(), 0, counters.sizeBytes());
  linear.add(graph, {input, outQ5, {}, {}, {}, {}, scratch}, q.q5, plan5);
  (void)backend.submitCommand(graph.dispatches());

  const auto *outputQ4 = static_cast<const uint16_t *>(outQ4.contents());
  const auto *outputQ5 = static_cast<const uint16_t *>(outQ5.contents());
  if (!randomHi) {
    if (std::memcmp(outputQ4, outputQ5, 2ULL * rows * n) != 0) {
      uint32_t printed = 0;
      for (uint64_t i = 0; i < uint64_t(rows) * n && printed < 10; ++i)
        if (outputQ4[i] != outputQ5[i]) {
          std::cerr << "mismatch i=" << i << " row=" << i / n << " col=" << i % n
                    << " q4=" << bf16ToFloat(outputQ4[i]) << " (" << outputQ4[i] << ")"
                    << " q5=" << bf16ToFloat(outputQ5[i]) << " (" << outputQ5[i] << ")\n";
          ++printed;
        }
      throw std::runtime_error("N=" + std::to_string(n) + " K=" + std::to_string(k) +
                               " promoted Q4->Q5 decode output is not byte-identical to Q4");
    }
    std::cout << "PASS q5 decode promoted-Q4 exact N=" << n << " K=" << k << '\n';
    return;
  }
  double maxRelError = 0;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t col = 0; col < n; ++col) {
      const double reference = exactValue(q, m, x, row, col);
      const double actual = bf16ToFloat(outputQ5[row * n + col]);
      const double denom = std::max(1.0, std::fabs(reference));
      maxRelError = std::max(maxRelError, std::fabs(actual - reference) / denom);
    }
  require(maxRelError < 0.02, "decode Q5 exceeds generous tolerance, max_rel=" + std::to_string(maxRelError));
  std::cout << "PASS q5 decode random-hi N=" << n << " K=" << k << " max_rel=" << maxRelError << '\n';
}

void runPrefill(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t rows, bool randomHi) {
  DeviceCapabilities device;
  device.appleGpuFamily = 7;
  device.gpuCoreCount = 16;
  Linear linear(device);
  const LinearMatrix m{n, k};
  const auto q = buildProjections(backend, m, 61, randomHi);
  const uint32_t tileRows = 32, storageRows = ((rows + tileRows - 1) / tileRows) * tileRows;
  const auto x = randomInput(storageRows, k, 11);
  metal::MetalBuffer input = bf16Buffer(backend, x);
  metal::MetalBuffer outQ4 = backend.allocateBuffer(2ULL * storageRows * n);
  metal::MetalBuffer outQ5 = backend.allocateBuffer(2ULL * storageRows * n);
  std::memset(outQ4.contents(), 0, 2ULL * storageRows * n);
  std::memset(outQ5.contents(), 0, 2ULL * storageRows * n);
  // Row sums the same way prefill_linear_q4_sums32 does (linear_q4.metal,
  // excluded from the macos15/Apple7-8 metallib -- this test targets only
  // the Q5/Q4 Mma64 tiles, so compute them on the CPU instead of depending
  // on that unrelated kernel).
  const uint32_t groupsPerRowTile = k / 64;
  metal::MetalBuffer sums = backend.allocateBuffer(4ULL * storageRows * groupsPerRowTile);
  auto *sumsPtr = static_cast<float *>(sums.contents());
  for (uint32_t tile = 0; tile < storageRows / tileRows; ++tile)
    for (uint32_t group = 0; group < groupsPerRowTile; ++group)
      for (uint32_t row = 0; row < tileRows; ++row) {
        float sum = 0;
        for (uint32_t z = 0; z < 64; ++z)
          sum += bf16ToFloat(floatToBf16(x[uint64_t(tile * tileRows + row) * k + group * 64 + z]));
        sumsPtr[tile * tileRows * groupsPerRowTile + group * tileRows + row] = sum;
      }

  const auto plan4 = linear.prefillPlan(q.q4, rows, LinearEpilogue::None);
  LinearWorkload w5{m, rows, LinearPhase::Prefill};
  w5.bits = 5;
  const auto plan5 = linear.plan(w5, q.q5);
  require(plan4.configuration().tile == LinearTile::Mma64, "Q4 prefill baseline is not Mma64 on Apple7");
  require(plan5.configuration().tile == LinearTile::Mma64, "Q5 prefill baseline is not Mma64 on Apple7");

  metal::CommandGraph graph;
  linear.add(graph, {input, outQ4, sums, {}, {}, {}, {}}, q.q4, plan4);
  linear.add(graph, {input, outQ5, sums, {}, {}, {}, {}}, q.q5, plan5);
  (void)backend.submitCommand(graph.dispatches());

  const auto *outputQ4 = static_cast<const uint16_t *>(outQ4.contents());
  const auto *outputQ5 = static_cast<const uint16_t *>(outQ5.contents());
  if (!randomHi) {
    require(std::memcmp(outputQ4, outputQ5, 2ULL * rows * n) == 0,
            "N=" + std::to_string(n) + " K=" + std::to_string(k) +
                " promoted Q4->Q5 prefill output is not byte-identical to Q4");
    std::cout << "PASS q5 prefill promoted-Q4 exact N=" << n << " K=" << k << " rows=" << rows << '\n';
    return;
  }
  double maxRelError = 0;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t col = 0; col < n; ++col) {
      const double reference = exactValue(q, m, x, row, col);
      const double actual = bf16ToFloat(outputQ5[row * n + col]);
      const double denom = std::max(1.0, std::fabs(reference));
      maxRelError = std::max(maxRelError, std::fabs(actual - reference) / denom);
    }
  require(maxRelError < 0.02, "prefill Q5 exceeds generous tolerance, max_rel=" + std::to_string(maxRelError));
  std::cout << "PASS q5 prefill random-hi N=" << n << " K=" << k << " rows=" << rows << " max_rel=" << maxRelError
            << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) { std::cerr << "usage: q5-linear-metal-test <metallib>\n"; return 1; }
  try {
    metal::MetalBackend backend(argv[1]);
    for (bool randomHi : {false, true}) {
      runDecode(backend, 256, 5120, randomHi);
      runDecode(backend, 1024, 5120, randomHi);
      runPrefill(backend, 256, 5120, 32, randomHi);
      runPrefill(backend, 256, 5120, 256, randomHi);
    }
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
