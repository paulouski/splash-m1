// PQ2_0 one-row GEMV (decode/linear_gguf_gemv.metal) against the 8-row decode tile of ops::Linear on the Bonsai
// decode shapes, DRAM-cold (a ring of weight copies of at least 384 MiB, one command per case), with a row-0 check
// against the fp64 dequantized reference and the sum of the best GEMV per linear over one token:
//   gguf-gemv-benchmark <metallib> [rounds]
#include "../tests/engine/GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

namespace {

struct Shape { const char *name; uint32_t N, K; char ep; uint32_t perToken; };
constexpr Shape kShapes[] = {
    {"gate+up", 17408, 5120, 'g', 64}, {"down", 5120, 17408, 'r', 64},   {"gdn in", 16384, 5120, 'a', 48},
    {"gdn out", 5120, 6144, 'r', 48},   {"attn in", 14336, 5120, 'a', 16}, {"attn out", 5120, 6144, 'r', 16},
    {"head", 248320, 5120, 'f', 1}};
constexpr uint64_t kRingBytes = 384ull << 20;
constexpr uint32_t kMaxSplits = 16;

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = backend.allocateBuffer(bytes.size(), metal::BufferStorage::Shared, "gemv-bench");
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}
MetalBuffer zeros(MetalBackend &backend, uint64_t n) {
  if (!n) return MetalBuffer{};
  MetalBuffer b = backend.allocateBuffer(n, metal::BufferStorage::Shared, "gemv-bench");
  std::memset(b.contents(), 0, n);
  return b;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) { std::cerr << "usage: gguf-gemv-benchmark <metallib> [rounds]\n"; return 2; }
    const uint32_t rounds = argc > 2 ? uint32_t(std::stoul(argv[2])) : 6;
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities());
      std::mt19937 rng(9);
      std::printf("%-9s %7s %6s %5s | %9s %7s | %-24s %9s %7s | %s\n", "shape", "N", "K", "MB", "M8 ms", "GB/s",
                  "best GEMV (splits)", "ms", "GB/s", "row0 max err / bound");
      double sumM8 = 0, sumGemv = 0;
      for (const Shape &s : kShapes) {
        const uint32_t N = s.N, K = s.K;
        const bool pair = s.ep == 'g';
        const std::vector<uint8_t> native = makeNative(PQ20, N, K, rng);
        const std::vector<uint8_t> native2 = pair ? makeNative(PQ20, N, K, rng) : std::vector<uint8_t>{};
        const Packed packed = repack(PQ20, native, N, K, nullptr);
        const Packed packed2 = pair ? repack(PQ20, native2, N, K, nullptr) : Packed{};
        const double weightBytes = (double(N) * K * 2 / 8 + double(N) * K / 128 * 2) * (pair ? 2 : 1);
        const uint32_t copies = uint32_t(std::clamp<uint64_t>(uint64_t(kRingBytes / weightBytes) + 1, 3, 64));
        std::vector<MetalBuffer> w0, meta, w0u, metau;
        std::vector<Projection> ring, ringUp;
        const auto projection = [&](const Packed &pk, std::vector<MetalBuffer> &w, std::vector<MetalBuffer> &m,
                                    std::vector<Projection> &out) {
          w.push_back(upload(backend, pk.w0));
          m.push_back(upload(backend, pk.meta));
          BlockWeights weights;
          weights.segments.push_back(QuantizedSegment::planes(PQ20, N, K, w.back(), MetalBuffer{}, m.back()));
          out.emplace_back(N, K, std::move(weights));
        };
        for (uint32_t c = 0; c < copies; ++c) {
          projection(packed, w0, meta, ring);
          if (pair) projection(packed2, w0u, metau, ringUp);
        }
        const bool f32 = s.ep == 'f', residual = s.ep == 'r';
        const uint64_t outBytes = f32 ? 4 : 2;
        MetalBuffer input = zeros(backend, 8ull * K * 2), output = zeros(backend, 8ull * N * outBytes),
                    aux = zeros(backend, 8ull * N * 2), partials = zeros(backend, 2ull * kMaxSplits * N * 4),
                    staged = zeros(backend, uint64_t(K) * 4 + K / 32 + 64);
        std::vector<float> xf(K), auxf(N);
        {
          std::uniform_real_distribution<float> unit(-1.f, 1.f);
          auto *x = static_cast<__bf16 *>(input.contents());
          for (uint64_t i = 0; i < 8ull * K; ++i) x[i] = __bf16(unit(rng));
          for (uint32_t i = 0; i < K; ++i) xf[i] = float(x[i]);
          auto *a = static_cast<__bf16 *>(aux.contents());
          for (uint64_t i = 0; i < 8ull * N; ++i) a[i] = __bf16(unit(rng));
          for (uint32_t i = 0; i < N; ++i) auxf[i] = float(a[i]);
        }
        const LinearWorkload w{{N, K}, 8, LinearPhase::Decode,
                               residual ? LinearEpilogue::Residual : pair ? LinearEpilogue::GateUp : LinearEpilogue::None,
                               WeightLayout::Block32};
        const LinearConfig policy = linear.plan(w, ring.front()).configuration();
        const LinearPlan m8 = Linear::plan(w, policy, f32 ? FloatOutput::Float32 : FloatOutput::BFloat16);
        const LinearScratchSize size = m8.scratchSize();
        const LinearScratch scratch{zeros(backend, size.input), zeros(backend, size.sums),
                                    zeros(backend, size.partials), zeros(backend, size.counters)};
        MetalBuffer gate = zeros(backend, 8ull * N * 2);
        uint32_t next = 0;
        const auto timeM8 = [&] {
          CommandGraph graph;
          const LinearBuffers b{.input = input, .output = output, .residual = residual ? aux : MetalBuffer{},
                                .gateScratch = gate, .scratch = scratch};
          for (uint32_t i = 0; i < copies; ++i, ++next)
            static_cast<void>(linear.add(graph, b, ring[next % copies], m8, pair ? &ringUp[next % copies] : nullptr));
          return backend.submitCommand(graph.dispatches()).gpuSeconds * 1e3 / copies;
        };
        const char *suffix = f32 ? "a_f32" : residual ? "r" : "a";
        const auto buildGemv = [&](CommandGraph &graph, uint32_t c, uint32_t splits) {
          const GgufDecodeParams p{K, splits, N, 0};
          graph.add("gguf_gemv_stage", {input, staged}, p, {(K / 128 + 7) / 8, 1, 1}, {256, 1, 1});
          if (pair) {
            graph.add("gguf_gemv_pq20_g", {staged, w0[c], meta[c], w0u[c], metau[c], output, partials}, p,
                      {N / 256, splits, 1}, {256, 1, 1});
            if (splits > 1)
              graph.add("gguf_gemv_reduce_g", {partials, output}, p, {(N + 255) / 256, 1, 1}, {256, 1, 1});
            return;
          }
          graph.add(std::string("gguf_gemv_pq20_") + suffix, {staged, w0[c], meta[c], output, partials, aux},
                    p, {N / 256, splits, 1}, {128, 1, 1});
          if (splits > 1)
            graph.add(std::string("gguf_gemv_reduce_") + suffix, {partials, output, aux}, p, {(N + 255) / 256, 1, 1},
                      {256, 1, 1});
        };
        const auto timeGemv = [&](uint32_t splits) {
          CommandGraph graph;
          for (uint32_t i = 0; i < copies; ++i, ++next) buildGemv(graph, next % copies, splits);
          return backend.submitCommand(graph.dispatches()).gpuSeconds * 1e3 / copies;
        };
        std::vector<uint32_t> splitList;
        for (uint32_t sp = 1; sp <= kMaxSplits; sp *= 2)
          if ((K / 128) % sp == 0) splitList.push_back(sp);
        std::vector<uint32_t> rows;
        for (uint32_t i = 0; i < 64; ++i) rows.push_back(uint32_t((uint64_t(i) * 2654435761u) % N));
        const uint32_t rb = rowBytes(PQ20, K);
        std::string verdict;
        double worst = 0;
        const auto bf16 = [](double v) { return double(float(__bf16(float(v)))); };
        for (const uint32_t splits : splitList) {
          std::memset(output.contents(), 0, 8ull * N * outBytes);
          CommandGraph graph;
          buildGemv(graph, 0, splits);
          static_cast<void>(backend.submitCommand(graph.dispatches()));
          for (const uint32_t r : rows) {
            std::vector<float> wv(K), wu(K);
            rowValues(PQ20, native.data() + size_t(r) * rb, K, wv.data());
            const Dot d = dot(xf.data(), wv.data(), K);
            double ref = d.value + (residual ? auxf[r] : 0.0);
            double bound = projectionBound(d, false) + (f32 || pair ? 0.0 : std::ldexp(std::fabs(ref), -8));
            if (pair) {
              rowValues(PQ20, native2.data() + size_t(r) * rb, K, wu.data());
              const Dot e = dot(xf.data(), wu.data(), K);
              const double g = d.value, u = e.value, silu = g / (1 + std::exp(-g));
              ref = silu * u;
              bound = std::fabs(silu) * projectionBound(e, false) + std::fabs(u) * 1.1 * projectionBound(d, false) +
                      std::ldexp(std::fabs(ref) + std::fabs(silu * u) * 3, -7);
            }
            const double got = f32 ? double(static_cast<float *>(output.contents())[r])
                                   : double(float(static_cast<__bf16 *>(output.contents())[r]));
            const double err = std::fabs(got - ref);
            if (!(err <= bound)) verdict = "FAIL s" + std::to_string(splits) + " row " + std::to_string(r);
            worst = std::isfinite(err) ? std::max(worst, err / bound) : INFINITY;
          }
        }
        static_cast<void>(bf16);
        if (verdict.empty()) verdict = "ok";
        for (uint32_t i = 0; i < 2; ++i) { timeM8(); for (const uint32_t c : splitList) timeGemv(c); }
        std::vector<double> tm8;
        std::vector<std::vector<double>> tg(splitList.size());
        for (uint32_t r = 0; r < rounds; ++r) {
          tm8.push_back(timeM8());
          for (size_t i = 0; i < splitList.size(); ++i) {
            const size_t j = r % 2 ? splitList.size() - 1 - i : i;
            tg[j].push_back(timeGemv(splitList[j]));
          }
          tm8.push_back(timeM8());
        }
        const auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double ms8 = median(tm8);
        size_t best = 0;
        std::string all;
        for (size_t i = 0; i < splitList.size(); ++i) {
          const double m = median(tg[i]);
          if (m < median(tg[best])) best = i;
          char buf[48];
          std::snprintf(buf, sizeof buf, " s%u=%.3f", splitList[i], m);
          all += buf;
        }
        const double msG = median(tg[best]);
        char bestLabel[32];
        std::snprintf(bestLabel, sizeof bestLabel, "s%u", splitList[best]);
        std::printf("%-9s %7u %6u %5.0f | %9.3f %7.1f | %-24s %9.3f %7.1f | %s worst %.2f of bound\n   sweep:%s\n", s.name,
                    N, K, weightBytes / 1e6, ms8, weightBytes / ms8 / 1e6, bestLabel, msG, weightBytes / msG / 1e6,
                    verdict.c_str(), worst, all.c_str());
        sumM8 += ms8 * s.perToken;
        sumGemv += msG * s.perToken;
      }
      std::printf("token sum: M=8 linears %.1f ms, GEMV linears %.1f ms (GO <= 40 ms)\n", sumM8, sumGemv);
    } catch (const std::exception &e) {
      std::cerr << "gguf-gemv-benchmark: " << e.what() << '\n';
      return 1;
    }
    return 0;
  }
}
