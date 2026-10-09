// The staged GGUF tile's dequantization (dequant32, kernels/shared/gguf_linear.metal, which the test kernels of
// gguf_dequant_test.metal expose before the matmul): every weight of every format is the half rounding of its
// GGML fp32 value, the CPU reference that gguf-reference pins to llama.cpp's golden hashes.
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::sharedBuffer;

namespace {

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-dequant <gguf-dequant.metallib>\n";
      return 2;
    }
    // One thread per row and group of 32 weights, 32 threads per threadgroup (gguf_dequant_test.metal).
    constexpr uint32_t kGroup = 32, kThreads = 32;
    // [256, 1024] in every format, then Flash-Next's [320, 640] (N not a whole plane tile, K a 64-input span and no
    // superblock) in the formats whose blocks divide 640.
    struct Shape { uint32_t N, K; };
    std::mt19937 rng(42);
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      for (const Shape shape : {Shape{256, 1024}, Shape{320, 640}})
      for (int fi = 0; fi < FMT_COUNT; ++fi) {
        const Fmt f = Fmt(fi);
        const uint32_t N = shape.N, K = shape.K;
        if (K % kQuantFormats[f].block_elements) continue;
        std::vector<float> values;
        const Packed planes = repack(f, makeNative(f, N, K, rng), N, K, &values);
        const MetalBuffer w0 = upload(backend, planes.w0), meta = upload(backend, planes.meta);
        const MetalBuffer w1 = kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1) : meta;
        const MetalBuffer output = sharedBuffer(backend, uint64_t{N} * K * 2);
        CommandGraph graph;
        graph.add(std::string("gguf_test_dequant_") + fmtName(f), {w0, w1, meta, output}, GgufDecodeParams{K, 1, N, 0},
                  {N * (K / kGroup) / kThreads, 1, 1}, {kThreads, 1, 1});
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        const auto *got = static_cast<const uint16_t *>(output.contents());
        size_t differ = 0;
        for (size_t i = 0; i < values.size(); ++i) differ += got[i] != f2h(values[i]);
        std::cout << fmtName(f) << " [" << N << ", " << K << "]: " << differ << " of " << values.size()
                  << " weights differ from FP16(GGML fp32 dequantization) " << (differ ? "FAIL" : "ok") << '\n';
        failures += differ != 0;
      }
    } catch (const std::exception &e) {
      std::cerr << "gguf-dequant: FAIL: " << e.what() << '\n';
      return 1;
    }
    return failures ? 1 : 0;
  }
}
