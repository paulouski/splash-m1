// Full-vocab fp32 logits from the Splash engine at chosen positions of fixed token segments.
//
//   splash-logits-dump METALLIB MODEL_DIR CORPUS_DIR OUT_DIR [--stride N] [--chunk N]
//
// CORPUS_DIR/<seg>.tokens.txt: one line of token ids. For every position p = stride-1,
// 2*stride-1, ... the prefix tokens[0..p] is cold-prefilled and the final-prompt-position
// logits row (the distribution of token p+1) is captured through model::gLogitsDump.
// Writes OUT_DIR/<seg>.ours.f32 (rows x n_vocab) and OUT_DIR/<seg>.ours.json.
// Teacher-forced by construction: the prefix is the corpus. Cold prefill per prefix costs
// sum(p) tokens per segment, hence the stride.

#include "engine/RuntimeResources.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/Runtime.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

std::vector<float> drain(std::FILE *file) {
  std::fflush(file);
  const long size = std::ftell(file);
  std::vector<float> rows(size / sizeof(float));
  std::rewind(file);
  if (std::fread(rows.data(), 1, size, file) != static_cast<size_t>(size))
    throw std::runtime_error("tap read failed");
  std::rewind(file);
  if (ftruncate(fileno(file), 0) != 0) throw std::runtime_error("tap truncate failed");
  return rows;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 5) {
        std::cerr << "usage: splash-logits-dump METALLIB MODEL_DIR CORPUS_DIR OUT_DIR [--stride N] [--chunk N]\n";
        return 2;
      }
      uint32_t stride = 16, chunk = 256;
      for (int i = 5; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--stride") stride = std::atoi(argv[i + 1]);
        else if (option == "--chunk") chunk = std::atoi(argv[i + 1]);
        else throw std::invalid_argument("unknown option " + option);
      }
      const std::filesystem::path corpus = argv[3], out = argv[4];
      std::filesystem::create_directories(out);

      RuntimeResourcesConfig config;
      config.metallibPath = argv[1];
      config.modelPaths = model::ModelPaths{argv[2], {}, {}};
      config.model = model::inspectLocalModel(config.modelPaths.target, std::nullopt);
      config.buildId = "kl-dump";
      config.requestedContextTokens = 2048;
      auto resources = RuntimeResources::create(config);
      model::RuntimeContext context = resources->modelContext();
      model::Runtime runtime(context);

      std::vector<std::filesystem::path> files;
      for (const auto &entry : std::filesystem::directory_iterator(corpus))
        if (entry.path().string().ends_with(".tokens.txt")) files.push_back(entry.path());
      std::sort(files.begin(), files.end());

      // One lane's pages: contiguous ids 0..need-1 backed by whole extents.
      const uint32_t need = 2048 / kv::kPageTokens + 2;
      const uint32_t extent = context.kvPages.extentPages();
      for (uint32_t e = 0; e < (need + extent - 1) / extent; ++e) {
        try {
          if (!context.kvPages.allocateExtent(e)) throw std::runtime_error("KV extent refused");
        } catch (const std::logic_error &) {
          // The warm-up already backs it.
        }
      }
      std::vector<uint32_t> pageTable(need);
      std::iota(pageTable.begin(), pageTable.end(), 0);

      std::FILE *tap = std::tmpfile();
      model::gLogitsDump = tap;
      uint64_t nextId = 1;
      for (const auto &file : files) {
        const std::string fname = file.filename().string();
        const std::string name = fname.substr(0, fname.size() - 11);
        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        std::istringstream words(line);
        std::vector<uint32_t> ids;
        for (uint32_t t; words >> t;) ids.push_back(t);

        std::FILE *rowsOut = std::fopen((out / (name + ".ours.f32")).c_str(), "wb");
        std::vector<uint64_t> positions;
        uint64_t vocabulary = 0;
        for (uint64_t p = stride - 1; p < ids.size(); p += stride) {
          const std::span<const uint32_t> prefix(ids.data(), p + 1);
          ModelRequest request;
          request.id = nextId++;
          request.prompt = prefix;
          request.maxNewTokens = 1;
          runtime.beginColdRequest(request, 0);
          for (uint32_t at = 0; at < prefix.size(); at += chunk) {
            const uint32_t rows = std::min<uint32_t>(chunk, prefix.size() - at);
            BatchPlan plan{.kind = WorkKind::Prefill, .items = {{request.id, rows}}};
            ModelBatchItem item{request.id, at, rows, pageTable};
            item.pageTableRevision = 1;
            item.inputTokens = prefix.subspan(at, rows);
            auto results = runtime.prefill(plan, std::span<const ModelBatchItem>(&item, 1));
            if (results.size() != 1 || results[0].consumedPromptTokens != rows || !results[0].failure.empty())
              throw std::runtime_error("prefill failed at position " + std::to_string(p));
          }
          runtime.end(request.id);
          const std::vector<float> row = drain(tap);
          if (row.empty() || (vocabulary && row.size() != vocabulary))
            throw std::runtime_error("expected one logits row at position " + std::to_string(p));
          vocabulary = row.size();
          std::fwrite(row.data(), sizeof(float), row.size(), rowsOut);
          positions.push_back(p);
        }
        std::fclose(rowsOut);
        std::ofstream meta(out / (name + ".ours.json"));
        meta << "{\"n_vocab\": " << vocabulary << ", \"positions\": [";
        for (size_t i = 0; i < positions.size(); ++i) meta << (i ? ", " : "") << positions[i];
        meta << "]}\n";
        std::printf("%s rows=%zu vocab=%llu\n", name.c_str(), positions.size(),
                    static_cast<unsigned long long>(vocabulary));
        std::fflush(stdout);
      }
      model::gLogitsDump = nullptr;
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "splash-logits-dump: " << error.what() << '\n';
      return 1;
    }
  }
}
