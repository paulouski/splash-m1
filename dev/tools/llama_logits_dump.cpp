// Full-vocab fp32 logits of every position of token-id segments, from llama.cpp (CPU).
//
//   llama-logits-dump MODEL.gguf CORPUS_DIR OUT_DIR [threads] [n_batch]
//
// CORPUS_DIR/<seg>.tokens.txt: one line of token ids. Writes OUT_DIR/<seg>.ref.f32
// (n rows x n_vocab raw logits; row p is the distribution of token p+1) and
// OUT_DIR/<seg>.ref.json ({"n":..,"n_vocab":..}).
// Build: c++ -std=c++20 -O2 -I$L/include -I$L/ggml/include llama_logits_dump.cpp -L$L/build/bin
//   -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o llama-logits-dump   (L = llama.cpp checkout)
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s MODEL.gguf CORPUS_DIR OUT_DIR [threads] [n_batch]\n", argv[0]);
    return 2;
  }
  const std::filesystem::path corpus = argv[2], out = argv[3];
  const int threads = argc > 4 ? std::atoi(argv[4]) : 8;
  const int batch = argc > 5 ? std::atoi(argv[5]) : 512;
  std::filesystem::create_directories(out);

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  static ggml_backend_dev_t noDevices[] = {nullptr};  // CPU only
  mp.devices = noDevices;
  mp.n_gpu_layers = 0;
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
  llama_model *model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 1;
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 2048;
  cp.n_batch = batch;
  cp.n_ubatch = batch;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) return 1;

  std::vector<std::filesystem::path> files;
  for (const auto &entry : std::filesystem::directory_iterator(corpus))
    if (entry.path().string().ends_with(".tokens.txt")) files.push_back(entry.path());
  std::sort(files.begin(), files.end());

  for (const auto &file : files) {
    const std::string name = file.filename().string().substr(0, file.filename().string().size() - 11);
    std::ifstream in(file);
    std::string line;
    std::getline(in, line);
    std::istringstream words(line);
    std::vector<llama_token> ids;
    for (llama_token t; words >> t;) ids.push_back(t);
    const int n = static_cast<int>(ids.size());
    if (n == 0 || n > static_cast<int>(cp.n_ctx)) return 1;

    llama_memory_clear(llama_get_memory(ctx), true);
    std::FILE *rows = std::fopen((out / (name + ".ref.f32")).c_str(), "wb");
    for (int at = 0; at < n; at += batch) {
      const int count = std::min(batch, n - at);
      llama_batch b = llama_batch_init(count, 0, 1);
      b.n_tokens = count;
      for (int i = 0; i < count; ++i) {
        b.token[i] = ids[at + i];
        b.pos[i] = at + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
      }
      if (llama_decode(ctx, b) != 0) return 1;
      for (int i = 0; i < count; ++i)
        std::fwrite(llama_get_logits_ith(ctx, i), sizeof(float), vocab, rows);
      llama_batch_free(b);
    }
    std::fclose(rows);
    std::ofstream(out / (name + ".ref.json")) << "{\"n\": " << n << ", \"n_vocab\": " << vocab << "}\n";
    std::printf("%s n=%d vocab=%d\n", name.c_str(), n, vocab);
    std::fflush(stdout);
  }
  llama_free(ctx);
  llama_model_free(model);
  return 0;
}
