// Requantizes (split) GGUF shards tensor by tensor with a per-tensor type recipe and an optional imatrix, writing one
// GGUF. Unlisted tensors are copied as stored. Built against llama.cpp's ggml; see dev/tools/fn_quantize.md.
//
//   fn_quantize --out OUT.gguf [--recipe FILE] [--imatrix FILE] [--only REGEX] [--drop REGEX] [--threads N]
//               [--dry-run] [--info] SHARD...
// Recipe lines: `REGEX = TYPE` (ECMAScript, full match on the tensor name, first match wins); `#` starts a comment.
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Shard {
  std::string path;
  gguf_context *gguf = nullptr;
  ggml_context *meta = nullptr;
  const uint8_t *data = nullptr; // mmap of the whole file
  size_t size = 0;
};

struct Item {
  const Shard *shard;
  ggml_tensor *tensor;
  size_t offset; // of the tensor data in the shard's mapping
  ggml_type type; // output type
  bool requantize;
};

struct Imatrix {
  std::vector<float> values; // sums / count per expert
  int64_t experts = 0, emptyExperts = 0;
};

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

ggml_type typeByName(const std::string &name) {
  for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
    const char *candidate = ggml_type_name(static_cast<ggml_type>(i));
    if (candidate && strcasecmp(candidate, name.c_str()) == 0) return static_cast<ggml_type>(i);
  }
  fail("unknown type " + name);
}

std::map<std::string, Imatrix> loadImatrix(const std::string &path) {
  ggml_context *ctx = nullptr;
  gguf_context *gguf = gguf_init_from_file(path.c_str(), {false, &ctx});
  if (!gguf) fail("cannot read imatrix " + path);
  std::map<std::string, ggml_tensor *> sums, counts;
  for (ggml_tensor *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
    const std::string name = t->name;
    const auto ends = [&](const char *suffix) {
      const size_t n = strlen(suffix);
      return name.size() > n && name.compare(name.size() - n, n, suffix) == 0 ? name.substr(0, name.size() - n) : std::string();
    };
    if (auto base = ends(".in_sum2"); !base.empty()) sums[base] = t;
    else if (auto base2 = ends(".counts"); !base2.empty()) counts[base2] = t;
  }
  std::map<std::string, Imatrix> result;
  for (const auto &[name, sum] : sums) {
    const auto count = counts.find(name);
    if (count == counts.end() || sum->type != GGML_TYPE_F32 || count->second->type != GGML_TYPE_F32)
      fail("imatrix entry " + name + " lacks F32 sums or counts");
    const int64_t values = ggml_nelements(sum), experts = ggml_nelements(count->second), width = values / experts;
    Imatrix &entry = result[name];
    entry.experts = experts;
    entry.values.resize(values);
    for (int64_t e = 0; e < experts; ++e) {
      const float n = static_cast<const float *>(count->second->data)[e];
      if (n <= 0) ++entry.emptyExperts;
      for (int64_t i = 0; i < width; ++i)
        entry.values[e * width + i] = n > 0 ? static_cast<const float *>(sum->data)[e * width + i] / n : 1.0f;
    }
    for (float v : entry.values)
      if (!std::isfinite(v)) fail("imatrix entry " + name + " has a non-finite value");
  }
  gguf_free(gguf);
  ggml_free(ctx);
  return result;
}

// ggml's Q2_0 reference sets d = amax, so levels {-d, 0, d, 2d} fit poorly; search d (either sign) for the least
// weighted error instead. Block: fp16 d, then 64 codes (q + 1) in 2 bits, low bits first.
void quantizeRowQ2_0(const float *x, const float *qw, int64_t n, uint8_t *out) {
  for (int64_t b = 0; b < n / 64; ++b, x += 64, out += 18) {
    float weight[64], amax = 0, mean2 = 0;
    for (int j = 0; j < 64; ++j) {
      amax = std::max(amax, std::fabs(x[j]));
      mean2 += x[j] * x[j] / 64;
    }
    for (int j = 0; j < 64; ++j) weight[j] = (qw ? qw[b * 64 + j] : 1.0f) * std::sqrt(mean2 + x[j] * x[j]);
    float bestD = 0, bestError = INFINITY;
    int8_t best[64] = {};
    for (const float sign : {1.0f, -1.0f})
      for (const float divisor : {1.0f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f}) {
        float d = amax > 0 ? sign * amax / divisor : 0;
        int8_t q[64];
        float error = 0;
        for (int iteration = 0; iteration < 4; ++iteration) {
          d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
          const float inverse = d != 0 ? 1.0f / d : 0.0f;
          float numerator = 0, denominator = 0;
          error = 0;
          for (int j = 0; j < 64; ++j) {
            q[j] = static_cast<int8_t>(std::clamp<int>(static_cast<int>(std::lround(x[j] * inverse)), -1, 2));
            numerator += weight[j] * x[j] * q[j];
            denominator += weight[j] * q[j] * q[j];
            error += weight[j] * (x[j] - q[j] * d) * (x[j] - q[j] * d);
          }
          if (error < bestError) {
            bestError = error;
            bestD = d;
            memcpy(best, q, sizeof(q));
          }
          if (denominator <= 0) break;
          d = numerator / denominator;
        }
      }
    const uint16_t half = ggml_fp32_to_fp16(bestD);
    memcpy(out, &half, 2);
    memset(out + 2, 0, 16);
    for (int j = 0; j < 64; ++j) out[2 + j / 4] |= static_cast<uint8_t>((best[j] + 1) << (2 * (j % 4)));
  }
}

void dequantize(const Item &item, int64_t firstRow, int64_t rows, float *out) {
  const ggml_tensor *t = item.tensor;
  const int64_t width = t->ne[0];
  const uint8_t *src = item.shard->data + item.offset + static_cast<size_t>(firstRow) * t->nb[1];
  if (t->type == GGML_TYPE_F32) {
    memcpy(out, src, sizeof(float) * width * rows);
    return;
  }
  const ggml_type_traits *traits = ggml_get_type_traits(t->type);
  if (!traits->to_float) fail(std::string("cannot dequantize ") + ggml_type_name(t->type));
  for (int64_t r = 0; r < rows; ++r) traits->to_float(src + r * t->nb[1], out + r * width, width);
}

} // namespace

int main(int argc, char **argv) {
  std::string out, recipePath, imatrixPath, only, drop;
  int threads = 8;
  bool dryRun = false, info = false;
  std::vector<std::string> inputs;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&] {
      if (i + 1 >= argc) fail("missing value for " + arg);
      return std::string(argv[++i]);
    };
    if (arg == "--out") out = next();
    else if (arg == "--recipe") recipePath = next();
    else if (arg == "--imatrix") imatrixPath = next();
    else if (arg == "--only") only = next();
    else if (arg == "--drop") drop = next();
    else if (arg == "--threads") threads = std::stoi(next());
    else if (arg == "--dry-run") dryRun = true;
    else if (arg == "--info") info = true;
    else inputs.push_back(arg);
  }
  try {
    if (inputs.empty() || (out.empty() && !info && !dryRun)) fail("usage: fn_quantize --out OUT [--recipe F] [--imatrix F] [--only R] [--drop R] [--threads N] [--dry-run|--info] SHARD...");
    std::vector<std::pair<std::regex, ggml_type>> recipe;
    if (!recipePath.empty()) {
      std::ifstream file(recipePath);
      if (!file) fail("cannot read recipe " + recipePath);
      for (std::string line; std::getline(file, line);) {
        line = line.substr(0, line.find('#'));
        const size_t eq = line.rfind('=');
        if (eq == std::string::npos) continue;
        const auto trim = [](std::string s) {
          s.erase(0, s.find_first_not_of(" \t"));
          s.erase(s.find_last_not_of(" \t\r") + 1);
          return s;
        };
        recipe.emplace_back(std::regex(trim(line.substr(0, eq))), typeByName(trim(line.substr(eq + 1))));
      }
    }
    std::map<std::string, Imatrix> imatrix;
    if (!imatrixPath.empty()) imatrix = loadImatrix(imatrixPath);
    if (info) {
      for (const auto &[name, e] : imatrix)
        printf("%-48s values=%zu experts=%lld empty=%lld\n", name.c_str(), e.values.size(), (long long)e.experts, (long long)e.emptyExperts);
      if (inputs.empty()) return 0;
    }

    std::vector<std::unique_ptr<Shard>> shards;
    std::vector<Item> items;
    for (const std::string &path : inputs) {
      auto shard = std::make_unique<Shard>();
      shard->path = path;
      shard->gguf = gguf_init_from_file(path.c_str(), {true, &shard->meta});
      if (!shard->gguf) fail("cannot read " + path);
      const int fd = open(path.c_str(), O_RDONLY);
      struct stat st;
      if (fd < 0 || fstat(fd, &st) != 0) fail("cannot open " + path);
      shard->size = st.st_size;
      void *map = mmap(nullptr, shard->size, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (map == MAP_FAILED) fail("cannot map " + path);
      shard->data = static_cast<const uint8_t *>(map);
      const size_t base = gguf_get_data_offset(shard->gguf);
      for (int64_t i = 0; i < gguf_get_n_tensors(shard->gguf); ++i) {
        const char *name = gguf_get_tensor_name(shard->gguf, i);
        ggml_tensor *t = ggml_get_tensor(shard->meta, name);
        if (!only.empty() && !std::regex_match(name, std::regex(only))) continue;
        if (!drop.empty() && std::regex_match(name, std::regex(drop))) continue;
        Item item{shard.get(), t, base + gguf_get_tensor_offset(shard->gguf, i), t->type, false};
        for (const auto &[pattern, type] : recipe)
          if (std::regex_match(name, pattern)) {
            item.type = type;
            break;
          }
        item.requantize = item.type != t->type;
        if (item.requantize && t->ne[0] % ggml_blck_size(item.type) != 0)
          fail(std::string(name) + ": row of " + std::to_string(t->ne[0]) + " does not fit " + ggml_type_name(item.type));
        if (item.offset + ggml_nbytes(t) > shard->size) fail(std::string(name) + " lies outside " + path + " (incomplete download?)");
        items.push_back(item);
      }
      shards.push_back(std::move(shard));
    }

    std::map<std::string, std::pair<int, double>> groups; // type -> (tensors, GiB)
    double totalBytes = 0;
    gguf_context *outGguf = gguf_init_empty();
    gguf_set_kv(outGguf, shards[0]->gguf);
    for (const char *key : {"split.no", "split.count", "split.tensors.count"}) gguf_remove_key(outGguf, key);
    ggml_context *outMeta = ggml_init({ggml_tensor_overhead() * (items.size() + 8), nullptr, true});
    for (const Item &item : items) {
      const ggml_tensor *t = item.tensor;
      ggml_tensor *copy = ggml_new_tensor(outMeta, item.type, ggml_n_dims(t), t->ne);
      ggml_set_name(copy, t->name);
      gguf_add_tensor(outGguf, copy);
      auto &group = groups[std::string(t->name).find("_exps") != std::string::npos ? std::string("experts ") + ggml_type_name(item.type)
                                                                                       : std::string("other ") + ggml_type_name(item.type)];
      ++group.first;
      group.second += ggml_nbytes(copy) / 1073741824.0;
      totalBytes += ggml_nbytes(copy);
    }
    for (const auto &[name, g] : groups) printf("%-20s %4d tensors %9.3f GiB\n", name.c_str(), g.first, g.second);
    printf("total %.3f GiB, %zu tensors\n", totalBytes / 1073741824.0, items.size());
    if (dryRun || out.empty()) return 0;

    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    if (!file) fail("cannot write " + out);
    std::vector<char> meta(gguf_get_meta_size(outGguf));
    gguf_get_meta_data(outGguf, meta.data());
    file.write(meta.data(), meta.size());
    const size_t alignment = gguf_get_alignment(outGguf);
    size_t written = 0;
    const auto pad = [&](size_t size) {
      const size_t padded = (size + alignment - 1) / alignment * alignment;
      static const char zeros[256] = {};
      for (size_t n = padded - size; n;) {
        const size_t chunk = std::min(n, sizeof(zeros));
        file.write(zeros, chunk);
        n -= chunk;
      }
    };
    const auto start = std::chrono::steady_clock::now();
    size_t index = 0;
    for (const Item &item : items) {
      const ggml_tensor *t = item.tensor;
      const int64_t width = t->ne[0], rowsPerExpert = t->ne[1], experts = ggml_n_dims(t) == 3 ? t->ne[2] : 1;
      const int64_t rows = rowsPerExpert * experts;
      std::vector<uint8_t> quantized;
      const uint8_t *bytes = item.shard->data + item.offset;
      size_t size = ggml_nbytes(t);
      if (item.requantize) {
        const Imatrix *weights = nullptr;
        if (const auto found = imatrix.find(t->name); found != imatrix.end()) {
          if (static_cast<int64_t>(found->second.values.size()) != width * experts)
            fail(std::string("imatrix size differs from tensor ") + t->name);
          weights = &found->second;
        } else if (ggml_quantize_requires_imatrix(item.type)) {
          fail(std::string(t->name) + ": " + ggml_type_name(item.type) + " requires an imatrix entry");
        }
        // Units: one expert, or 64 rows of a plain matrix; a plain matrix is written in segments of ~64M values.
        const int64_t unitRows = experts > 1 ? rowsPerExpert : 64;
        const int64_t segmentRows = experts > 1 ? rows : std::max<int64_t>(unitRows, (int64_t{1} << 26) / width / unitRows * unitRows);
        const size_t rowBytes = ggml_row_size(item.type, width);
        size = rowBytes * rows;
        for (int64_t segment = 0; segment < rows; segment += segmentRows) {
          const int64_t segmentCount = std::min(segmentRows, rows - segment), units = (segmentCount + unitRows - 1) / unitRows;
          quantized.resize(rowBytes * segmentCount);
          std::atomic<int64_t> next{0};
          std::vector<std::thread> pool;
          for (int w = 0; w < threads; ++w)
            pool.emplace_back([&] {
              std::vector<float> floats;
              for (int64_t u; (u = next++) < units;) {
                const int64_t first = u * unitRows, count = std::min(unitRows, segmentCount - first);
                floats.resize(width * count);
                dequantize(item, segment + first, count, floats.data());
                const float *w = weights ? weights->values.data() + ((segment + first) / rowsPerExpert) * width : nullptr;
                size_t got = rowBytes * count;
                if (item.type == GGML_TYPE_Q2_0)
                  for (int64_t r = 0; r < count; ++r)
                    quantizeRowQ2_0(floats.data() + r * width, w, width, quantized.data() + (first + r) * rowBytes);
                else
                  got = ggml_quantize_chunk(item.type, floats.data(), quantized.data() + first * rowBytes, 0, count, width, w);
                if (got != rowBytes * count) fail("quantizer returned an unexpected size");
              }
            });
          for (auto &thread : pool) thread.join();
          file.write(reinterpret_cast<const char *>(quantized.data()), quantized.size());
        }
        bytes = nullptr;
      }
      if (bytes) file.write(reinterpret_cast<const char *>(bytes), size);
      pad(size);
      written += size;
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      printf("[%4zu/%zu] %-40s %-6s -> %-8s %9.1f MiB  %.0fs\n", ++index, items.size(), t->name, ggml_type_name(t->type),
             ggml_type_name(item.type), size / 1048576.0, seconds);
      fflush(stdout);
    }
    file.close();
    if (!file) fail("write failed");
    printf("wrote %s (%.3f GiB data)\n", out.c_str(), written / 1073741824.0);
  } catch (const std::exception &error) {
    fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
