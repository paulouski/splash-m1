// Random 90-byte row reads from a large file, to size the PLE SSD store.
// Usage: ple_ssd_bench <file> <rows> <threads 1..64> <read KiB 4|16> <nocache 0|1> <reads> [data offset]
// Each thread runs one blocking aligned pread at a time, so threads = queue depth.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: %s <file> <rows> <threads> <read KiB 4|16> <nocache 0|1> <reads> [data offset]\n", argv[0]);
    return 2;
  }
  const uint64_t rows = std::strtoull(argv[2], nullptr, 10), reads = std::strtoull(argv[6], nullptr, 10);
  const unsigned threads = std::clamp<unsigned>(std::atoi(argv[3]), 1, 64);
  const uint64_t window = uint64_t(std::atoi(argv[4])) * 1024, offset = argc > 7 ? std::strtoull(argv[7], nullptr, 10) : 0;
  if (rows == 0 || reads == 0 || (window != 4096 && window != 16384)) return 2;
  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    std::perror("open");
    return 1;
  }
  fcntl(fd, F_NOCACHE, std::atoi(argv[5]));
  fcntl(fd, F_RDAHEAD, 0);

  std::vector<std::vector<double>> latencies(threads);
  std::atomic<uint64_t> next{0}, failures{0};
  const auto begin = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      std::mt19937_64 rng(1234 + t);
      void *raw = nullptr;
      posix_memalign(&raw, 16384, 2 * window);
      while (next.fetch_add(1) < reads) {
        const uint64_t row = rng() % rows, at = offset + row * 90, start = at & ~(window - 1);
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t length = at - start + 90 > window ? 2 * window : window; // a straddling row reads two windows
        const ssize_t n = pread(fd, raw, length, off_t(start));
        const auto t1 = std::chrono::steady_clock::now();
        if (n < ssize_t(at - start + 90)) failures.fetch_add(1);
        latencies[t].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      free(raw);
    });
  for (std::thread &thread : pool) thread.join();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

  std::vector<double> all;
  for (auto &v : latencies) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  if (all.empty()) return 1;
  std::printf("threads=%u read=%lluKiB nocache=%s reads=%zu iops=%.0f p50=%.0fus p99=%.0fus max=%.0fus failures=%llu\n", threads,
              (unsigned long long)(window / 1024), argv[5], all.size(), double(all.size()) / seconds, all[all.size() / 2],
              all[std::min(all.size() - 1, all.size() * 99 / 100)], all.back(), (unsigned long long)failures.load());
  close(fd);
  return failures.load() ? 1 : 0;
}
