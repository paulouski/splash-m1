#include "TestChecks.hpp"
#include "model/PleStore.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <unistd.h>

using namespace splash;
using splash::test::rejects;
using splash::test::require;

namespace {

constexpr uint64_t rows = 1'000'000, dataOffset = 4096 + 123;

uint8_t rowByte(uint64_t row, uint32_t j) { return uint8_t(row * 31 + j * 7 + (row >> 8)); }

void writeTable(const std::filesystem::path &path, uint64_t count, uint32_t rowBytes = pleRowBytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  std::vector<char> header(dataOffset, 'h');
  file.write(header.data(), header.size());
  std::vector<char> buffer(uint64_t{65536} * rowBytes);
  for (uint64_t first = 0; first < count; first += 65536) {
    const uint64_t n = std::min<uint64_t>(65536, count - first);
    for (uint64_t r = 0; r < n; ++r)
      for (uint32_t j = 0; j < rowBytes; ++j) buffer[r * rowBytes + j] = char(rowByte(first + r, j));
    file.write(buffer.data(), n * rowBytes);
  }
}

struct Fetched {
  PleStore::Result result;
  std::vector<uint8_t> staging;
  std::vector<uint32_t> slots;
};

Fetched fetch(PleStore &store, const std::vector<uint32_t> &ids, uint32_t rowBytes = pleRowBytes) {
  Fetched f;
  f.staging.assign(ids.size() * rowBytes, 0);
  f.slots.assign(ids.size(), 0);
  f.result = store.fetch(ids, f.staging, f.slots);
  return f;
}

void checkRows(const Fetched &f, const std::vector<uint32_t> &ids, uint32_t rowBytes = pleRowBytes) {
  require(f.result.ok(), "fetch failed: " + f.result.error);
  std::set<uint32_t> unique(ids.begin(), ids.end());
  require(f.result.stats.hits + f.result.stats.misses == unique.size(), "hits + misses != unique ids");
  require(f.result.stats.dupes == ids.size() - unique.size(), "dupes");
  for (size_t k = 0; k < ids.size(); ++k) {
    // Unique rows sit in ascending id order.
    require(uint32_t(std::distance(unique.begin(), unique.find(ids[k]))) == f.slots[k], "slot is not the id's rank");
    for (uint32_t j = 0; j < rowBytes; ++j)
      require(f.staging[uint64_t{f.slots[k]} * rowBytes + j] == rowByte(ids[k], j), "row bytes differ");
  }
}

std::vector<uint32_t> randomIds(std::mt19937 &rng, size_t n, uint32_t limit) {
  std::vector<uint32_t> ids(n);
  for (uint32_t &id : ids) id = rng() % limit;
  return ids;
}

} // namespace

int main() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("ple-store-test-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  const auto path = dir / "table.bin";
  int status = 0;
  try {
    writeTable(path, rows);
    std::mt19937 rng(7);

    {
      // Exact bytes, duplicates, counters, hits on a repeat; edge rows included.
      PleStore store(PleSource(path.string(), dataOffset, rows), 100 * 4096, 8);
      std::vector<uint32_t> ids = randomIds(rng, 300, rows);
      ids.insert(ids.end(), {0, 0, uint32_t(rows - 1), uint32_t(rows - 1), ids[3], ids[3]});
      Fetched first = fetch(store, ids);
      checkRows(first, ids);
      require(first.result.stats.dupes >= 3 && first.result.stats.hits == 0, "first fetch counters");
      require(first.result.stats.bytes >= first.result.stats.misses * 4096ull, "bytes below one window per miss");
      Fetched again = fetch(store, ids);
      checkRows(again, ids);
      require(again.result.stats.misses == 0 && again.result.stats.bytes == 0, "repeat fetch read the device");
      require(again.result.stats.hits == first.result.stats.misses, "repeat hits");
      require(again.staging == first.staging && again.slots == first.slots, "repeat differs");
      require(fetch(store, {}).result.ok(), "empty fetch");
    }

    {
      // CLOCK: capacity 64, far more keys than that; a re-referenced set survives a scan.
      PleStore store(PleSource(path.string(), dataOffset, rows), 100 * 64, 4);
      std::vector<uint32_t> hot;
      for (uint32_t i = 0; i < 8; ++i) hot.push_back(1000 + i * 977);
      checkRows(fetch(store, hot), hot);
      uint32_t scan = 50000;
      for (int round = 0; round < 200; ++round) {
        std::vector<uint32_t> ids = hot;
        for (int i = 0; i < 24; ++i) ids.push_back(scan++);
        Fetched f = fetch(store, ids);
        checkRows(f, ids);
        if (round > 0) require(f.result.stats.hits >= 8, "hot rows were evicted by a scan");
        require(f.result.stats.misses >= 24, "scan rows cannot all be cached");
      }
      std::vector<uint32_t> old;
      for (uint32_t i = 0; i < 24; ++i) old.push_back(50000 + i);
      require(fetch(store, old).result.stats.misses == 24, "evicted scan rows are still cached");
      // Heavy random churn through a tiny cache stays exact.
      PleStore tiny(PleSource(path.string(), dataOffset, rows), 100 * 5, 2);
      for (int round = 0; round < 100; ++round) {
        std::vector<uint32_t> ids = randomIds(rng, 40, 30);
        checkRows(fetch(tiny, ids), ids);
      }
      // Disabled cache never hits.
      PleStore none(PleSource(path.string(), dataOffset, rows), 0, 4);
      std::vector<uint32_t> ids = randomIds(rng, 50, rows);
      checkRows(fetch(none, ids), ids);
      Fetched twice = fetch(none, ids);
      checkRows(twice, ids);
      require(twice.result.stats.hits == 0, "disabled cache hit");
      // clear() empties the cache.
      PleStore cleared(PleSource(path.string(), dataOffset, rows), 100 * 64, 2);
      fetch(cleared, ids);
      cleared.clearCache();
      require(fetch(cleared, ids).result.stats.hits == 0, "clear kept rows");
    }

    {
      // The same result for 1, 8 and 32 threads.
      std::vector<uint32_t> ids = randomIds(rng, 6000, rows);
      Fetched reference;
      for (uint32_t threads : {1u, 8u, 32u}) {
        PleStore store(PleSource(path.string(), dataOffset, rows), 0, threads);
        Fetched f = fetch(store, ids);
        checkRows(f, ids);
        if (threads == 1) reference = f;
        require(f.staging == reference.staging && f.slots == reference.slots, "thread count changed the result");
        // A step at the inline limit and just above it.
        for (size_t n : {size_t{1}, size_t{8}, size_t{9}}) {
          std::vector<uint32_t> few(ids.begin(), ids.begin() + n);
          checkRows(fetch(store, few), few);
        }
      }
    }

    {
      // Failures. An out-of-range id writes nothing and does not poison.
      PleStore store(PleSource(path.string(), dataOffset, rows), 100 * 64, 4);
      std::vector<uint32_t> ids{5, 6, uint32_t(rows)};
      Fetched bad = fetch(store, ids);
      require(!bad.result.ok() && !store.poisoned(), "out-of-range id");
      require(std::all_of(bad.staging.begin(), bad.staging.end(), [](uint8_t b) { return b == 0; }), "wrote on error");
      checkRows(fetch(store, {5, 6}), {5, 6});
      // Truncated file: identity check fails, store is poisoned, later fetches fail even for cached rows.
      std::filesystem::resize_file(path, dataOffset + (rows / 2) * pleRowBytes);
      Fetched cut = fetch(store, {900000, 5});
      require(!cut.result.ok() && store.poisoned(), "truncated file went unnoticed");
      require(!fetch(store, {5}).result.ok(), "poisoned store served a fetch");
      writeTable(path, rows);
    }

    {
      // File replaced by another with the same size.
      PleStore store(PleSource(path.string(), dataOffset, rows), 0, 4);
      checkRows(fetch(store, {1, 2, 3}), {1, 2, 3});
      const auto other = dir / "other.bin";
      writeTable(other, rows);
      std::filesystem::rename(other, path);
      Fetched f = fetch(store, {10, 20, 30});
      require(!f.result.ok() && store.poisoned(), "replaced file went unnoticed");
      require(f.result.error.find("replaced") != std::string::npos, "replacement message");
    }

    {
      // Open-time failures: too short for the rows, validation hook, missing file; a symlinked shard opens.
      rejects([&] { PleSource(path.string(), dataOffset, rows + 1); }, "past the end", "short shard accepted");
      rejects([&] { PleSource(path.string(), dataOffset, rows, pleRowBytes, [](const PleSource &) { return std::string("bad shard"); }); },
              "bad shard", "validation hook ignored");
      rejects([&] { PleSource((dir / "missing").string(), 0, 1); }, "open", "missing file accepted");
      std::filesystem::create_symlink(path, dir / "link.bin");
      PleStore linked(PleSource((dir / "link.bin").string(), dataOffset, rows), 0, 2);
      checkRows(fetch(linked, {7, 8}), {7, 8});
      require(fetch(linked, {9}).result.ok() && !linked.poisoned(), "identity check failed through a symlink");
    }

    for (const uint32_t width : {pleMaxRowBytes, 100u}) {
      // Wider rows (Q8_0, Q4_1) straddling pages: exact bytes, cache hits, truncation.
      const auto wide = dir / "wide.bin";
      writeTable(wide, rows, width);
      PleStore store(PleSource(wide.string(), dataOffset, rows, width), 20000 * (width + 10), 8);
      std::vector<uint32_t> ids = randomIds(rng, 400, rows);
      ids.insert(ids.end(), {0, uint32_t(rows - 1), ids[5]});
      checkRows(fetch(store, ids, width), ids, width);
      Fetched again = fetch(store, ids, width);
      checkRows(again, ids, width);
      require(again.result.stats.misses == 0, "wide repeat read the device");
      rejects([&] { PleSource(wide.string(), dataOffset, rows + 1, width); }, "past the end", "wide short shard accepted");
    }

    std::cout << "ple store tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    status = 1;
  }
  std::filesystem::remove_all(dir);
  return status;
}
