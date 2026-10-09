#include "model/PleStore.hpp"

#include "AwakeClock.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash {

namespace {

constexpr uint64_t pageBytes = 4096;

std::string errnoMessage(const char *what) { return std::string(what) + ": " + std::strerror(errno); }

} // namespace

PleHasher::PleHasher(PleHashParams params) : params_(std::move(params)) {
  if (params_.ngram < 2 || params_.ngram > 4) throw std::invalid_argument("PLE n-gram must be 2 to 4");
  if (params_.heads() == 0 || params_.heads() > pleMaximumHeads)
    throw std::invalid_argument("PLE head count is outside 1 to 16");
  if (params_.vocabularies.size() != params_.heads() || params_.offsets.size() != params_.heads())
    throw std::invalid_argument("PLE head tables do not match the head count");
  for (uint32_t vocabulary : params_.vocabularies)
    if (vocabulary == 0) throw std::invalid_argument("PLE head vocabulary is empty");
}

// PleSource

PleSource::PleSource(const std::string &path, uint64_t dataOffset, uint64_t rows, const Validate &validate)
    : path_(path), dataOffset_(dataOffset), rows_(rows) {
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd_ < 0) throw std::runtime_error("PLE shard " + errnoMessage("open"));
  fcntl(fd_, F_NOCACHE, 1);
  fcntl(fd_, F_RDAHEAD, 0);
  struct stat st {};
  if (fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("PLE shard " + errnoMessage("fstat"));
  }
  identity_ = {uint64_t(st.st_dev), uint64_t(st.st_ino), uint64_t(st.st_size), int64_t(st.st_mtimespec.tv_sec),
               int64_t(st.st_mtimespec.tv_nsec)};
  std::string error;
  if (dataOffset > identity_.size || rows > (identity_.size - dataOffset) / pleRowBytes)
    error = "PLE table runs past the end of the shard";
  else if (validate)
    error = validate(*this);
  if (!error.empty()) {
    ::close(fd_);
    throw std::runtime_error(error);
  }
}

PleSource::PleSource(PleSource &&other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)), dataOffset_(other.dataOffset_),
      rows_(other.rows_), identity_(other.identity_) {}

PleSource::~PleSource() {
  if (fd_ >= 0) ::close(fd_);
}

std::string PleSource::checkIdentity() const {
  struct stat open {}, named {};
  if (fstat(fd_, &open) != 0) return errnoMessage("PLE shard fstat");
  if (stat(path_.c_str(), &named) != 0) return errnoMessage("PLE shard stat");
  const Identity now{uint64_t(open.st_dev), uint64_t(open.st_ino), uint64_t(open.st_size),
                     int64_t(open.st_mtimespec.tv_sec), int64_t(open.st_mtimespec.tv_nsec)};
  if (now != identity_ || named.st_dev != open.st_dev || named.st_ino != open.st_ino)
    return "PLE shard changed or was replaced";
  return {};
}

// PleRowCache

PleRowCache::PleRowCache(uint64_t bytes) {
  // 90 B row + key + state + index slots at 40-60% load.
  capacity_ = static_cast<uint32_t>(std::min<uint64_t>(bytes / 100, 0x40000000u));
  if (capacity_ == 0) return;
  uint64_t slots = 4;
  while (slots * 6 < uint64_t{capacity_} * 10) slots *= 2; // capacity <= 0.6 * slots
  mask_ = static_cast<uint32_t>(slots - 1);
  index_.assign(slots, 0);
  keys_.assign(capacity_, 0);
  state_.assign(capacity_, 0);
  rows_.assign(uint64_t{capacity_} * pleRowBytes, 0);
}

static uint32_t cacheHome(uint32_t id, uint32_t mask) { return uint32_t((uint64_t{id} * 0x9E3779B97F4A7C15ull) >> 32) & mask; }

uint32_t PleRowCache::probe(uint32_t id) const {
  for (uint32_t i = cacheHome(id, mask_);; i = (i + 1) & mask_) {
    const uint32_t entry = index_[i];
    if (entry == 0 || keys_[entry - 1] == id) return i;
  }
}

const uint8_t *PleRowCache::find(uint32_t id) {
  if (capacity_ == 0) return nullptr;
  const uint32_t entry = index_[probe(id)];
  if (entry == 0) return nullptr;
  state_[entry - 1] = 2;
  return rows_.data() + uint64_t{entry - 1} * pleRowBytes;
}

void PleRowCache::erase(uint32_t slot) {
  uint32_t hole = probe(keys_[slot]);
  for (uint32_t next = (hole + 1) & mask_; index_[next] != 0; next = (next + 1) & mask_) {
    const uint32_t home = cacheHome(keys_[index_[next] - 1], mask_);
    if (((next - home) & mask_) >= ((next - hole) & mask_)) {
      index_[hole] = index_[next];
      hole = next;
    }
  }
  index_[hole] = 0;
  state_[slot] = 0;
  --size_;
}

void PleRowCache::insert(uint32_t id, const uint8_t *row, bool hot) {
  if (capacity_ == 0) return;
  uint32_t at = probe(id);
  uint32_t slot;
  if (index_[at] != 0) {
    slot = index_[at] - 1;
  } else {
    if (size_ < capacity_) {
      slot = size_; // slots fill in order until the first eviction
    } else {
      while (state_[hand_] == 2) {
        state_[hand_] = 1;
        hand_ = hand_ + 1 == capacity_ ? 0 : hand_ + 1;
      }
      slot = hand_;
      hand_ = hand_ + 1 == capacity_ ? 0 : hand_ + 1;
      erase(slot);
      at = probe(id);
    }
    keys_[slot] = id;
    index_[at] = slot + 1;
    ++size_;
  }
  state_[slot] = hot ? 2 : 1;
  std::memcpy(rows_.data() + uint64_t{slot} * pleRowBytes, row, pleRowBytes);
}

void PleRowCache::clear() {
  std::fill(index_.begin(), index_.end(), 0);
  std::fill(state_.begin(), state_.end(), 0);
  size_ = hand_ = 0;
}

// PleReader

PleReader::PleReader(uint32_t threads) {
  for (uint32_t i = 1; i < threads; ++i) workers_.emplace_back([this] { run(); });
}

PleReader::~PleReader() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread &worker : workers_) worker.join();
}

void PleReader::work(Job &job) {
  alignas(pageBytes) uint8_t window[2 * pageBytes];
  for (;;) {
    const size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
    if (i >= job.count || job.failed.load(std::memory_order_relaxed)) return;
    const Miss miss = job.misses[i];
    const uint64_t offset = job.source->rowOffset(miss.id);
    const uint64_t start = offset & ~(pageBytes - 1), skew = offset - start;
    const uint64_t length = skew + pleRowBytes > pageBytes ? 2 * pageBytes : pageBytes;
    uint64_t have = 0;
    std::string error;
    while (have < skew + pleRowBytes) {
      const ssize_t n = pread(job.source->fd(), window + have, length - have, off_t(start + have));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        error = n < 0 ? errnoMessage("PLE pread") : "PLE short read (shard truncated)";
        break;
      }
      have += uint64_t(n);
    }
    if (!error.empty()) {
      std::lock_guard lock(job.errorMutex);
      if (job.error.empty()) job.error = error;
      job.failed.store(true);
      return;
    }
    std::memcpy(job.out + uint64_t{miss.slot} * pleRowBytes, window + skew, pleRowBytes);
    job.bytes.fetch_add(have, std::memory_order_relaxed);
  }
}

void PleReader::run() {
  uint64_t seen = 0;
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;
    Job *job = job_;
    lock.unlock();
    work(*job);
    lock.lock();
    if (--active_ == 0) done_.notify_all();
  }
}

uint64_t PleReader::read(const PleSource &source, std::span<const Miss> misses, uint8_t *out, std::string &error) {
  Job job;
  job.source = &source;
  job.misses = misses.data();
  job.count = misses.size();
  job.out = out;
  if (misses.size() <= inlineMisses || workers_.empty()) {
    work(job);
  } else {
    {
      std::lock_guard lock(mutex_);
      job_ = &job;
      active_ = static_cast<uint32_t>(workers_.size());
      ++generation_;
    }
    wake_.notify_all();
    work(job);
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return active_ == 0; });
  }
  error = job.error;
  return job.bytes.load();
}

// PleStore

PleStore::PleStore(PleSource source, uint64_t cacheBytes, uint32_t threads)
    : source_(std::move(source)), cache_(cacheBytes), reader_(std::max(threads, 1u)) {}

PleStore::Result PleStore::fetch(std::span<const uint32_t> ids, std::span<uint8_t> staging,
                                 std::span<uint32_t> slots) {
  const auto begin = AwakeClock::now();
  Result result;
  auto fail = [&](std::string error, bool poison) {
    poisoned_ = poisoned_ || poison;
    result.error = std::move(error);
    return result;
  };
  if (poisoned_) return fail("PLE store is poisoned by an earlier failure", false);
  if (slots.size() < ids.size()) return fail("PLE slot span is smaller than the ids", false);
  order_.resize(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] >= source_.rows()) return fail("PLE row id " + std::to_string(ids[i]) + " is out of range", false);
    order_[i] = uint64_t{ids[i]} << 32 | i;
  }
  std::sort(order_.begin(), order_.end());
  uniqueIds_.clear();
  for (size_t i = 0; i < order_.size(); ++i) {
    const uint32_t id = static_cast<uint32_t>(order_[i] >> 32);
    if (uniqueIds_.empty() || uniqueIds_.back() != id) uniqueIds_.push_back(id);
    slots[static_cast<uint32_t>(order_[i])] = static_cast<uint32_t>(uniqueIds_.size() - 1);
  }
  if (staging.size() < uniqueIds_.size() * pleRowBytes) return fail("PLE staging buffer is too small", false);
  result.stats.dupes = static_cast<uint32_t>(ids.size() - uniqueIds_.size());
  misses_.clear();
  for (uint32_t slot = 0; slot < uniqueIds_.size(); ++slot) {
    if (const uint8_t *row = cache_.find(uniqueIds_[slot])) {
      std::memcpy(staging.data() + uint64_t{slot} * pleRowBytes, row, pleRowBytes);
      ++result.stats.hits;
    } else {
      misses_.push_back({uniqueIds_[slot], slot});
    }
  }
  result.stats.misses = static_cast<uint32_t>(misses_.size());
  if (!misses_.empty()) {
    if (std::string changed = source_.checkIdentity(); !changed.empty()) return fail(std::move(changed), true);
    std::string error;
    result.stats.bytes = reader_.read(source_, misses_, staging.data(), error);
    if (!error.empty()) return fail(std::move(error), true);
    for (const PleReader::Miss &miss : misses_)
      cache_.insert(miss.id, staging.data() + uint64_t{miss.slot} * pleRowBytes, false);
  }
  result.stats.ns = uint64_t((AwakeClock::now() - begin).count());
  return result;
}

} // namespace splash
