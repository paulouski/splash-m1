#pragma once

// SPLASH_DRAFT_SHORTLIST: the token ids the MTP draft head scores, in the row order of a gathered copy of the logits
// head. The set is the request's history, the anchor and the top ids of the target's earlier rows; it only grows
// (appending the new ids in ascending order, for gather locality) until the capacity, then restarts from the history.

#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace splash::model {

class DraftShortlist final {
public:
  static constexpr uint32_t kTileRows = QUANT_TILE_ROWS;

  // `seedAll`: every id from the first update (a positive control: the draft then equals the full head).
  DraftShortlist(uint32_t vocabulary, uint32_t capacity, uint32_t gate, bool seedAll = false)
      : vocabulary_(vocabulary), capacity_(capacity), gate_(gate), seedAll_(seedAll), mask_(vocabulary, 0) {}

  // Ids the target ranked highly, merged by the next update.
  void addPicks(std::span<const uint32_t> ids) { picks_.insert(picks_.end(), ids.begin(), ids.end()); }

  // Rows [first, first + count) of the shortlist head to gather from map()[first, first + count): the new ids, then
  // copies of id 0 padding the set to whole tiles.
  struct Update {
    uint32_t first = 0;
    uint32_t count = 0;
  };

  // Forgets the set (its gathered rows are no longer trusted).
  void reset() {
    clear();
    picks_.clear();
    owner_ = ~uint64_t{0};
  }

  Update update(uint64_t owner, std::span<const uint32_t> history, uint32_t anchor) {
    if (owner != owner_ || history.size() < scanned_) {
      clear();
      picks_.clear();
    }
    owner_ = owner;
    std::vector<uint32_t> fresh;
    const auto take = [&](uint32_t id) {
      if (id < vocabulary_ && !mask_[id]) {
        mask_[id] = 1;
        fresh.push_back(id);
      }
    };
    if (seedAll_ && ids_.empty()) {
      fresh.resize(vocabulary_);
      std::iota(fresh.begin(), fresh.end(), 0u);
      std::fill(mask_.begin(), mask_.end(), 1);
    }
    for (size_t i = scanned_; i < history.size(); ++i) take(history[i]);
    take(anchor);
    const std::vector<uint32_t> picks = std::move(picks_);
    picks_.clear();
    for (const uint32_t id : picks) take(id);
    scanned_ = history.size();
    if (ids_.size() + fresh.size() > capacity_) {
      for (const uint32_t id : fresh) mask_[id] = 0;
      fresh.clear();
      clear();
      scanned_ = history.size();
      for (const uint32_t id : history) take(id);
      take(anchor);
      for (const uint32_t id : picks) take(id);
    }
    // ponytail: a history with more distinct ids than the capacity keeps its first ones only.
    if (fresh.size() > capacity_) {
      for (size_t i = capacity_; i < fresh.size(); ++i) mask_[fresh[i]] = 0;
      fresh.resize(capacity_);
    }
    std::sort(fresh.begin(), fresh.end());
    const uint32_t first = static_cast<uint32_t>(ids_.size());
    ids_.insert(ids_.end(), fresh.begin(), fresh.end());
    if (ids_.empty()) return {};
    const uint32_t rows = (static_cast<uint32_t>(ids_.size()) + kTileRows - 1) / kTileRows * kTileRows;
    if (map_.size() < rows) map_.resize(rows);
    for (uint32_t i = first; i < rows; ++i) map_[i] = i < ids_.size() ? ids_[i] : ids_.front();
    return {first, fresh.empty() ? 0 : rows - first};
  }

  // Whether the head should score the shortlist: it holds at least `gate` ids.
  [[nodiscard]] bool active() const noexcept { return ids_.size() >= gate_ && !ids_.empty(); }
  // The rows the shortlist head covers: the set padded to whole tiles.
  [[nodiscard]] uint32_t rows() const noexcept {
    return static_cast<uint32_t>((ids_.size() + kTileRows - 1) / kTileRows * kTileRows);
  }
  [[nodiscard]] uint32_t size() const noexcept { return static_cast<uint32_t>(ids_.size()); }
  // Token id of each shortlist row (padding rows repeat the first).
  [[nodiscard]] std::span<const uint32_t> map() const noexcept { return {map_.data(), rows()}; }

private:
  void clear() {
    for (const uint32_t id : ids_) mask_[id] = 0;
    if (seedAll_) std::fill(mask_.begin(), mask_.end(), 0);
    ids_.clear();
    scanned_ = 0;
  }

  uint32_t vocabulary_;
  uint32_t capacity_;
  uint32_t gate_;
  bool seedAll_;
  std::vector<uint8_t> mask_;
  std::vector<uint32_t> ids_;
  std::vector<uint32_t> map_;
  std::vector<uint32_t> picks_;
  uint64_t owner_ = ~uint64_t{0};
  size_t scanned_ = 0;
};

} // namespace splash::model
