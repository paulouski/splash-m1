#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace splash::model {

// Per-request host work that only some model families need (a PLE token
// history, per-sequence attention-block records, state zeroed at activation).
// Runtime owns at most one and calls it at the points below, so a family with
// none pays a null test per call. Hooks run on the thread that drives the
// runtime and must not throw.
class FamilyRuntime {
public:
  virtual ~FamilyRuntime() = default;

  // A request's state lane was granted (begin, resume); `prompt` is its full
  // prompt, including a replayed generation.
  virtual void onActivate(uint64_t /*requestId*/, uint32_t /*stateLane*/,
                          std::span<const uint32_t> /*prompt*/) {}
  // The request leaves the runtime (end).
  virtual void onEnd(uint64_t /*requestId*/) {}
  // A packed-prefill sequence of `inputTokens` from `position`, whose rows
  // start at `rowBegin` of the batch.
  virtual void beforePrefillSequence(uint64_t /*requestId*/, uint64_t /*position*/,
                                     std::span<const uint32_t> /*inputTokens*/,
                                     uint32_t /*rowBegin*/) {}
  // A decode lane of `stateLane` at decode `lane`; `anchor` is the pending
  // token that starts the step at `position`.
  virtual void beforeDecodeLane(uint64_t /*requestId*/, uint32_t /*lane*/,
                                uint32_t /*stateLane*/, uint64_t /*position*/,
                                uint32_t /*anchor*/) {}
  // The target selected `selected` (the new anchor last) while
  // `previousAnchor` was pending, if any; the lane has read the old anchor
  // and every selection before the new one.
  virtual void afterCommit(uint64_t /*requestId*/, std::optional<uint32_t> /*previousAnchor*/,
                           std::span<const uint32_t> /*selected*/) {}
};

} // namespace splash::model
