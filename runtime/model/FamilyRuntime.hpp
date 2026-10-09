#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace splash::model {

struct ModelPackage;

// The CPU-visible arena buffers a family stages its per-step data into before
// a command is encoded: rows it gathered, the index of each row's source
// among them, and fixed-size records. The packed prefill step has one set;
// each decode lane its own slice (the runtime binds the first lanes' slices
// as one packed buffer, so a lane's staged rows sit at its offset in it).
struct FamilyScratch final {
  metal::MetalBuffer prefillRows;
  metal::MetalBuffer prefillSlots;
  metal::MetalBuffer prefillBlocks;
  std::vector<metal::MetalBuffer> decodeRows;
  std::vector<metal::MetalBuffer> decodeSlots;
  std::vector<metal::MetalBuffer> decodeBlocks;
};

// Per-request host work that only some model families need (a PLE token
// history, per-sequence attention-block records, state zeroed at activation).
// Runtime owns at most one and calls it at the points below, so a family with
// none pays a null test per call. Hooks run on the thread that drives the
// runtime. A hook that cannot do its work (a failed read) throws, which fails
// the step that called it.
class FamilyRuntime {
public:
  virtual ~FamilyRuntime() = default;

  // The arena buffers the hooks below stage into, handed over once the
  // runtime's arenas exist. Empty buffers for tensors the family's model
  // does not size.
  virtual void attach(const FamilyScratch & /*scratch*/) {}

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
  // The verify step of a decode lane is final: `rowTokens` are the tokens its
  // rows read (the anchor, then the proposals, clamped into the vocabulary).
  virtual void beforeVerifyLane(uint64_t /*requestId*/, uint32_t /*lane*/, uint64_t /*position*/,
                                std::span<const uint32_t> /*rowTokens*/) {}
  // The target selected `selected` (the new anchor last) while
  // `previousAnchor` was pending, if any; the lane has read the old anchor
  // and every selection before the new one.
  virtual void afterCommit(uint64_t /*requestId*/, std::optional<uint32_t> /*previousAnchor*/,
                           std::span<const uint32_t> /*selected*/) {}
};

// The family runtime `package`'s model needs, or null for a family without.
[[nodiscard]] std::unique_ptr<FamilyRuntime> makeFamilyRuntime(const ModelPackage &package);
// The host RAM that family runtime keeps (zero for none), which the memory
// plan reports beside the Metal arenas.
[[nodiscard]] uint64_t familyHostBytes(const ModelPackage &package);

} // namespace splash::model
