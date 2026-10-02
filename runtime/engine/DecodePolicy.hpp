#pragma once

// Adaptive AR<->speculative decode mode policy ("M1 ladder"), one instance
// per decoding request/lane. Pure state machine: no Metal/model dependency,
// so it is exercised by a CPU-only unit test independent of the engine.
//
// Splash's decode graph has exactly two useful modes, unlike the oMLX
// reference (which also chose between verify block sizes 8/4): every verify
// cycle dispatches the fixed SPLASH_TARGET_VERIFY_ROWS=8 kernel geometry
// (metal/abi/ExecutionGeometry.h; static_assert'd throughout runtime/ops), so
// a smaller speculative block is not a cheap option here (see
// _local/IMPROVEMENTS_FROM_QWEN38_REFORGE.md #1 and the task report). The
// only real choice is:
//   Speculative -- draft proposes, target verifies all 8 rows (existing path)
//   Ar          -- draft skipped; target still runs the 8-row verify kernel
//                  for one committed anchor, but the acceptance kernel is
//                  capped at maximumRetained=1 so only row 0 (causally
//                  independent of the unused rows 1..7) is ever committed.
// This makes Ar bit-exact with a "real" single-token target step: row 0's
// computation does not depend on the draft at all.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <string_view>

namespace splash::engine {

enum class DecodeMode : uint8_t { Speculative, Ar };

struct DecodePolicyConfig final {
  bool enabled = false;
  // Rolling window (in eligible cycles of the same mode) used for the
  // tokens/second estimate of each mode.
  uint32_t window = 8;
  // Home cycles between probes of the other mode.
  uint32_t eligibleCycles = 8;
  // Length of a probe, in cycles.
  uint32_t probeCycles = 8;
  // A probe wins (switches modes) only if probe_tps >= home_tps * gate.
  double gate = 1.05;
  // A probe is inconclusive (no switch, no backoff) below this much
  // accumulated evidence in the probed mode's window.
  uint32_t minEvidence = 8;
  // A conclusive probe that loses by more than this fraction doubles the
  // period before the next probe of that same mode (capped at maxPeriod).
  double backoffLoss = 0.20;
  uint32_t maxPeriod = 64;

  [[nodiscard]] bool valid() const noexcept {
    return window > 0 && eligibleCycles > 0 && probeCycles > 0 && gate > 0.0 &&
           minEvidence > 0 && backoffLoss > 0.0 && backoffLoss < 1.0 &&
           maxPeriod >= eligibleCycles;
  }
};

// Per-request adaptive decode policy. record() must be called exactly once
// per completed decode cycle for this request, in dispatch order, with the
// tokens committed and the cycle's wall-clock cost. mode() reports which
// mode the *next* cycle should use.
class DecodePolicy final {
public:
  struct Metrics final {
    uint64_t specCycles = 0;
    uint64_t arCycles = 0;
    uint64_t switchesToAr = 0;
    uint64_t switchesToSpec = 0;
    uint64_t probesRun = 0;
  };

  DecodePolicy() = default;
  explicit DecodePolicy(DecodePolicyConfig config) : config_(config) {
    curPeriod_ = config_.eligibleCycles;
  }

  [[nodiscard]] DecodeMode mode() const noexcept {
    if (!config_.enabled)
      return DecodeMode::Speculative;
    return probing_ ? probeMode_ : homeMode_;
  }

  [[nodiscard]] const Metrics &metrics() const noexcept { return metrics_; }

  void record(uint32_t committedTokens, double costMilliseconds) {
    if (!config_.enabled || !config_.valid())
      return;
    ++cycleIndex_;
    const DecodeMode effective = probing_ ? probeMode_ : homeMode_;
    if (effective == DecodeMode::Speculative)
      ++metrics_.specCycles;
    else
      ++metrics_.arCycles;

    if (cycleIndex_ == 1) {
      // The very first recorded cycle of a request includes draft-cache
      // prefill overhead and is never representative; exclude it from both
      // windows (matches the oMLX reference's `ladder_cycle_idx != 1`).
    } else {
      window(effective).push_back({committedTokens, costMilliseconds});
      while (window(effective).size() > config_.window)
        window(effective).pop_front();
    }

    if (probing_) {
      if (--probeRemaining_ > 0)
        return;
      finishProbe();
      return;
    }

    ++homeCycles_;
    if (homeCycles_ >= curPeriod_)
      startProbe();
  }

private:
  struct Sample final {
    uint32_t tokens = 0;
    double costMilliseconds = 0.0;
  };

  [[nodiscard]] std::deque<Sample> &window(DecodeMode mode) noexcept {
    return mode == DecodeMode::Speculative ? specWindow_ : arWindow_;
  }

  [[nodiscard]] static DecodeMode other(DecodeMode mode) noexcept {
    return mode == DecodeMode::Speculative ? DecodeMode::Ar
                                            : DecodeMode::Speculative;
  }

  // nullopt (0.0 with zero evidence) when the window holds no cost yet.
  [[nodiscard]] double tokensPerSecond(DecodeMode mode) const noexcept {
    const std::deque<Sample> &samples =
        mode == DecodeMode::Speculative ? specWindow_ : arWindow_;
    uint64_t tokens = 0;
    double costMilliseconds = 0.0;
    for (const Sample &sample : samples) {
      tokens += sample.tokens;
      costMilliseconds += sample.costMilliseconds;
    }
    if (costMilliseconds <= 0.0)
      return 0.0;
    return static_cast<double>(tokens) / (costMilliseconds / 1000.0);
  }

  [[nodiscard]] size_t evidence(DecodeMode mode) const noexcept {
    return mode == DecodeMode::Speculative ? specWindow_.size()
                                           : arWindow_.size();
  }

  void startProbe() {
    probing_ = true;
    probeMode_ = other(homeMode_);
    probeRemaining_ = config_.probeCycles;
    ++metrics_.probesRun;
  }

  void finishProbe() {
    const double probeTps = tokensPerSecond(probeMode_);
    const double homeTps = tokensPerSecond(homeMode_);
    const bool enoughEvidence = evidence(probeMode_) >= config_.minEvidence;
    const bool switched =
        enoughEvidence && probeTps >= homeTps * config_.gate;
    if (switched) {
      if (probeMode_ == DecodeMode::Ar)
        ++metrics_.switchesToAr;
      else
        ++metrics_.switchesToSpec;
      homeMode_ = probeMode_;
      curPeriod_ = config_.eligibleCycles;
    } else if (!enoughEvidence) {
      curPeriod_ = config_.eligibleCycles;
    } else if (probeTps < homeTps * (1.0 - config_.backoffLoss)) {
      curPeriod_ = std::min(curPeriod_ * 2, config_.maxPeriod);
    } else {
      curPeriod_ = config_.eligibleCycles;
    }
    probing_ = false;
    probeRemaining_ = 0;
    homeCycles_ = 0;
  }

  DecodePolicyConfig config_{};
  DecodeMode homeMode_ = DecodeMode::Speculative;
  DecodeMode probeMode_ = DecodeMode::Speculative;
  bool probing_ = false;
  uint32_t probeRemaining_ = 0;
  uint32_t homeCycles_ = 0;
  uint32_t curPeriod_ = 8;
  uint64_t cycleIndex_ = 0;
  std::deque<Sample> specWindow_;
  std::deque<Sample> arWindow_;
  Metrics metrics_{};
};

namespace decode_policy_detail {
inline bool envFlag(const char *name, bool defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  return std::string_view(raw) == "1";
}
inline uint32_t envUint(const char *name, uint32_t defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  const long value = std::strtol(raw, nullptr, 10);
  return value > 0 ? static_cast<uint32_t>(value) : defaultValue;
}
inline double envDouble(const char *name, double defaultValue) {
  const char *raw = std::getenv(name);
  if (!raw || !*raw)
    return defaultValue;
  char *end = nullptr;
  const double value = std::strtod(raw, &end);
  return end != raw ? value : defaultValue;
}
} // namespace decode_policy_detail

// SPLASH_DECODE_M1=1: single-row decode of prompt-lookup misses through the PQ2_0 GEMV (default off). Read once per
// model runtime construction.
inline bool decodeSingleRowFromEnvironment() {
  return decode_policy_detail::envFlag("SPLASH_DECODE_M1", false);
}

// Reads SPLASH_DECODE_LADDER (default off) and its tuning overrides. Called
// once per model runtime construction; the CLI --decode-ladder flag sets the
// same environment variable before the model is built (runtime/main.mm).
inline DecodePolicyConfig decodePolicyConfigFromEnvironment() {
  using namespace decode_policy_detail;
  DecodePolicyConfig config;
  config.enabled = envFlag("SPLASH_DECODE_LADDER", false);
  config.window = envUint("SPLASH_DECODE_LADDER_WINDOW", config.window);
  config.eligibleCycles =
      envUint("SPLASH_DECODE_LADDER_ELIGIBLE", config.eligibleCycles);
  config.probeCycles =
      envUint("SPLASH_DECODE_LADDER_PROBE", config.probeCycles);
  config.gate = envDouble("SPLASH_DECODE_LADDER_GATE", config.gate);
  config.minEvidence =
      envUint("SPLASH_DECODE_LADDER_MIN_EVIDENCE", config.minEvidence);
  config.backoffLoss =
      envDouble("SPLASH_DECODE_LADDER_BACKOFF_LOSS", config.backoffLoss);
  config.maxPeriod =
      envUint("SPLASH_DECODE_LADDER_MAX_PERIOD", config.maxPeriod);
  return config;
}

} // namespace splash::engine
