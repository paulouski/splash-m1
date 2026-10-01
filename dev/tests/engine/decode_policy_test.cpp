#include "engine/DecodePolicy.hpp"

#include <iostream>
#include <stdexcept>

namespace {

using namespace splash::engine;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

DecodePolicyConfig testConfig() {
  DecodePolicyConfig config;
  config.enabled = true;
  config.window = 8;
  config.eligibleCycles = 8;
  config.probeCycles = 8;
  config.gate = 1.05;
  config.minEvidence = 8;
  config.backoffLoss = 0.20;
  config.maxPeriod = 64;
  return config;
}

// Feeds `count` cycles at a fixed (tokens, costMs) rate.
void feed(DecodePolicy &policy, uint32_t count, uint32_t tokens,
          double costMilliseconds) {
  for (uint32_t i = 0; i < count; ++i)
    policy.record(tokens, costMilliseconds);
}

void testDisabledAlwaysSpeculative() {
  DecodePolicyConfig config;
  config.enabled = false;
  DecodePolicy policy(config);
  for (uint32_t i = 0; i < 64; ++i) {
    require(policy.mode() == DecodeMode::Speculative,
            "disabled policy must stay speculative");
    policy.record(8, 1.0);
  }
  require(policy.metrics().specCycles == 0 && policy.metrics().arCycles == 0,
          "disabled policy must not record cycles");
}

// SPEC home runs at 2 tokens / 100ms = 20 tok/s. After 8 home cycles a probe
// of Ar runs for 8 cycles; feed it a clearly winning rate (1 token / 20ms =
// 50 tok/s, well over the 1.05x gate) and expect a latch to Ar.
void testLatchesToArOnClearWin() {
  DecodePolicy policy(testConfig());
  require(policy.mode() == DecodeMode::Speculative, "must start speculative");

  feed(policy, 8, 2, 100.0);
  require(policy.mode() == DecodeMode::Ar, "probe must start in Ar mode");
  require(policy.metrics().probesRun == 1, "one probe must have started");

  feed(policy, 8, 1, 20.0);
  require(policy.mode() == DecodeMode::Ar, "policy must latch to Ar after the win");
  require(policy.metrics().switchesToAr == 1, "exactly one switch to Ar");
  require(policy.metrics().switchesToSpec == 0, "no switch back yet");

  // Home is now Ar; it stays Ar until the next probe fires (eligibleCycles
  // more home cycles away).
  for (uint32_t i = 0; i < 7; ++i) {
    require(policy.mode() == DecodeMode::Ar, "Ar home holds between probes");
    policy.record(1, 20.0);
  }
}

// Probe loses clearly (Ar at 1 token / 100ms = 10 tok/s vs Spec's 20 tok/s,
// a >20% loss): no switch, and the next probe of Ar is delayed (period
// doubles from 8 to 16).
void testBackoffOnLosingProbe() {
  DecodePolicy policy(testConfig());
  feed(policy, 8, 2, 100.0); // enter probe of Ar
  require(policy.metrics().probesRun == 1, "probe entered");

  feed(policy, 8, 1, 100.0);
  require(policy.mode() == DecodeMode::Speculative,
          "a losing probe must not switch modes");
  require(policy.metrics().switchesToAr == 0, "no switch on a clear loss");

  uint32_t cyclesUntilNextProbe = 0;
  while (policy.metrics().probesRun == 1) {
    policy.record(2, 100.0);
    ++cyclesUntilNextProbe;
    require(cyclesUntilNextProbe <= 32, "backoff must not stall forever");
  }
  require(cyclesUntilNextProbe == 16,
          "backoff must double the probe period after a clear loss");
}

// A probe shorter than minEvidence cycles cannot decide even if it would
// clearly win: no switch, no backoff (next probe stays at the base period).
void testInconclusiveProbeDoesNotBackoff() {
  DecodePolicyConfig config = testConfig();
  config.probeCycles = 4; // below minEvidence == 8
  DecodePolicy policy(config);
  feed(policy, 8, 2, 100.0);
  require(policy.metrics().probesRun == 1, "probe entered");
  feed(policy, 4, 1, 20.0); // would clearly win at 50 tok/s if it counted
  require(policy.mode() == DecodeMode::Speculative,
          "insufficient evidence must not switch modes");
  require(policy.metrics().switchesToAr == 0, "no switch without evidence");

  uint32_t cyclesUntilNextProbe = 0;
  while (policy.metrics().probesRun == 1) {
    policy.record(2, 100.0);
    ++cyclesUntilNextProbe;
    require(cyclesUntilNextProbe <= 16, "inconclusive probe must not stall");
  }
  require(cyclesUntilNextProbe == 8,
          "inconclusive probe must retry at the base period");
}

void testReturnsToSpecAfterArRegresses() {
  DecodePolicy policy(testConfig());
  feed(policy, 8, 2, 100.0);
  feed(policy, 8, 1, 20.0); // latch to Ar (50 tok/s vs 20 tok/s home)
  require(policy.mode() == DecodeMode::Ar, "latched to Ar");
  require(policy.metrics().switchesToAr == 1, "one switch recorded");

  // Home is Ar; after eligibleCycles more Ar cycles a probe of Spec runs.
  feed(policy, 8, 1, 20.0);
  require(policy.metrics().probesRun == 2, "second probe must start");
  require(policy.mode() == DecodeMode::Speculative, "probe must be of Spec");

  // Spec now clearly beats the established Ar home rate (50 tok/s): 6
  // tokens / 100ms = 60 tok/s.
  feed(policy, 8, 6, 100.0);
  require(policy.mode() == DecodeMode::Speculative,
          "policy must return to Spec when it clearly wins");
  require(policy.metrics().switchesToSpec == 1, "one switch back to Spec");
}

} // namespace

int main() {
  try {
    testDisabledAlwaysSpeculative();
    testLatchesToArOnClearWin();
    testBackoffOnLosingProbe();
    testInconclusiveProbeDoesNotBackoff();
    testReturnsToSpecAfterArRegresses();
  } catch (const std::exception &error) {
    std::cerr << "decode_policy_test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "decode_policy_test: all tests passed\n";
  return 0;
}
