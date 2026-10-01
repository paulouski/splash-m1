#include "ops/Logprobs.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  using splash::ops::tokenLogprobs;
  const float row[6] = {1.0F, 3.0F, 3.0F, -2.0F, 0.5F, 100.0F};
  const auto plain = tokenLogprobs(row, 5, 4, 3);
  double sum = 0.0;
  for (uint32_t token = 0; token < 5; ++token)
    sum += std::exp(static_cast<double>(row[token]));
  require(std::fabs(plain.logprob - (0.5 - std::log(sum))) < 1e-6,
          "chosen logprob");
  require(plain.topIds.size() == 3 && plain.topIds[0] == 1 &&
              plain.topIds[1] == 2 && plain.topIds[2] == 0,
          "top-k order with tie to lower id");
  require(std::fabs(plain.topLogprobs[0] - (3.0 - std::log(sum))) < 1e-6,
          "top logprob");
  double probability = 0.0;
  const auto all = tokenLogprobs(row, 5, 0, 5);
  for (float value : all.topLogprobs)
    probability += std::exp(static_cast<double>(value));
  require(std::fabs(probability - 1.0) < 1e-6, "full top-k sums to one");
  require(tokenLogprobs(row, 5, 1, 0).topIds.empty(), "k=0 has no top list");
  // A huge logit must not overflow the normalizer.
  const auto large = tokenLogprobs(row, 6, 5, 2);
  require(std::fabs(large.logprob) < 1e-6 && large.topIds[0] == 5,
          "stable with large logit");
  std::cout << "logprobs_test ok\n";
  return 0;
}
