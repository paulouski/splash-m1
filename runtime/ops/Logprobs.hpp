#pragma once

// Full-vocabulary log-softmax summary of one fp32 logits row: the chosen
// token's logprob and the top-k tokens with their logprobs. CPU-only; the
// runtime calls it solely for requests that ask for logprobs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace splash::ops {

inline constexpr uint32_t kMaximumTopLogprobs = 20;

struct TokenLogprobs final {
  float logprob = 0.0F;
  std::vector<uint32_t> topIds;
  std::vector<float> topLogprobs;

  bool operator==(const TokenLogprobs &) const = default;
};

inline TokenLogprobs tokenLogprobs(const float *row, uint32_t vocabulary,
                                   uint32_t chosen, uint32_t topK) {
  float maximum = row[0];
  for (uint32_t token = 1; token < vocabulary; ++token)
    maximum = std::max(maximum, row[token]);
  double sum = 0.0;
  for (uint32_t token = 0; token < vocabulary; ++token)
    sum += std::exp(static_cast<double>(row[token]) - maximum);
  const double logSumExp = static_cast<double>(maximum) + std::log(sum);

  // Descending by logit; lower token id wins ties, matching argmax.
  TokenLogprobs result;
  result.logprob = static_cast<float>(row[chosen] - logSumExp);
  std::vector<uint32_t> top;
  top.reserve(topK + 1);
  for (uint32_t token = 0; topK != 0 && token < vocabulary; ++token) {
    if (top.size() == topK && row[token] <= row[top.back()])
      continue;
    auto at = top.begin();
    while (at != top.end() && row[*at] >= row[token])
      ++at;
    top.insert(at, token);
    if (top.size() > topK)
      top.pop_back();
  }
  for (uint32_t token : top) {
    result.topIds.push_back(token);
    result.topLogprobs.push_back(static_cast<float>(row[token] - logSumExp));
  }
  return result;
}

} // namespace splash::ops
