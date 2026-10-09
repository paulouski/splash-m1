#include "TestChecks.hpp"
#include "model/PleStore.hpp"

#include <array>
#include <iostream>
#include <vector>

using namespace splash;
using splash::test::require;

namespace {

PleHashParams params(uint32_t ngram) {
  PleHashParams p;
  p.ngram = ngram;
  p.headsPerNgram = 2;
  p.eos = 2;
  p.multipliers[0] = 3, p.multipliers[1] = 5, p.multipliers[2] = 7, p.multipliers[3] = 11;
  p.vocabularies = {1000, 2000, 3000, 4000, 5000, 6000};
  p.offsets = {0, 1000, 3000, 6000, 10000, 15000};
  p.vocabularies.resize(p.heads());
  p.offsets.resize(p.heads());
  return p;
}

// The qwen4_ple_hash kernel's arithmetic, lane by lane (before = the two tokens before a lane's first row).
std::vector<uint32_t> kernelRows(const PleHashParams &p, std::span<const uint32_t> tokens, uint32_t laneRows,
                                 std::span<const uint32_t> before) {
  std::vector<uint32_t> out(tokens.size() * p.heads());
  for (uint32_t index = 0; index < tokens.size(); ++index) {
    const uint32_t lane = index / laneRows, r = index % laneRows;
    uint64_t context[4] = {tokens[index], p.eos, p.eos, p.eos};
    bool cut = false;
    for (uint32_t back = 1; back < p.ngram; ++back) {
      const uint32_t previous = back <= r ? tokens[index - back] : before[lane * 2 + (back - r - 1)];
      cut = cut || previous == 0xffffffffu || previous == p.eos;
      context[back] = cut ? p.eos : previous;
    }
    for (uint32_t gram = 2; gram <= p.ngram; ++gram) {
      uint64_t mixed = context[0] * p.multipliers[0];
      for (uint32_t j = 1; j < gram; ++j) mixed ^= context[j] * p.multipliers[j];
      for (uint32_t g = 0; g < p.headsPerNgram; ++g) {
        const uint32_t head = (gram - 2) * p.headsPerNgram + g;
        out[uint64_t{index} * p.heads() + head] = uint32_t(mixed % p.vocabularies[head] + p.offsets[head]);
      }
    }
  }
  return out;
}

std::vector<uint32_t> hasherRows(const PleHasher &hasher, std::span<const uint32_t> sequence) {
  std::vector<uint32_t> out(sequence.size() * hasher.heads());
  for (uint64_t position = 0; position < sequence.size(); ++position)
    hasher.rows(position, sequence[position],
                [&](uint64_t at) -> std::optional<uint32_t> { return sequence[at]; },
                out.data() + position * hasher.heads());
  return out;
}

} // namespace

int main() {
  try {
    // Golden values computed independently (Python) for 3-gram, EOS 2, sequence 10 20 30 EOS 40 50.
    const std::array<uint32_t, 6> sequence{10, 20, 30, 2, 40, 50};
    const std::array<std::array<uint32_t, 4>, 6> golden{{{20, 1020, 3026, 6026},
                                                          {14, 1014, 3000, 6000},
                                                          {62, 1062, 3120, 6120},
                                                          {144, 1144, 3028, 6028},
                                                          {114, 1114, 3124, 6124},
                                                          {94, 1094, 3080, 6080}}};
    const PleHasher three(params(3));
    const std::vector<uint32_t> got = hasherRows(three, sequence);
    for (size_t row = 0; row < sequence.size(); ++row)
      for (size_t head = 0; head < 4; ++head)
        require(got[row * 4 + head] == golden[row][head], "3-gram golden row mismatch");

    // Row 0 starts the sequence and row 4 follows an EOS: both mix only EOS behind the token.
    // A 2-gram hasher produces exactly the first 2-gram heads of the 3-gram one.
    const PleHasher two(params(2));
    require(two.heads() == 2, "2-gram head count");
    const std::vector<uint32_t> got2 = hasherRows(two, sequence);
    for (size_t row = 0; row < sequence.size(); ++row)
      for (size_t head = 0; head < 2; ++head)
        require(got2[row * 2 + head] == got[row * 4 + head], "2-gram heads differ from the 3-gram's");
    // The 3-gram heads depend on the second predecessor; the 2-gram heads do not.
    const std::array<uint32_t, 3> a{7, 8, 9}, b{6, 8, 9};
    const std::vector<uint32_t> ha = hasherRows(three, a), hb = hasherRows(three, b);
    require(ha[2 * 4 + 0] == hb[2 * 4 + 0] && ha[2 * 4 + 1] == hb[2 * 4 + 1], "2-gram heads see the third token back");
    require(ha[2 * 4 + 2] != hb[2 * 4 + 2], "3-gram heads ignore the third token back");

    // Every row is inside its head's range.
    const PleHashParams p = params(3);
    for (size_t row = 0; row < sequence.size(); ++row)
      for (size_t head = 0; head < 4; ++head)
        require(got[row * 4 + head] >= p.offsets[head] && got[row * 4 + head] < p.offsets[head] + p.vocabularies[head],
                "row outside its head");

    // Matches the GPU kernel arithmetic: one lane of 6 rows, none before it, then two lanes with history.
    require(kernelRows(p, sequence, 6, std::array<uint32_t, 2>{0xffffffffu, 0xffffffffu}) == got,
            "kernel reference differs (sequence start)");
    const std::array<uint32_t, 6> stream{5, 6, 7, 8, 9, 12};
    const std::vector<uint32_t> whole = hasherRows(three, stream);
    // lane 0 = rows 0..2 from the start; lane 1 = rows 3..5 with before = {stream[2], stream[1]}.
    const std::array<uint32_t, 4> before{0xffffffffu, 0xffffffffu, 7, 6};
    require(kernelRows(p, stream, 3, before) == whole, "kernel reference differs (lanes)");
    // An EOS in the history cuts the window.
    const std::array<uint32_t, 4> cutBefore{0xffffffffu, 0xffffffffu, 2, 6};
    const std::array<uint32_t, 3> tail{8, 9, 12};
    const std::array<uint32_t, 5> withEos{6, 2, 8, 9, 12};
    const std::vector<uint32_t> expected = hasherRows(three, withEos);
    const std::vector<uint32_t> lane = kernelRows(p, tail, 3, std::span<const uint32_t>(cutBefore).subspan(2));
    for (size_t i = 0; i < 3 * 4; ++i) require(lane[i] == expected[2 * 4 + i], "EOS history does not cut");

    // A start position beyond the history behaves like the sequence start.
    uint32_t out[4];
    three.rows(0, 10, [](uint64_t) -> std::optional<uint32_t> { return 99; }, out);
    for (size_t head = 0; head < 4; ++head) require(out[head] == golden[0][head], "position 0 sees a predecessor");

    std::cout << "ple hash tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
