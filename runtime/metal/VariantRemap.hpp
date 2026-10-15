#pragma once

// Instrumented build only (SPLASH_VARIANT_FILE): pipeline-name remap parsed from "base variant" lines.
// A variant of "-" skips every dispatch of the base pipeline (ablation control, goes through the SPLASH_SKIP_PIPELINES path).
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace splash::metal {

struct VariantRemap {
  std::unordered_map<std::string, std::string> map;
  std::string tag = "A";  // timeline tag: "A" when empty, else "v=base>variant+..." sorted by base

  // Blank lines and '#' comments are skipped; a line without exactly two words is ignored.
  static VariantRemap parse(std::string_view text) {
    VariantRemap remap;
    std::istringstream in{std::string(text)};
    std::string line;
    while (std::getline(in, line)) {
      if (const size_t hash = line.find('#'); hash != std::string::npos) line.erase(hash);
      std::istringstream words(line);
      std::string base, variant, extra;
      if (!(words >> base >> variant) || (words >> extra)) continue;
      remap.map[base] = variant;
    }
    if (!remap.map.empty()) {
      std::map<std::string, std::string> sorted(remap.map.begin(), remap.map.end());
      remap.tag = "v=";
      for (const auto &[base, variant] : sorted) remap.tag += base + ">" + variant + "+";
      remap.tag.pop_back();
    }
    return remap;
  }

  // Pipeline to look up for a dispatch of `name` (the name itself when unmapped or skipped).
  [[nodiscard]] const std::string &apply(const std::string &name) const {
    const auto found = map.find(name);
    return found == map.end() || found->second == "-" ? name : found->second;
  }

  [[nodiscard]] bool skipped(const std::string &name) const {
    const auto found = map.find(name);
    return found != map.end() && found->second == "-";
  }

  // SPLASH_SKIP_PIPELINES syntax: exact-name entries "base$,base2$".
  [[nodiscard]] std::string skipList() const {
    std::string list;
    for (const auto &[base, variant] : std::map<std::string, std::string>(map.begin(), map.end()))
      if (variant == "-") list += (list.empty() ? "" : ",") + base + "$";
    return list;
  }
};

}  // namespace splash::metal
