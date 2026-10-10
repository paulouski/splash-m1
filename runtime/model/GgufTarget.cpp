#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <cstdlib>
#include <regex>
#include <string>

namespace splash::model {
std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error))
    if (entry.path().extension() == ".gguf") found.push_back(entry.path());
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  if (found.size() == 1) return found;
  // A split GGUF's parts end in -0000K-of-0000N; any other second file is an error.
  std::sort(found.begin(), found.end());
  const std::regex part(R"(.*-(\d{5})-of-(\d{5})\.gguf)");
  for (size_t index = 0; index < found.size(); ++index) {
    std::smatch match;
    const std::string name = found[index].filename().string();
    if (!std::regex_match(name, match, part) || std::stoul(match[1]) != index + 1 ||
        std::stoul(match[2]) != found.size())
      throw GgufError("target directory holds more than one GGUF, and not the parts of one split GGUF: " +
                      directory.string());
  }
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, const std::vector<std::filesystem::path> &paths,
                                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend),
      images_(PreparedFiles([&backend] { backend.checkOperation(); }, std::move(admitConversion),
                            [this] { checkUnchanged(); }),
              [this](const gguf::Image &image) {
                return WeightWriter([this, &image](int destination, const PreparationCheck &admit) {
                  writeGgufImage(backend_, sources_, destination, image, admit);
                });
              }) {
  std::vector<WeightSource *> parsed;
  for (const std::filesystem::path &path : paths) {
    files_.emplace_back(path, [&backend] { backend.checkOperation(); });
    parsed.push_back(&files_.back());
    sources_.push_back(&files_.back());
  }
  const GgufFile file(parsed);
  checkUnchanged();
  // Validates the whole source before its tensor data is hashed.
  plan(file, geometry, {});
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, const PrismMlxDirectory &directory,
                                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend),
      images_(PreparedFiles([&backend] { backend.checkOperation(); }, std::move(admitConversion),
                            [this] { checkUnchanged(); }),
              [this](const gguf::Image &image) {
                return WeightWriter([this, &image](int destination, const PreparationCheck &admit) {
                  writeGgufImage(backend_, sources_, destination, image, admit);
                });
              }) {
  WeightSource &source = files_.emplace_back(prismMlxFile(directory.path), [&backend] { backend.checkOperation(); });
  sources_.push_back(&source);
  PrismMlxView view = bindPrismMlx(source, directory.path, geometry);
  const GgufFile file(source, std::move(view.header));
  checkUnchanged();
  plan(file, geometry, view.identity);
}

void GgufTargetLoader::checkUnchanged() const {
  for (const WeightSource *source : sources_) source->checkUnchanged();
}

void GgufTargetLoader::plan(const GgufFile &file, const gguf::TargetGeometry &geometry,
                            std::string_view prismIdentity) {
  rotation_ = file.rotation();
  // SPLASH_PLE_TABLE: a GGUF whose per_layer_token_embd.weight replaces the model's.
  if (const char *external = std::getenv("SPLASH_PLE_TABLE"); external && *external) {
    WeightSource source(external, [this] { backend_.checkOperation(); });
    const GgufFile table(source);
    ple_ = gguf::readPleTable(file, geometry, &table);
  } else {
    ple_ = gguf::readPleTable(file, geometry);
  }
  for (gguf::Image &image : gguf::planImages(file, geometry)) {
    backend_.checkOperation();
    PreparedWeight weight = ggufImageWeight(sources_, image);
    if (!prismIdentity.empty()) weight.key = prismMlxKey(weight.key, prismIdentity);
    images_.add(std::move(image), std::move(weight));
  }
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= images_.size() - 2) throw GgufError("target layer is out of range");
  return images_.open(backend_, index);
}

WeightFile GgufTargetLoader::head() { return images_.open(backend_, images_.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return images_.open(backend_, images_.size() - 1); }

void GgufTargetLoader::prepare() { images_.prepare(); }

GgufMtpLoader::GgufMtpLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                             const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend), source_(path, [&backend] { backend.checkOperation(); }),
      images_(PreparedFiles([&backend] { backend.checkOperation(); }, std::move(admitConversion),
                            [this] { source_.checkUnchanged(); }),
              [this](const gguf::Image &image) {
                return WeightWriter([this, &image](int destination, const PreparationCheck &admit) {
                  writeGgufImage(backend_, source_, destination, image, admit);
                });
              }) {
  const GgufFile file(source_);
  source_.checkUnchanged();
  gguf::Image image = gguf::planMtpImage(file, geometry);
  PreparedWeight weight = ggufImageWeight(source_, image);
  weight.component = "mtp/" + image.name;
  images_.add(std::move(image), std::move(weight));
}

void GgufMtpLoader::prepare() { images_.prepare(); }

WeightFile GgufMtpLoader::open() { return images_.open(backend_, 0); }

} // namespace splash::model
