#pragma once

#include "MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// An ordered dispatch list for one command buffer. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payloads_.back().data(), sizeof(Params)});
  }

  // As above, then `after` from index n + 1 and `tail` after them: a
  // kernel whose parameter struct precedes further bindings.
  template <class Params, class Tail>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers, const Params &params,
           std::vector<MetalBuffer> after, const Tail &tail, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params> && std::is_trivially_copyable_v<Tail>,
                  "dispatch parameters must be plain data");
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    ComputeDispatch &dispatch = dispatches_.back();
    uint32_t index = static_cast<uint32_t>(dispatch.buffers.size()) + 1;
    for (MetalBuffer &buffer : after) dispatch.buffers.push_back({index++, std::move(buffer)});
    payloads_.emplace_back(sizeof(Tail));
    std::memcpy(payloads_.back().data(), &tail, sizeof(Tail));
    dispatch.bytes.push_back({index, payloads_.back().data(), sizeof(Tail)});
  }

  // The command signals `value` of MetalBackend::syncEvent after the last
  // dispatch added, and waits for `value` before the next one.
  void signalAfterLast(uint64_t value) { dispatches_.back().signalValue = value; }
  void waitBeforeNext(uint64_t value) { pendingWait_ = value; }
  // The next dispatch waits for `value` of MetalBackend::gateEvent (the host's signalGate).
  void gateBeforeNext(uint64_t value) { pendingGate_ = value; }
  // Host work the submitter runs just before committing the graph, in order:
  // it starts what the graph's events wait on.
  void beforeSubmit(std::function<void()> work) { hostWork_.push_back(std::move(work)); }
  [[nodiscard]] bool hasHostWork() const noexcept { return !hostWork_.empty(); }
  void runHostWork() const {
    for (const std::function<void()> &work : hostWork_) work();
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }

private:
  ComputeDispatch &push(std::string pipeline, std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.waitValue = std::exchange(pendingWait_, 0);
    dispatch.gateValue = std::exchange(pendingGate_, 0);
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    dispatches_.push_back(std::move(dispatch));
    return dispatches_.back();
  }

  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
  uint64_t pendingWait_ = 0;
  uint64_t pendingGate_ = 0;
  std::vector<std::function<void()>> hostWork_;
};

} // namespace splash::metal
