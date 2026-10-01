#include "../../../runtime/metal/MetalBackend.hpp"

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace splash::metal;

void require(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Function>
void requireBackendError(Function &&function, const std::string &message) {
    try {
        function();
    } catch (const MetalBackendError &) {
        return;
    }
    require(false, message);
}

class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement) {
        method_ = class_getInstanceMethod(object_getClass(object), selector);
        require(method_ != nullptr, "Metal completion method is missing");
        original = method_setImplementation(method_, replacement);
    }
    ~MethodReplacement() { method_setImplementation(method_, original); }

    IMP original = nullptr;

private:
    Method method_ = nullptr;
};

IMP originalCompletedHandler = nullptr;
IMP originalCommandCommit = nullptr;
std::mutex observedCommandsMutex;
std::vector<id<MTLCommandBuffer>> observedCommands;
std::atomic<bool> delayNextCompletion{false};
std::atomic<unsigned> returnedCompletions{0};
std::atomic<unsigned> commandCommitCalls{0};
std::atomic<unsigned> failCommitCall{0};
std::promise<void> delayedCompletionEntered;
std::promise<void> releaseDelayedCompletion;
std::shared_future<void> delayedRelease;
std::mutex delayedCompletionThreadMutex;
std::thread delayedCompletionThread;

void observeCompletedHandler(id command, SEL selector,
                             MTLCommandBufferHandler handler) {
    {
        std::lock_guard lock(observedCommandsMutex);
        observedCommands.push_back(command);
    }
    const bool delay = delayNextCompletion.exchange(false);
    if (delay) {
        reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
            originalCompletedHandler)(command, selector,
                ^(id<MTLCommandBuffer> completed) {
                    MTLCommandBufferHandler forwardedHandler = [handler copy];
                    __strong id<MTLCommandBuffer> retainedCommand = completed;
                    std::lock_guard lock(delayedCompletionThreadMutex);
                    delayedCompletionThread = std::thread(
                        [forwardedHandler, retainedCommand] {
                            delayedCompletionEntered.set_value();
                            delayedRelease.wait();
                            forwardedHandler(retainedCommand);
                            ++returnedCompletions;
                        });
                });
        return;
    }
    reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
        originalCompletedHandler)(command, selector,
            ^(id<MTLCommandBuffer> completed) {
                handler(completed);
                ++returnedCompletions;
            });
}

void failSelectedCommandCommit(id command, SEL selector) {
    const unsigned call = ++commandCommitCalls;
    if (call == failCommitCall.load()) {
        @throw [NSException exceptionWithName:@"GroupedCommitTestFailure"
                                       reason:@"injected child commit failure"
                                     userInfo:nil];
    }
    reinterpret_cast<void (*)(id, SEL)>(originalCommandCommit)(command,
                                                               selector);
}

void resetObservedCommands() {
    std::lock_guard lock(observedCommandsMutex);
    observedCommands.clear();
}

std::vector<id<MTLCommandBuffer>> observedCommandSnapshot() {
    std::lock_guard lock(observedCommandsMutex);
    return observedCommands;
}

ComputeDispatch copyDispatch(MetalBuffer source, MetalBuffer destination,
                             const uint32_t *count) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_copy_u32";
    dispatch.buffers = {{0, std::move(source)}, {1, std::move(destination)}};
    dispatch.bytes = {{2, count, sizeof(*count)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    return dispatch;
}

ComputeDispatch addDispatch(MetalBuffer values, const uint32_t *count,
                            const uint32_t *increment) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, std::move(values)}};
    dispatch.bytes = {{1, count, sizeof(*count)},
                      {2, increment, sizeof(*increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    return dispatch;
}

void orderedChildrenAndTicketLifetime(const std::string &metallib) {
    MetalBackend backend(metallib);
    const uint64_t baselineBytes = backend.memoryStats().allocatedBytes;
    MetalBuffer source = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t sourceBytes =
        backend.memoryStats().allocatedBytes - baselineBytes;
    MetalBuffer output = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t bothBytes = backend.memoryStats().allocatedBytes;
    const uint64_t outputBytes = bothBytes - baselineBytes - sourceBytes;
    *static_cast<uint32_t *>(source.contents()) = 5;
    *static_cast<uint32_t *>(output.contents()) = 0;

    uint32_t copyCount = 1;
    uint32_t addCount = 1;
    uint32_t increment = 7;
    std::vector<ComputeDispatch> dispatches;
    dispatches.push_back(copyDispatch(source, output, &copyCount));
    dispatches.push_back(addDispatch(output, &addCount, &increment));

    id<MTLCommandBuffer> probe =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    resetObservedCommands();
    MethodReplacement completion(probe, @selector(addCompletedHandler:),
                                 reinterpret_cast<IMP>(observeCompletedHandler));
    originalCompletedHandler = completion.original;

    std::atomic<unsigned> notifications{0};
    auto ticket = backend.submitCommandAsync(
        dispatches, [&](uint64_t) { ++notifications; }, 1);
    require(observedCommandSnapshot().size() == 2,
            "one-dispatch grouping did not create two child buffers");
    dispatches.clear();
    source = {};
    copyCount = 0;
    addCount = 0;
    increment = 1000;
    require(backend.memoryStats().allocatedBytes == bothBytes,
            "the ticket did not retain graph allocations after graph destruction");

    auto blocked = addDispatch(output, &addCount, &increment);
    requireBackendError(
        [&] { (void)backend.submitCommandAsync(
                  std::span<const ComputeDispatch>(&blocked, 1)); },
        "a second sequence passed the submission gate before ticket consumption");

    (void)ticket.wait();
    require(notifications == 1,
            "grouped submission did not notify exactly once");
    require(*static_cast<uint32_t *>(output.contents()) == 12,
            "ordered child dispatches or copied byte bindings produced a wrong result");
    require(backend.memoryStats().allocatedBytes == baselineBytes + outputBytes,
            "the consumed ticket retained source allocations");

    addCount = 1;
    increment = 1;
    (void)backend.submitAsync(
        addDispatch(output, &addCount, &increment)).wait();
    require(*static_cast<uint32_t *>(output.contents()) == 13,
            "submission gate did not reopen after ticket consumption");
    std::cout << "PASS grouped child ordering, byte copies, and ticket lifetime\n";
}

void delayedCallbacksRecoverAllChildren(const std::string &metallib) {
    MetalBackend backend(metallib, 0.1);
    MetalBuffer source = backend.allocateBuffer(sizeof(uint32_t));
    MetalBuffer output = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(source.contents()) = 5;
    *static_cast<uint32_t *>(output.contents()) = 0;

    uint32_t count = 1;
    uint32_t increment = 7;
    std::vector<ComputeDispatch> dispatches{
        copyDispatch(source, output, &count),
        addDispatch(output, &count, &increment)};

    id<MTLCommandBuffer> probe =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    resetObservedCommands();
    returnedCompletions = 0;
    delayedCompletionEntered = std::promise<void>{};
    releaseDelayedCompletion = std::promise<void>{};
    auto delayedEntered = delayedCompletionEntered.get_future();
    delayedRelease = releaseDelayedCompletion.get_future().share();
    {
        std::lock_guard lock(delayedCompletionThreadMutex);
        require(!delayedCompletionThread.joinable(),
                "previous delayed completion thread was not joined");
    }
    delayNextCompletion = true;
    MethodReplacement completion(probe, @selector(addCompletedHandler:),
                                 reinterpret_cast<IMP>(observeCompletedHandler));
    originalCompletedHandler = completion.original;

    std::atomic<unsigned> notifications{0};
    auto ticket = backend.submitCommandAsync(
        dispatches, [&](uint64_t) { ++notifications; }, 1);
    require(delayedEntered.wait_for(std::chrono::seconds(5)) ==
                std::future_status::ready,
            "first grouped child did not reach its delayed completion handler");
    const auto children = observedCommandSnapshot();
    require(children.size() == 2,
            "recovery test did not create two child buffers");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    bool allTerminal = false;
    while (!allTerminal && std::chrono::steady_clock::now() < deadline) {
        allTerminal = std::all_of(children.begin(), children.end(),
            [](id<MTLCommandBuffer> child) {
                return child.status == MTLCommandBufferStatusCompleted ||
                       child.status == MTLCommandBufferStatusError;
            });
        if (!allTerminal)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(allTerminal, "grouped child buffers did not reach terminal status");
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    backend.checkHealth();
    require(ticket.ready(),
            "watchdog recovery did not scan all terminal child buffers");
    (void)ticket.wait();
    require(notifications == 1,
            "watchdog recovery did not notify exactly once");
    require(*static_cast<uint32_t *>(output.contents()) == 12,
            "watchdog recovery exposed an incomplete grouped result");

    releaseDelayedCompletion.set_value();
    std::thread deferredCompletion;
    {
        std::lock_guard lock(delayedCompletionThreadMutex);
        if (delayedCompletionThread.joinable())
            deferredCompletion = std::move(delayedCompletionThread);
    }
    if (deferredCompletion.joinable()) deferredCompletion.join();
    const auto callbackDeadline = std::chrono::steady_clock::now() +
                                 std::chrono::seconds(5);
    while (returnedCompletions.load() != children.size() &&
           std::chrono::steady_clock::now() < callbackDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(returnedCompletions == children.size() && notifications == 1,
            "late child handlers changed aggregate completion or failed to drain");
    std::cout << "PASS grouped watchdog scans every child and tolerates late handlers\n";
}

void partialCommitFailureDrainsSubmittedChildren(const std::string &metallib) {
    MetalBackend backend(metallib);
    const uint64_t baselineBytes = backend.memoryStats().allocatedBytes;
    MetalBuffer source = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t sourceBytes =
        backend.memoryStats().allocatedBytes - baselineBytes;
    MetalBuffer output = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t bothBytes = backend.memoryStats().allocatedBytes;
    const uint64_t outputBytes = bothBytes - baselineBytes - sourceBytes;
    *static_cast<uint32_t *>(source.contents()) = 5;
    *static_cast<uint32_t *>(output.contents()) = 0;

    uint32_t count = 1;
    uint32_t increment = 7;
    std::vector<ComputeDispatch> dispatches{
        copyDispatch(source, output, &count),
        addDispatch(output, &count, &increment)};

    id<MTLCommandBuffer> probe =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    MethodReplacement commit(probe, @selector(commit),
                             reinterpret_cast<IMP>(failSelectedCommandCommit));
    originalCommandCommit = commit.original;
    commandCommitCalls = 0;
    failCommitCall = 2;

    std::atomic<unsigned> notifications{0};
    auto ticket = backend.submitCommandAsync(
        dispatches, [&](uint64_t) { ++notifications; }, 1);
    dispatches.clear();
    source = {};
    require(backend.memoryStats().allocatedBytes == bothBytes,
            "partial submission released graph allocations before ticket consumption");

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (!ticket.ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(ticket.ready(),
            "partial child commit failure did not drain the committed child");
    require(notifications == 1,
            "partial child commit failure did not notify exactly once");
    require(*static_cast<uint32_t *>(output.contents()) == 5,
            "partial child commit failure released before committed work completed");
    require(backend.memoryStats().allocatedBytes == bothBytes,
            "completed but unconsumed failed ticket released its allocations");

    auto blocked = addDispatch(output, &count, &increment);
    requireBackendError(
        [&] { (void)backend.submitCommandAsync(
                  std::span<const ComputeDispatch>(&blocked, 1)); },
        "partial submission released the gate before ticket consumption");
    bool surfacedCommitFailure = false;
    try {
        (void)ticket.wait();
    } catch (const MetalBackendError &error) {
        surfacedCommitFailure =
            std::string(error.what()).find("injected child commit failure") !=
            std::string::npos;
    }
    require(surfacedCommitFailure,
            "partial child commit failure was not returned by ticket.wait");
    require(backend.memoryStats().allocatedBytes == baselineBytes + outputBytes,
            "consuming the failed ticket did not release its source allocation");
    require(!backend.healthy(),
            "partial child commit failure did not mark the backend unhealthy");
    failCommitCall = 0;
    std::cout << "PASS partial grouped commit failure drains prior child and retains ticket resources\n";
}

}  // namespace

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: metal_grouped_test <test.metallib>\n";
            return 2;
        }
        try {
            orderedChildrenAndTicketLifetime(argv[1]);
            delayedCallbacksRecoverAllChildren(argv[1]);
            partialCommitFailureDrainsSubmittedChildren(argv[1]);
        } catch (const std::exception &error) {
            std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
