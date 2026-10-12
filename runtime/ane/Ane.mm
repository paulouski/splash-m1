#include "ane/Ane.hpp"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// The private interface, as ds4 d50435b declares it.
@protocol SplashAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol SplashAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos
                    error:(NSError **)error;
@end
@protocol SplashAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol SplashAneRequest
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs
          outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber *)procedure sharedEvents:(id)events transactionHandle:(NSNumber *)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end
@protocol SplashAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
@end

namespace splash::ane {
namespace {

// How long an evaluation may run once the GPU signalled its input ready: one takes 2-60 ms, and Metal fails a
// command buffer whose wait stays unmet for about 5 s.
constexpr int64_t kTimeoutNanoseconds = 2 * int64_t{NSEC_PER_SEC};
constexpr unsigned kQos = 21;

struct Surface {
  IOSurfaceRef surface = nullptr;
  id object = nil;
  metal::MetalBuffer buffer;
  uint32_t stride = 0;
};

NSString *dims(uint64_t rows, uint64_t width) {
  return [NSString stringWithFormat:@"[1, 1, %llu, %llu]", (unsigned long long)rows, (unsigned long long)width];
}

NSString *bufferType(const Surface &s, uint32_t elementBytes, uint32_t rows, uint32_t width) {
  const uint64_t stride = s.stride / elementBytes, plane = uint64_t{rows} * stride;
  return [NSString stringWithFormat:@"tensor_buffer<%s, shape=%@, strides=[%llu, %llu, %llu, 1], interleave_factors=[1, 1, 1, 1]>",
                                    elementBytes == 1 ? "int8" : "fp16", dims(rows, width), (unsigned long long)plane,
                                    (unsigned long long)plane, (unsigned long long)stride];
}

// One evaluation in flight: its event values and whether it has finished (by its completion handler, or failed by
// the watchdog first).
struct Pending {
  uint64_t ready = 0, done = 0;
  std::atomic<bool> finished{false};
};

NSString *cacheRoot() {
  const char *weights = std::getenv("SPLASH_WEIGHT_CACHE");
  NSString *base = weights && weights[0] ? @(weights)
      : [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject
            stringByAppendingPathComponent:@"splash"];
  return [base stringByAppendingPathComponent:@"ane"];
}

// Writes path only when it does not already hold these bytes: the service keys its compilation by the directory, and
// a rewrite can strand it.
void writeIfChanged(NSString *path, NSData *data) {
  NSData *old = [NSData dataWithContentsOfFile:path];
  if (!old || ![old isEqualToData:data]) [data writeToFile:path atomically:YES];
}

} // namespace

bool requested() noexcept {
  const char *value = std::getenv("SPLASH_ANE");
  return value && value[0] && std::strcmp(value, "0") != 0;
}

struct Program::Impl {
  id model = nil;
  uint32_t inputSize = 0, width = 0, maximumRows = 0;
  std::vector<uint32_t> bucketRows;
  std::vector<NSNumber *> procedureId;
  std::vector<NSArray *> inputIndices, outputIndices, inputNames;
  Surface x, y, w[kSlots];
  metal::MetalBuffer scales[kSlots];
  std::shared_ptr<void> keep[2 + kSlots];
  uint32_t slots = 0;
};

Program::Program(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() = default;
uint32_t Program::inputSize() const noexcept { return impl_->inputSize; }
uint32_t Program::width() const noexcept { return impl_->width; }
uint32_t Program::maximumRows() const noexcept { return impl_->maximumRows; }
bool Program::runs(uint32_t rows) const noexcept { return rows >= kMinimumRows && rows <= impl_->maximumRows; }
uint32_t Program::weightStride() const noexcept { return impl_->w[0].stride; }
uint32_t Program::inputStride() const noexcept { return impl_->x.stride / 2; }
uint32_t Program::outputStride() const noexcept { return impl_->y.stride / 2; }
const metal::MetalBuffer &Program::input() const noexcept { return impl_->x.buffer; }
const metal::MetalBuffer &Program::output() const noexcept { return impl_->y.buffer; }
const metal::MetalBuffer &Program::weights(uint32_t slot) const noexcept { return impl_->w[slot].buffer; }
uint32_t Program::nextSlot() noexcept { return impl_->slots++ % kSlots; }
const metal::MetalBuffer &Program::scales(uint32_t slot) const noexcept { return impl_->scales[slot]; }

struct Runtime::Impl {
  explicit Impl(metal::MetalBackend &b) : backend(b) {}

  metal::MetalBackend &backend;
  id<MTLSharedEvent> event = nil;
  id<SplashAneClient> client = nil;
  Class<SplashAneModel> modelClass;
  Class<SplashAneSurface> surfaceClass;
  Class<SplashAneRequest> requestClass;
  Class<SplashAneEvents> eventsClass, signalClass, waitClass;
  MTLSharedEventListener *listener = nil;
  dispatch_queue_t queue = nil;
  metal::MetalBuffer status;
  std::vector<std::shared_ptr<Program>> programs;
  uint64_t sequence = 0;
  std::atomic<bool> failed{false};
  std::atomic<uint64_t> launched{0}, completed{0};
  uint64_t evaluations = 0;
  uint64_t faultAt = 0;
  int faultKind = 0;
  bool logging = false;

  bool open() {
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
      std::fprintf(stderr, "splash: ANE: AppleNeuralEngine does not load\n");
      return false;
    }
    client = [(Class<SplashAneClient>)NSClassFromString(@"_ANEClient") sharedConnection];
    modelClass = NSClassFromString(@"_ANEModel");
    surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    requestClass = NSClassFromString(@"_ANERequest");
    eventsClass = NSClassFromString(@"_ANESharedEvents");
    signalClass = NSClassFromString(@"_ANESharedSignalEvent");
    waitClass = NSClassFromString(@"_ANESharedWaitEvent");
    if (!client || !modelClass || !surfaceClass || !requestClass || !eventsClass || !signalClass || !waitClass) {
      std::fprintf(stderr, "splash: ANE: private interface missing\n");
      return false;
    }
    event = (__bridge id<MTLSharedEvent>)backend.syncEvent();
    queue = dispatch_queue_create("splash.ane", DISPATCH_QUEUE_SERIAL);
    listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:queue];
    status = backend.allocateBuffer(sizeof(uint32_t), metal::BufferStorage::Shared, "ane status");
    *static_cast<uint32_t *>(status.contents()) = 0;
    logging = std::getenv("SPLASH_ANE_LOG") != nullptr;
    // SPLASH_ANE_TEST_FAIL=N[:error|:nan|:hang] replaces the Nth evaluation by a failure of that kind.
    if (const char *fault = std::getenv("SPLASH_ANE_TEST_FAIL"); fault && fault[0]) {
      faultAt = std::strtoull(fault, nullptr, 10);
      const char *kind = std::strchr(fault, ':');
      faultKind = !kind || !std::strcmp(kind, ":error") ? 1 : !std::strcmp(kind, ":nan") ? 2 : 3;
    }
    return true;
  }

  bool makeSurface(Surface &s, std::shared_ptr<void> &keep, uint32_t rows, uint32_t width, uint32_t elementBytes) {
    const uint32_t stride = (width * elementBytes + 63u) / 64u * 64u;
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    const uint64_t size = (uint64_t{stride} * rows + page - 1) / page * page;
    NSDictionary *properties = @{
      (id)kIOSurfaceWidth : @(width), (id)kIOSurfaceHeight : @(rows), (id)kIOSurfaceBytesPerElement : @(elementBytes),
      (id)kIOSurfaceBytesPerRow : @(stride), (id)kIOSurfaceAllocSize : @(size),
      (id)kIOSurfacePixelFormat : @(elementBytes == 1 ? 0x4c303038 : 0x4c303068),
    };
    s.surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!s.surface) return false;
    keep = std::shared_ptr<void>(const_cast<void *>(static_cast<const void *>(s.surface)), [](void *p) { CFRelease(p); });
    if (IOSurfaceGetBytesPerRow(s.surface) != stride) return false;
    s.stride = stride;
    s.object = [surfaceClass objectWithIOSurface:s.surface];
    s.buffer = backend.wrapSharedMemory(IOSurfaceGetBaseAddress(s.surface), size, keep, "ane surface");
    return s.object != nil;
  }

  bool load(Program::Impl &p, NSString *mil, NSString *tag) {
    // The service caches its compilation by this directory: keep it.
    NSString *dir = [cacheRoot() stringByAppendingPathComponent:tag];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    uint8_t blob[128] = {0};
    blob[0] = 1;
    blob[4] = 2;
    writeIfChanged([dir stringByAppendingPathComponent:@"model.mil"], [mil dataUsingEncoding:NSUTF8StringEncoding]);
    writeIfChanged([dir stringByAppendingPathComponent:@"weights.bin"], [NSData dataWithBytes:blob length:sizeof blob]);
    p.model = [modelClass modelAtURL:[NSURL fileURLWithPath:dir isDirectory:YES]
                                 key:[@"splash-ane-" stringByAppendingString:tag]];
    NSDictionary *options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
    NSError *error = nil;
    if (![client compiledModelExistsFor:p.model] && ![client compileModel:p.model options:options qos:kQos error:&error]) {
      std::fprintf(stderr, "splash: ANE: compile failed: %s\n", error.description.UTF8String);
      return false;
    }
    if (![client loadModel:p.model options:@{} qos:kQos error:&error]) {
      error = nil;
      [client purgeCompiledModel:p.model];
      if (![client compileModel:p.model options:options qos:kQos error:&error] ||
          ![client loadModel:p.model options:@{} qos:kQos error:&error]) {
        std::fprintf(stderr, "splash: ANE: load failed: %s\n", error.description.UTF8String);
        return false;
      }
    }
    NSDictionary *description = [p.model modelAttributes][@"ANEFModelDescription"];
    NSDictionary *ids = description[@"kANEFModelProcedureNameToIDMapKey"];
    NSArray *procedures = description[@"ANEFModelProcedures"];
    NSArray *symbols = description[@"kANEFModelInputSymbolsArrayKey"];
    for (uint32_t rows : p.bucketRows) {
      NSNumber *procedure = ids[[NSString stringWithFormat:@"r%u", rows]];
      NSDictionary *entry = nil;
      for (NSDictionary *e in procedures)
        if (procedure && [e[@"ANEFModelProcedureID"] isEqual:procedure]) entry = e;
      if (!entry) return false;
      NSMutableArray *names = [NSMutableArray array];
      for (NSNumber *index in entry[@"ANEFModelInputSymbolIndexArray"]) [names addObject:symbols[index.unsignedIntValue]];
      if ([entry[@"ANEFModelOutputSymbolIndexArray"] count] != 1) return false;
      p.procedureId.push_back(procedure);
      p.inputIndices.push_back(entry[@"ANEFModelInputSymbolIndexArray"]);
      p.outputIndices.push_back(entry[@"ANEFModelOutputSymbolIndexArray"]);
      p.inputNames.push_back(names);
    }
    return true;
  }

  std::shared_ptr<Program> create(uint32_t K, uint32_t N, uint32_t R) {
    @autoreleasepool {
      auto p = std::make_unique<Program::Impl>();
      p->inputSize = K;
      p->width = N;
      p->maximumRows = R;
      for (uint32_t rows = Program::kMinimumRows; rows < R; rows += Program::kRowStep) p->bucketRows.push_back(rows);
      p->bucketRows.push_back(R);
      if (!makeSurface(p->x, p->keep[0], R, K, 2) || !makeSurface(p->y, p->keep[1], R, N, 2)) return nullptr;
      for (uint32_t slot = 0; slot < Program::kSlots; ++slot) {
        if (!makeSurface(p->w[slot], p->keep[2 + slot], N, K, 1)) return nullptr;
        p->scales[slot] = backend.allocateBuffer(uint64_t{N} * sizeof(float), metal::BufferStorage::Shared, "ane scales");
      }
      NSMutableString *mil = [NSMutableString stringWithString:@"program(1.3)\n{\n"];
      for (uint32_t rows : p->bucketRows) {
        [mil appendFormat:@"    func r%u<ios18>(%@ x, %@ w) {\n", rows, bufferType(p->x, 2, rows, K), bufferType(p->w[0], 1, N, K)];
        [mil appendFormat:@"        tensor<fp16, %@> x_t = tensor_buffer_to_tensor<ios17>(input = x);\n", dims(rows, K)];
        [mil appendFormat:@"        tensor<int8, %@> w_t = tensor_buffer_to_tensor<ios17>(input = w);\n", dims(N, K)];
        // int8 / 128: the row scales carry the factor back
        [mil appendFormat:@"        tensor<fp16, %@> w_d = dequantize(input = w_t, scale = fp16(0x1p-7));\n", dims(N, K)];
        [mil appendFormat:@"        tensor<fp16, %@> yt = matmul(transpose_x = bool(false), transpose_y = bool(true), x = x_t, y = w_d);\n",
                          dims(rows, N)];
        const uint64_t yStride = p->y.stride / 2, yPlane = uint64_t{rows} * yStride;
        [mil appendFormat:@"        %@ y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([%llu, %llu, %llu, 1]));\n",
                          bufferType(p->y, 2, rows, N), (unsigned long long)yPlane, (unsigned long long)yPlane,
                          (unsigned long long)yStride];
        [mil appendString:@"    } -> (y);\n"];
      }
      [mil appendString:@"}\n"];
      if (!load(*p, mil, [NSString stringWithFormat:@"mm-%u-%u-%u", K, N, R])) return nullptr;
      return std::make_shared<Program>(std::move(p));
    }
  }

  void release(uint64_t done) {
    if (event.signaledValue < done) event.signaledValue = done;
  }

  // Records the evaluation as finished, once.
  bool finish(Pending &pending) {
    if (pending.finished.exchange(true)) return false;
    completed.fetch_add(1);
    return true;
  }

  // A failure stops the engine for the session and releases the GPU's wait.
  void fail(uint64_t done, const char *why) {
    if (!failed.exchange(true))
      std::fprintf(stderr, "splash: ANE evaluation %s; the GPU runs prefill alone from now on\n", why);
    release(done);
  }

  void launch(const std::shared_ptr<Program> &program, uint32_t slot, uint32_t rows, uint64_t ready, uint64_t done) {
    @autoreleasepool {
      Program::Impl &p = program->state();
      Program::Impl *state = &p;
      auto pending = std::make_shared<Pending>();
      pending->ready = ready;
      pending->done = done;
      launched.fetch_add(1);
      ++evaluations;
      const bool alreadyFailed = failed.load();
      int fault = 0;
      if (faultKind && evaluations == faultAt) fault = faultKind;
      // The watchdog starts when the GPU signals the input ready, and so do the test faults, as a real evaluation
      // would. After a failure the GPU's waits are released as the GPU reaches them.
      [event notifyListener:listener atValue:ready block:^(id<MTLSharedEvent> ev, uint64_t) {
        if (failed.load()) {
          finish(*pending);
          release(done);
          return;
        }
        if (fault == 1) {
          if (finish(*pending)) fail(done, "failed (test)");
          return;
        }
        if (fault == 2) {
          uint16_t *y = static_cast<uint16_t *>(state->y.buffer.contents());
          for (uint64_t i = 0; i < state->y.buffer.sizeBytes() / 2u; i += 7) y[i] = 0x7e00u;
          finish(*pending);
          if (ev.signaledValue < done) ev.signaledValue = done;
          return;
        }
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds), queue, ^{
          if (finish(*pending)) fail(done, "timed out");
        });
      }];
      if (alreadyFailed || fault) {
        if (alreadyFailed) finish(*pending);
        return;
      }
      uint32_t bucket = 0;
      while (bucket + 1 < p.bucketRows.size() && p.bucketRows[bucket] < rows) ++bucket;
      id signal = [signalClass signalEventWithValue:done symbolIndex:0 eventType:0 sharedEvent:event];
      id wait = [waitClass waitEventWithValue:ready sharedEvent:event eventType:0];
      id events = [eventsClass sharedEventsWithSignalEvents:@[ signal ] waitEvents:@[ wait ]];
      NSMutableArray *inputs = [NSMutableArray array];
      for (NSString *name in p.inputNames[bucket])
        [inputs addObject:[name isEqualToString:@"x"] ? p.x.object : p.w[slot].object];
      id request = [requestClass requestWithInputs:inputs inputIndices:p.inputIndices[bucket] outputs:@[ p.y.object ]
                                     outputIndices:p.outputIndices[bucket] weightsBuffer:nil perfStats:nil
                                    procedureIndex:p.procedureId[bucket] sharedEvents:events transactionHandle:nil];
      if (!events || !request) {
        if (finish(*pending)) fail(done, "was not created");
        return;
      }
      [request setCompletionHandler:^(BOOL ok, NSError *error) {
        if (finish(*pending) && !ok)
          fail(done, [NSString stringWithFormat:@"failed (%@)", error ? error.localizedDescription : @"no reason"].UTF8String);
      }];
      NSError *error = nil;
      if (![client evaluateWithModel:p.model options:@{} request:request qos:kQos error:&error] && finish(*pending))
        fail(done, [NSString stringWithFormat:@"was not queued (%@)", error.localizedDescription].UTF8String);
    }
  }
};

Runtime::Runtime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;

namespace {
std::unique_ptr<Runtime> &instance() {
  static std::unique_ptr<Runtime> runtime;
  return runtime;
}
} // namespace

Runtime *Runtime::get(metal::MetalBackend &backend) {
  static std::mutex mutex;
  static bool opened = false;
  std::lock_guard lock(mutex);
  if (!requested()) return nullptr;
  if (!opened) {
    opened = true;
    auto impl = std::make_unique<Impl>(backend);
    if (impl->open()) instance() = std::make_unique<Runtime>(std::move(impl));
    else std::fprintf(stderr, "splash: ANE unavailable; the GPU runs prefill alone\n");
  }
  Runtime *runtime = instance().get();
  return runtime && !runtime->impl_->failed.load() ? runtime : nullptr;
}

Runtime *Runtime::existing() noexcept { return instance().get(); }

std::shared_ptr<Program> Runtime::program(uint32_t inputSize, uint32_t width, uint32_t maximumRows) {
  for (const std::shared_ptr<Program> &program : impl_->programs)
    if (program->inputSize() == inputSize && program->width() == width && program->maximumRows() == maximumRows)
      return program;
  std::shared_ptr<Program> program = impl_->create(inputSize, width, maximumRows);
  if (!program) {
    impl_->failed = true;
    std::fprintf(stderr, "splash: ANE program %ux%u unavailable; the GPU runs prefill alone\n", inputSize, width);
    return nullptr;
  }
  if (impl_->logging) std::fprintf(stderr, "splash: ANE program %u x %u, up to %u rows\n", inputSize, width, maximumRows);
  impl_->programs.push_back(program);
  return program;
}

const metal::MetalBuffer &Runtime::status() const noexcept { return impl_->status; }

Runtime::Values Runtime::evaluate(metal::CommandGraph &graph, const std::shared_ptr<Program> &program,
                                  uint32_t rows) {
  const Values values{impl_->sequence + 1, impl_->sequence + 2, program->nextSlot()};
  impl_->sequence += 2;
  graph.beforeSubmit([impl = impl_.get(), program, rows, values] {
    impl->launch(program, values.slot, rows, values.ready, values.done);
  });
  return values;
}

bool Runtime::settle() {
  Impl &impl = *impl_;
  for (int i = 0; i < 3000 && impl.completed.load() < impl.launched.load(); ++i) usleep(1000);
  uint32_t *status = static_cast<uint32_t *>(impl.status.contents());
  if (impl.completed.load() < impl.launched.load()) impl.fail(impl.sequence, "did not finish");
  else if (*status) impl.fail(impl.sequence, "wrote non-finite rows");
  *status = 0;
  if (impl.logging)
    std::fprintf(stderr, "splash: ANE evaluations %llu%s\n", (unsigned long long)impl.evaluations,
                 impl.failed.load() ? " (failed)" : "");
  return !impl.failed.load();
}

} // namespace splash::ane
