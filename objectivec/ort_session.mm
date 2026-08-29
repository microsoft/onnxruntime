// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_session_internal.h"

#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#import "cxx_api.h"
#import "cxx_utils.h"
#import "error_utils.h"
#import "ort_enums_internal.h"
#import "ort_env_internal.h"
#import "ort_value_internal.h"

namespace {
enum class NamedValueType {
  Input,
  OverridableInitializer,
  Output,
};

OrtStatus* CreateCallbackStatus(OrtErrorCode code, NSString* message) {
  const char* messageCstr = message.UTF8String;
  return Ort::GetApi().CreateStatus(code, messageCstr != nullptr ? messageCstr : "EPContext data read block failed");
}

OrtStatus* _Nullable ORT_API_CALL EpContextDataReadCallback(void* state,
                                                            const char* name,
                                                            OrtAllocator* allocator,
                                                            void** buffer,
                                                            size_t* dataSize);
}  // namespace

NS_ASSUME_NONNULL_BEGIN

@interface ORTEpContextDataReadRegistration : NSObject

@property(nonatomic, copy, readonly) ORTEpContextDataReadBlock block;
@property(nonatomic, readonly) size_t maxDataSize;

- (instancetype)initWithBlock:(ORTEpContextDataReadBlock)block
                  maxDataSize:(size_t)maxDataSize;

@end

@implementation ORTEpContextDataReadRegistration

- (instancetype)initWithBlock:(ORTEpContextDataReadBlock)block
                  maxDataSize:(size_t)maxDataSize {
  if ((self = [super init]) == nil) {
    return nil;
  }

  _block = [block copy];
  _maxDataSize = maxDataSize;
  return self;
}

@end

namespace {
OrtStatus* _Nullable ORT_API_CALL EpContextDataReadCallback(void* state,
                                                            const char* name,
                                                            OrtAllocator* allocator,
                                                            void** buffer,
                                                            size_t* dataSize) {
  if (buffer == nullptr || dataSize == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "EPContext data read callback received null output parameters");
  }

  *buffer = nullptr;
  *dataSize = 0;

  if (state == nullptr || name == nullptr || allocator == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "EPContext data read callback received invalid input parameters");
  }

  try {
    @try {
      @autoreleasepool {
        ORTEpContextDataReadRegistration* registration =
            (__bridge ORTEpContextDataReadRegistration*)state;
        NSString* dataName = [NSString stringWithUTF8String:name];
        if (dataName == nil) {
          return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                            "EPContext data name is not valid UTF-8");
        }

        NSError* callbackError = nil;
        NSData* data = registration.block(dataName, &callbackError);
        if (data == nil) {
          if (callbackError == nil) {
            return Ort::GetApi().CreateStatus(
                ORT_FAIL, "EPContext data read block returned nil without setting an error");
          }

          NSString* message =
              [NSString stringWithFormat:@"EPContext data read block failed for '%@': %@",
                                         dataName, callbackError.localizedDescription];
          return CreateCallbackStatus(ORT_FAIL, message);
        }

        if (![data isKindOfClass:[NSData class]]) {
          return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                            "EPContext data read block returned an object that is not NSData");
        }

        const NSUInteger length = data.length;
        if (length > registration.maxDataSize) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "EPContext data returned by the read block exceeds maxDataSize");
        }
        if (length == 0) {
          return nullptr;
        }

        const void* bytes = data.bytes;
        if (bytes == nullptr) {
          return Ort::GetApi().CreateStatus(
              ORT_INVALID_ARGUMENT,
              "EPContext data read block returned non-empty NSData with null bytes");
        }

        void* allocatedBuffer = allocator->Alloc(allocator, static_cast<size_t>(length));
        if (allocatedBuffer == nullptr) {
          return Ort::GetApi().CreateStatus(ORT_FAIL,
                                            "Failed to allocate the EPContext data output buffer");
        }

        std::memcpy(allocatedBuffer, bytes, length);
        *buffer = allocatedBuffer;
        *dataSize = static_cast<size_t>(length);
        return nullptr;
      }
    } @catch (NSException* /*exception*/) {
      return Ort::GetApi().CreateStatus(
          ORT_FAIL, "EPContext data read block raised an Objective-C exception");
    }
  } catch (const std::exception& exception) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, exception.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_FAIL,
                                      "EPContext data read callback raised an unknown exception");
  }
}
}  // namespace

@implementation ORTSession {
  ORTEnv* _env;  // keep a strong reference so the ORTEnv doesn't get destroyed before this does
  id _epContextDataReadRegistration;
  std::optional<Ort::Session> _session;
}

#pragma mark - Public

- (nullable instancetype)initWithEnv:(ORTEnv*)env
                           modelPath:(NSString*)path
                      sessionOptions:(nullable ORTSessionOptions*)sessionOptions
                               error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    if (!sessionOptions) {
      sessionOptions = [[ORTSessionOptions alloc] initWithError:error];
      if (!sessionOptions) {
        return nil;
      }
    }

    id registrationSnapshot;
    Ort::SessionOptions sessionOptionsSnapshot{nullptr};
    @synchronized(sessionOptions) {
      registrationSnapshot = [sessionOptions epContextDataReadRegistrationSnapshot];
      sessionOptionsSnapshot = [sessionOptions CXXAPIOrtSessionOptions].Clone();
    }

    std::optional<Ort::Session> session;
    session.emplace([env CXXAPIOrtEnv],
                    path.UTF8String,
                    sessionOptionsSnapshot);

    _env = env;
    _epContextDataReadRegistration = registrationSnapshot;
    _session = std::move(session);
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (void)dealloc {
  // Execution providers may retain the raw callback state, so release the native session first.
  _session.reset();
  _epContextDataReadRegistration = nil;
}

- (BOOL)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
              outputs:(NSDictionary<NSString*, ORTValue*>*)outputs
           runOptions:(nullable ORTRunOptions*)runOptions
                error:(NSError**)error {
  try {
    if (!runOptions) {
      runOptions = [[ORTRunOptions alloc] initWithError:error];
      if (!runOptions) {
        return NO;
      }
    }

    std::vector<const char*> inputNames, outputNames;
    std::vector<const OrtValue*> inputCAPIValues;
    std::vector<OrtValue*> outputCAPIValues;

    inputNames.reserve(inputs.count);
    inputCAPIValues.reserve(inputs.count);
    for (NSString* inputName in inputs) {
      inputNames.push_back(inputName.UTF8String);
      inputCAPIValues.push_back(static_cast<const OrtValue*>([inputs[inputName] CXXAPIOrtValue]));
    }

    outputNames.reserve(outputs.count);
    outputCAPIValues.reserve(outputs.count);
    for (NSString* outputName in outputs) {
      outputNames.push_back(outputName.UTF8String);
      outputCAPIValues.push_back(static_cast<OrtValue*>([outputs[outputName] CXXAPIOrtValue]));
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, [runOptions CXXAPIOrtRunOptions],
                                        inputNames.data(), inputCAPIValues.data(), inputNames.size(),
                                        outputNames.data(), outputNames.size(), outputCAPIValues.data()));

    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (nullable NSDictionary<NSString*, ORTValue*>*)runWithInputs:(NSDictionary<NSString*, ORTValue*>*)inputs
                                                  outputNames:(NSSet<NSString*>*)outputNameSet
                                                   runOptions:(nullable ORTRunOptions*)runOptions
                                                        error:(NSError**)error {
  try {
    if (!runOptions) {
      runOptions = [[ORTRunOptions alloc] initWithError:error];
      if (!runOptions) {
        return nil;
      }
    }

    NSArray<NSString*>* outputNameArray = outputNameSet.allObjects;

    std::vector<const char*> inputNames, outputNames;
    std::vector<const OrtValue*> inputCAPIValues;
    std::vector<OrtValue*> outputCAPIValues;

    inputNames.reserve(inputs.count);
    inputCAPIValues.reserve(inputs.count);
    for (NSString* inputName in inputs) {
      inputNames.push_back(inputName.UTF8String);
      inputCAPIValues.push_back(static_cast<const OrtValue*>([inputs[inputName] CXXAPIOrtValue]));
    }

    outputNames.reserve(outputNameArray.count);
    outputCAPIValues.reserve(outputNameArray.count);
    for (NSString* outputName in outputNameArray) {
      outputNames.push_back(outputName.UTF8String);
      outputCAPIValues.push_back(nullptr);
    }

    Ort::ThrowOnError(Ort::GetApi().Run(*_session, [runOptions CXXAPIOrtRunOptions],
                                        inputNames.data(), inputCAPIValues.data(), inputNames.size(),
                                        outputNames.data(), outputNames.size(), outputCAPIValues.data()));

    NSMutableDictionary<NSString*, ORTValue*>* outputs = [[NSMutableDictionary alloc] init];
    for (NSUInteger i = 0; i < outputNameArray.count; ++i) {
      // Wrap the C OrtValue in a C++ Ort::Value to automatically handle its release.
      // Then, transfer that C++ Ort::Value to a new ORTValue.
      Ort::Value outputCXXAPIValue{outputCAPIValues[i]};
      ORTValue* outputValue = [[ORTValue alloc] initWithCXXAPIOrtValue:std::move(outputCXXAPIValue)
                                                    externalTensorData:nil
                                                                 error:error];
      if (!outputValue) {
        // clean up remaining C OrtValues which haven't been wrapped by a C++ Ort::Value yet
        for (NSUInteger j = i + 1; j < outputNameArray.count; ++j) {
          Ort::GetApi().ReleaseValue(outputCAPIValues[j]);
        }
        return nil;
      }

      outputs[outputNameArray[i]] = outputValue;
    }

    return outputs;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<NSString*>*)inputNamesWithError:(NSError**)error {
  return [self namesWithType:NamedValueType::Input error:error];
}

- (nullable NSArray<NSString*>*)overridableInitializerNamesWithError:(NSError**)error {
  return [self namesWithType:NamedValueType::OverridableInitializer error:error];
}

- (nullable NSArray<NSString*>*)outputNamesWithError:(NSError**)error {
  return [self namesWithType:NamedValueType::Output error:error];
}

#pragma mark - Private

- (nullable NSArray<NSString*>*)namesWithType:(NamedValueType)namedValueType
                                        error:(NSError**)error {
  try {
    auto getCount = [&session = *_session, namedValueType]() {
      if (namedValueType == NamedValueType::Input) {
        return session.GetInputCount();
      } else if (namedValueType == NamedValueType::OverridableInitializer) {
        return session.GetOverridableInitializerCount();
      } else {
        return session.GetOutputCount();
      }
    };

    auto getName = [&session = *_session, namedValueType](size_t i, OrtAllocator* allocator) {
      if (namedValueType == NamedValueType::Input) {
        return session.GetInputNameAllocated(i, allocator);
      } else if (namedValueType == NamedValueType::OverridableInitializer) {
        return session.GetOverridableInitializerNameAllocated(i, allocator);
      } else {
        return session.GetOutputNameAllocated(i, allocator);
      }
    };

    const size_t nameCount = getCount();

    Ort::AllocatorWithDefaultOptions allocator;
    NSMutableArray<NSString*>* result = [NSMutableArray arrayWithCapacity:nameCount];

    for (size_t i = 0; i < nameCount; ++i) {
      auto name = getName(i, allocator);
      NSString* nameNsstr = utils::toNSString(name.get());
      [result addObject:nameNsstr];
    }

    return result;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

@end

@implementation ORTSessionOptions {
  ORTEpContextDataReadRegistration* _epContextDataReadRegistration;
  std::optional<Ort::SessionOptions> _sessionOptions;
}

#pragma mark - Public

- (nullable instancetype)initWithError:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _sessionOptions = Ort::SessionOptions{};
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)appendExecutionProvider:(NSString*)providerName
                providerOptions:(NSDictionary<NSString*, NSString*>*)providerOptions
                          error:(NSError**)error {
  try {
    std::unordered_map<std::string, std::string> options;
    NSArray* keys = [providerOptions allKeys];

    for (NSString* key in keys) {
      NSString* value = [providerOptions objectForKey:key];
      options.emplace(key.UTF8String, value.UTF8String);
    }

    _sessionOptions->AppendExecutionProvider(providerName.UTF8String, options);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
}

- (BOOL)setIntraOpNumThreads:(int)intraOpNumThreads
                       error:(NSError**)error {
  try {
    _sessionOptions->SetIntraOpNumThreads(intraOpNumThreads);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setGraphOptimizationLevel:(ORTGraphOptimizationLevel)graphOptimizationLevel
                            error:(NSError**)error {
  try {
    _sessionOptions->SetGraphOptimizationLevel(
        PublicToCAPIGraphOptimizationLevel(graphOptimizationLevel));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setOptimizedModelFilePath:(NSString*)optimizedModelFilePath
                            error:(NSError**)error {
  try {
    _sessionOptions->SetOptimizedModelFilePath(optimizedModelFilePath.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setLogID:(NSString*)logID
           error:(NSError**)error {
  try {
    _sessionOptions->SetLogId(logID.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error {
  try {
    _sessionOptions->SetLogSeverityLevel(PublicToCAPILoggingLevel(loggingLevel));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error {
  try {
    _sessionOptions->AddConfigEntry(key.UTF8String, value.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setEpContextDataReadBlock:(ORTEpContextDataReadBlock)block
                      maxDataSize:(NSUInteger)maxDataSize
                            error:(NSError**)error {
  try {
    if (block == nil) {
      ORT_CXX_API_THROW("EPContext data read block must not be nil", ORT_INVALID_ARGUMENT);
    }
    if (maxDataSize == 0 || maxDataSize == std::numeric_limits<NSUInteger>::max()) {
      ORT_CXX_API_THROW("maxDataSize must be finite and greater than zero", ORT_INVALID_ARGUMENT);
    }

    @synchronized(self) {
      ORTEpContextDataReadRegistration* registration =
          [[ORTEpContextDataReadRegistration alloc] initWithBlock:block
                                                      maxDataSize:static_cast<size_t>(maxDataSize)];
      if (registration == nil) {
        ORT_CXX_API_THROW("Failed to create EPContext data read block registration", ORT_FAIL);
      }
      _sessionOptions->SetEpContextDataReadFunc(
          EpContextDataReadCallback, (__bridge void*)registration,
          registration.maxDataSize);
      _epContextDataReadRegistration = registration;
    }

    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)clearEpContextDataReadBlockWithError:(NSError**)error {
  try {
    @synchronized(self) {
      _sessionOptions->ClearEpContextDataReadFunc();
      _epContextDataReadRegistration = nil;
    }

    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)registerCustomOpsUsingFunction:(NSString*)registrationFuncName
                                 error:(NSError**)error {
  try {
    _sessionOptions->RegisterCustomOpsUsingFunction(registrationFuncName.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)registerCustomOpsUsingFunctionPointer:(ORTCAPIRegisterCustomOpsFnPtr)registerCustomOpsFn
                                        error:(NSError**)error {
  try {
    if (!registerCustomOpsFn) {
      ORT_CXX_API_THROW("registerCustomOpsFn must not be null", ORT_INVALID_ARGUMENT);
    }
    Ort::ThrowOnError((*registerCustomOpsFn)(static_cast<OrtSessionOptions*>(*_sessionOptions),
                                             OrtGetApiBase()));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)enableOrtExtensionsCustomOpsWithError:(NSError**)error {
  try {
    _sessionOptions->EnableOrtCustomOps();
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

#pragma mark - Internal

- (Ort::SessionOptions&)CXXAPIOrtSessionOptions {
  return *_sessionOptions;
}

- (nullable id)epContextDataReadRegistrationSnapshot {
  @synchronized(self) {
    return _epContextDataReadRegistration;
  }
}

@end

@implementation ORTRunOptions {
  std::optional<Ort::RunOptions> _runOptions;
}

#pragma mark - Public

- (nullable instancetype)initWithError:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _runOptions = Ort::RunOptions{};
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)setLogTag:(NSString*)logTag
            error:(NSError**)error {
  try {
    _runOptions->SetRunTag(logTag.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setLogSeverityLevel:(ORTLoggingLevel)loggingLevel
                      error:(NSError**)error {
  try {
    _runOptions->SetRunLogSeverityLevel(PublicToCAPILoggingLevel(loggingLevel));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)addConfigEntryWithKey:(NSString*)key
                        value:(NSString*)value
                        error:(NSError**)error {
  try {
    _runOptions->AddConfigEntry(key.UTF8String, value.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

#pragma mark - Internal

- (Ort::RunOptions&)CXXAPIOrtRunOptions {
  return *_runOptions;
}

@end

NS_ASSUME_NONNULL_END
