// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

#import <Foundation/Foundation.h>
#import <React/RCTBlobManager.h>
#import <React/RCTBridge+Private.h>
#import <React/RCTLog.h>

// Note: Using below syntax for including ort c api and ort extensions headers to resolve a compiling error happened
// in an expo react native ios app when ort extensions enabled (a redefinition error of multiple object types defined
// within ORT C API header). It's an edge case that compiler allows both ort c api headers to be included when #include
// syntax doesn't match. For the case when extensions not enabled, it still requires a onnxruntime prefix directory for
// searching paths. Also in general, it's a convention to use #include for C/C++ headers rather then #import. See:
// https://google.github.io/styleguide/objcguide.html#import-and-include
// https://microsoft.github.io/objc-guide/Headers/ImportAndInclude.html
#ifdef ORT_ENABLE_EXTENSIONS
#include "coreml_provider_factory.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_extensions.h"
#else
#include "onnxruntime/coreml_provider_factory.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

@implementation OnnxruntimeModule

struct SessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<const char *> inputNames;
  std::vector<Ort::AllocatedStringPtr> inputNames_ptrs;
  std::vector<const char *> outputNames;
  std::vector<Ort::AllocatedStringPtr> outputNames_ptrs;
};

static Ort::Env *ortEnv = new Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
static NSMutableDictionary *sessionMap = [NSMutableDictionary dictionary];
static Ort::AllocatorWithDefaultOptions ortAllocator;

static int nextSessionId = 0;
- (NSString *)getNextSessionKey {
  NSString *key = @(nextSessionId).stringValue;
  nextSessionId++;
  return key;
}

RCT_EXPORT_MODULE(Onnxruntime)

RCTBlobManager *blobManager = nil;

- (void)checkBlobManager {
  if (blobManager == nil) {
    blobManager = [[RCTBridge currentBridge] moduleForClass:RCTBlobManager.class];
    if (blobManager == nil) {
      @throw @"RCTBlobManager is not initialized";
    }
  }
}

- (void)setBlobManager:(RCTBlobManager *)manager {
  blobManager = manager;
}

/**
 * React native binding API to load a model using given uri.
 *
 * @param modelPath a model file location. it's used as a key when multiple sessions are created, i.e. multiple models
 * are loaded.
 * @param options onnxruntime session options
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
RCT_EXPORT_METHOD(loadModel
                  : (NSString *)modelPath options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *resultMap = [self loadModel:modelPath options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to load model", nil);
  }
}

/**
 * React native binding API to load a model using blob object that data stored in RCTBlobManager.
 *
 * @param modelDataBlob a model data blob object
 * @param options onnxruntime session options
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
RCT_EXPORT_METHOD(loadModelFromBlob
                  : (NSDictionary *)modelDataBlob options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    [self checkBlobManager];
    NSString *blobId = [modelDataBlob objectForKey:@"blobId"];
    long size = [[modelDataBlob objectForKey:@"size"] longValue];
    long offset = [[modelDataBlob objectForKey:@"offset"] longValue];
    auto modelData = [blobManager resolve:blobId offset:offset size:size];
    NSDictionary *resultMap = [self loadModelFromBuffer:modelData options:options];
    [blobManager remove:blobId];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to load model from buffer", nil);
  }
}

/**
 * React native binding API to dispose a session using given key from loadModel()
 *
 * @param key a model path location given at loadModel()
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 */
RCT_EXPORT_METHOD(dispose
                  : (NSString *)key resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    [self dispose:key];
    resolve(nil);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to dispose session", nil);
  }
}

/**
 * React native binding API to run a model using given uri.
 *
 * @param url a model path location given at loadModel()
 * @param input an input tensor
 * @param output an output names to be returned
 * @param options onnxruntime run options
 * @param resolve callback for returning an inference result back to react native js
 * @param reject callback for returning an error back to react native js
 */
RCT_EXPORT_METHOD(run
                  : (NSString *)url input
                  : (NSDictionary *)input output
                  : (NSArray *)output options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSDictionary *resultMap = [self run:url input:input output:output options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to run model", nil);
  }
}

/**
 * Load a model using given model path.
 *
 * @param modelPath a model file location.
 * @param options onnxruntime session options.
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
- (NSDictionary *)loadModel:(NSString *)modelPath options:(NSDictionary *)options {
  return [self loadModelImpl:modelPath modelData:nil options:options];
}

/**
 * Load a model using given model data array
 *
 * @param modelData the model data buffer.
 * @param options onnxruntime session options
 */
- (NSDictionary *)loadModelFromBuffer:(NSData *)modelData options:(NSDictionary *)options {
  return [self loadModelImpl:@"" modelData:modelData options:options];
}

/**
 * Load model implementation method given either model data array or model path
 *
 * @param modelPath the model file location.
 * @param modelData the model data buffer.
 * @param options onnxruntime session options.
 */
- (NSDictionary *)loadModelImpl:(NSString *)modelPath modelData:(NSData *)modelData options:(NSDictionary *)options {
  SessionInfo *sessionInfo = nullptr;
  sessionInfo = new SessionInfo();
  Ort::SessionOptions sessionOptions = [self parseSessionOptions:options];

#ifdef ORT_ENABLE_EXTENSIONS
  Ort::ThrowOnError(RegisterCustomOps(sessionOptions, OrtGetApiBase()));
#endif

  if (modelData == nil) {
    sessionInfo->session.reset(new Ort::Session(*ortEnv, [modelPath UTF8String], sessionOptions));
  } else {
    NSUInteger dataLength = [modelData length];
    Byte *modelBytes = (Byte *)[modelData bytes];
    sessionInfo->session.reset(new Ort::Session(*ortEnv, modelBytes, (size_t)dataLength, sessionOptions));
  }

  sessionInfo->inputNames.reserve(sessionInfo->session->GetInputCount());
  for (size_t i = 0; i < sessionInfo->session->GetInputCount(); ++i) {
    auto inputName = sessionInfo->session->GetInputNameAllocated(i, ortAllocator);
    sessionInfo->inputNames.emplace_back(inputName.get());
    sessionInfo->inputNames_ptrs.emplace_back(std::move(inputName));
  }

  sessionInfo->outputNames.reserve(sessionInfo->session->GetOutputCount());
  for (size_t i = 0; i < sessionInfo->session->GetOutputCount(); ++i) {
    auto outputName = sessionInfo->session->GetOutputNameAllocated(i, ortAllocator);
    sessionInfo->outputNames.emplace_back(outputName.get());
    sessionInfo->outputNames_ptrs.emplace_back(std::move(outputName));
  }

  NSString *key = [self getNextSessionKey];
  NSValue *value = [NSValue valueWithPointer:(void *)sessionInfo];
  sessionMap[key] = value;

  NSMutableDictionary *resultMap = [NSMutableDictionary dictionary];
  resultMap[@"key"] = key;

  NSMutableArray *inputNames = [NSMutableArray array];
  for (auto inputName : sessionInfo->inputNames) {
    [inputNames addObject:[NSString stringWithCString:inputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"inputNames"] = inputNames;

  NSMutableArray *outputNames = [NSMutableArray array];
  for (auto outputName : sessionInfo->outputNames) {
    [outputNames addObject:[NSString stringWithCString:outputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"outputNames"] = outputNames;

  return resultMap;
}

/**
 * Dispose a session given a key.
 *
 * @param key a session key returned from loadModel()
 */
- (void)dispose:(NSString *)key {
  NSValue *value = [sessionMap objectForKey:key];
  if (value == nil) {
    NSException *exception = [NSException exceptionWithName:@"onnxruntime"
                                                     reason:@"can't find onnxruntime session"
                                                   userInfo:nil];
    @throw exception;
  }
  [sessionMap removeObjectForKey:key];
  SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];
  delete sessionInfo;
  sessionInfo = nullptr;
}

/**
 * Run a model using given uri.
 *
 * @param url a model path location given at loadModel()
 * @param input an input tensor
 * @param output an output names to be returned
 * @param options onnxruntime run options
 */
- (NSDictionary *)run:(NSString *)url
                input:(NSDictionary *)input
               output:(NSArray *)output
              options:(NSDictionary *)options {
  NSValue *value = [sessionMap objectForKey:url];
  if (value == nil) {
    NSException *exception = [NSException exceptionWithName:@"onnxruntime"
                                                     reason:@"can't find onnxruntime session"
                                                   userInfo:nil];
    @throw exception;
  }
  SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];

  [self checkBlobManager];

  std::vector<Ort::Value> feeds;
  std::vector<Ort::MemoryAllocation> allocations;
  feeds.reserve(sessionInfo->inputNames.size());
  for (auto inputName : sessionInfo->inputNames) {
    NSDictionary *inputTensor = [input objectForKey:[NSString stringWithUTF8String:inputName]];
    if (inputTensor == nil) {
      NSException *exception = [NSException exceptionWithName:@"onnxruntime" reason:@"can't find input" userInfo:nil];
      @throw exception;
    }

    Ort::Value value = [TensorHelper createInputTensor:blobManager
                                                 input:inputTensor
                                          ortAllocator:ortAllocator
                                           allocations:allocations];
    feeds.emplace_back(std::move(value));
  }

  std::vector<const char *> requestedOutputs;
  requestedOutputs.reserve(output.count);
  for (NSString *outputName : output) {
    requestedOutputs.emplace_back([outputName UTF8String]);
  }
  Ort::RunOptions runOptions = [self parseRunOptions:options];

  auto result =
      sessionInfo->session->Run(runOptions, sessionInfo->inputNames.data(), feeds.data(),
                                sessionInfo->inputNames.size(), requestedOutputs.data(), requestedOutputs.size());

  NSDictionary *resultMap = [TensorHelper createOutputTensor:blobManager outputNames:requestedOutputs values:result];

  return resultMap;
}

static NSDictionary *graphOptimizationLevelTable = @{
  @"disabled" : @(ORT_DISABLE_ALL),
  @"basic" : @(ORT_ENABLE_BASIC),
  @"extended" : @(ORT_ENABLE_EXTENDED),
  @"all" : @(ORT_ENABLE_ALL)
};

static NSDictionary *executionModeTable = @{@"sequential" : @(ORT_SEQUENTIAL), @"parallel" : @(ORT_PARALLEL)};

- (Ort::SessionOptions)parseSessionOptions:(NSDictionary *)options {
  Ort::SessionOptions sessionOptions;

  if ([options objectForKey:@"intraOpNumThreads"]) {
    int intraOpNumThreads = [[options objectForKey:@"intraOpNumThreads"] intValue];
    if (intraOpNumThreads > 0 && intraOpNumThreads < INT_MAX) {
      sessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    }
  }

  if ([options objectForKey:@"interOpNumThreads"]) {
    int interOpNumThreads = [[options objectForKey:@"interOpNumThreads"] intValue];
    if (interOpNumThreads > 0 && interOpNumThreads < INT_MAX) {
      sessionOptions.SetInterOpNumThreads(interOpNumThreads);
    }
  }

  if ([options objectForKey:@"graphOptimizationLevel"]) {
    NSString *graphOptimizationLevel = [[options objectForKey:@"graphOptimizationLevel"] stringValue];
    if ([graphOptimizationLevelTable objectForKey:graphOptimizationLevel]) {
      sessionOptions.SetGraphOptimizationLevel(
          (GraphOptimizationLevel)[[graphOptimizationLevelTable objectForKey:graphOptimizationLevel] intValue]);
    }
  }

  if ([options objectForKey:@"enableCpuMemArena"]) {
    BOOL enableCpuMemArena = [[options objectForKey:@"enableCpuMemArena"] boolValue];
    if (enableCpuMemArena) {
      sessionOptions.EnableCpuMemArena();
    } else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  if ([options objectForKey:@"enableMemPattern"]) {
    BOOL enableMemPattern = [[options objectForKey:@"enableMemPattern"] boolValue];
    if (enableMemPattern) {
      sessionOptions.EnableMemPattern();
    } else {
      sessionOptions.DisableMemPattern();
    }
  }

  if ([options objectForKey:@"executionMode"]) {
    NSString *executionMode = [[options objectForKey:@"executionMode"] stringValue];
    if ([executionModeTable objectForKey:executionMode]) {
      sessionOptions.SetExecutionMode((ExecutionMode)[[executionModeTable objectForKey:executionMode] intValue]);
    }
  }

  if ([options objectForKey:@"executionProviders"]) {
    NSArray *executionProviders = [options objectForKey:@"executionProviders"];
    for (auto *executionProvider in executionProviders) {
      NSString *epName = nil;
      bool useOptions = false;
      if ([executionProvider isKindOfClass:[NSString class]]) {
        epName = (NSString *)executionProvider;
      } else {
        epName = [executionProvider objectForKey:@"name"];
        useOptions = true;
      }
      if ([epName isEqualToString:@"coreml"]) {
        uint32_t coreml_flags = 0;
        if (useOptions) {
          if ([[executionProvider objectForKey:@"useCPUOnly"] boolValue]) {
            coreml_flags |= COREML_FLAG_USE_CPU_ONLY;
          }
          if ([[executionProvider objectForKey:@"enableOnSubgraph"] boolValue]) {
            coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
          }
          if ([[executionProvider objectForKey:@"onlyEnableDeviceWithANE"] boolValue]) {
            coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
          }
        }
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags));
      } else if ([epName isEqualToString:@"xnnpack"]) {
        sessionOptions.AppendExecutionProvider("XNNPACK", {});
      } else if ([epName isEqualToString:@"cpu"]) {
        continue;
      } else {
        NSException *exception = [NSException exceptionWithName:@"onnxruntime"
                                                         reason:@"unsupported execution provider"
                                                       userInfo:nil];
        @throw exception;
      }
    }
  }

  if ([options objectForKey:@"logId"]) {
    NSString *logId = [[options objectForKey:@"logId"] stringValue];
    sessionOptions.SetLogId([logId UTF8String]);
  }

  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    sessionOptions.SetLogSeverityLevel(logSeverityLevel);
  }

  return sessionOptions;
}

- (Ort::RunOptions)parseRunOptions:(NSDictionary *)options {
  Ort::RunOptions runOptions;

  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    runOptions.SetRunLogSeverityLevel(logSeverityLevel);
  }

  if ([options objectForKey:@"tag"]) {
    NSString *tag = [[options objectForKey:@"tag"] stringValue];
    runOptions.SetRunTag([tag UTF8String]);
  }

  return runOptions;
}

- (void)dealloc {
  NSEnumerator *iterator = [sessionMap keyEnumerator];
  while (NSString *key = [iterator nextObject]) {
    [self dispose:key];
  }
  blobManager = nullptr;
}

@end
