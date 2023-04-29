// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

#import <Foundation/Foundation.h>
#import <React/RCTLog.h>
#import <onnxruntime/onnxruntime_cxx_api.h>

#ifdef ORT_ENABLE_EXTENSIONS
extern "C" {
// Note: Declared in onnxruntime_extensions.h but forward declared here to resolve a build issue:
// (A compilation error happened while building an expo react native ios app, onnxruntime_c_api.h header
// included in the onnxruntime_extensions.h leads to a redefinition conflicts with multiple object defined in the ORT C
// API.) So doing a forward declaration here instead of #include "onnxruntime_extensions.h" as a workaround for now
// before we have a fix.
// TODO: Investigate if we can include onnxruntime_extensions.h here
OrtStatus *RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api);
} // Extern C
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
 * React native binding API to load a model using BASE64 encoded model data string.
 *
 * @param modelData the BASE64 encoded model data string
 * @param options onnxruntime session options
 * @param resolve callback for returning output back to react native js
 * @param reject callback for returning an error back to react native js
 * @note when run() is called, the same modelPath must be passed into the first parameter.
 */
RCT_EXPORT_METHOD(loadModelFromBase64EncodedBuffer
                  : (NSString *)modelDataBase64EncodedString options
                  : (NSDictionary *)options resolver
                  : (RCTPromiseResolveBlock)resolve rejecter
                  : (RCTPromiseRejectBlock)reject) {
  @try {
    NSData *modelDataDecoded = [[NSData alloc] initWithBase64EncodedString:modelDataBase64EncodedString options:0];
    NSDictionary *resultMap = [self loadModelFromBuffer:modelDataDecoded options:options];
    resolve(resultMap);
  } @catch (...) {
    reject(@"onnxruntime", @"failed to load model from buffer", nil);
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

  std::vector<Ort::Value> feeds;
  std::vector<Ort::MemoryAllocation> allocations;
  feeds.reserve(sessionInfo->inputNames.size());
  for (auto inputName : sessionInfo->inputNames) {
    NSDictionary *inputTensor = [input objectForKey:[NSString stringWithUTF8String:inputName]];
    if (inputTensor == nil) {
      NSException *exception = [NSException exceptionWithName:@"onnxruntime" reason:@"can't find input" userInfo:nil];
      @throw exception;
    }

    Ort::Value value = [TensorHelper createInputTensor:inputTensor ortAllocator:ortAllocator allocations:allocations];
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

  NSDictionary *resultMap = [TensorHelper createOutputTensor:requestedOutputs values:result];

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
    NSValue *value = [sessionMap objectForKey:key];
    SessionInfo *sessionInfo = (SessionInfo *)[value pointerValue];
    delete sessionInfo;
    sessionInfo = nullptr;
  }
}

@end
