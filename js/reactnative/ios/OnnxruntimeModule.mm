// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#import <React/RCTLog.h>
#import "onnxruntime_cxx_api.h"
#import "OnnxruntimeModule.h"
#import "TensorHelper.h"

@implementation OnnxruntimeModule

struct SessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<Ort::MemoryAllocation> allocations;
  std::vector<const char*> inputNames;
  std::vector<const char*> outputNames;
};

static Ort::Env* ortEnv = new Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
static NSMutableDictionary* sessionMap = [NSMutableDictionary dictionary];
static Ort::AllocatorWithDefaultOptions ortAllocator;

RCT_EXPORT_MODULE(Onnxruntime)

RCT_EXPORT_METHOD(loadModel:(NSString*)modelPath
                  options:(NSDictionary*)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  @try {
    NSDictionary* resultMap = [self loadModel:modelPath options:options];
    resolve(resultMap);
  }
  @catch(NSException* exception) {
    reject(@"onnxruntime", @"can't load model", nil);
  }
}

RCT_EXPORT_METHOD(run:(NSString*)url
                  input:(NSDictionary*)input
                  output:(NSArray*)output
                  options:(NSDictionary*)options
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)
{
  @try {
    NSDictionary* resultMap = [self run:url input:input output:output options:options];
    resolve(resultMap);
  }
  @catch(NSException* exception) {
    reject(@"onnxruntime", @"can't run model", nil);
  }
}

-(NSDictionary*)loadModel:(NSString*)modelPath options:(NSDictionary*)options {
  NSValue* value = [sessionMap objectForKey:modelPath];
  SessionInfo* sessionInfo = nullptr;
  if (value == nil) {
    sessionInfo = new SessionInfo();
    
    Ort::SessionOptions sessionOptions = [self parseSessionOptions:options];
    sessionInfo->session.reset(new Ort::Session(*ortEnv, [modelPath UTF8String], sessionOptions));

    sessionInfo->inputNames.reserve(sessionInfo->session->GetInputCount());
    for (size_t i = 0; i < sessionInfo->session->GetInputCount(); ++i) {
      auto inputName = sessionInfo->session->GetInputName(i, ortAllocator);
      sessionInfo->allocations.emplace_back(ortAllocator, inputName, strlen(inputName) + 1);
      sessionInfo->inputNames.emplace_back(inputName);
    }

    sessionInfo->outputNames.reserve(sessionInfo->session->GetOutputCount());
    for (size_t i = 0; i < sessionInfo->session->GetOutputCount(); ++i) {
      auto outputName = sessionInfo->session->GetOutputName(i, ortAllocator);
      sessionInfo->allocations.emplace_back(ortAllocator, outputName, strlen(outputName) + 1);
      sessionInfo->outputNames.emplace_back(outputName);
    }
    
    value = [NSValue valueWithPointer:(void*)sessionInfo];
    sessionMap[modelPath] = value;
  } else {
    sessionInfo = (SessionInfo*)[value pointerValue];
  }
  
  NSMutableDictionary* resultMap = [NSMutableDictionary dictionary];
  resultMap[@"key"] = modelPath;
  
  NSMutableArray* inputNames = [NSMutableArray array];
  for (auto inputName : sessionInfo->inputNames) {
    [inputNames addObject:[NSString stringWithCString:inputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"inputNames"] = inputNames;
  
  NSMutableArray* outputNames = [NSMutableArray array];
  for (auto outputName : sessionInfo->outputNames) {
    [outputNames addObject:[NSString stringWithCString:outputName encoding:NSUTF8StringEncoding]];
  }
  resultMap[@"outputNames"] = outputNames;
  
  return resultMap;
}

-(NSDictionary*)run:(NSString*)url
              input:(NSDictionary*)input
             output:(NSArray*)output
            options:(NSDictionary*)options {
  NSValue* value = [sessionMap objectForKey:url];
  if (value == nil) {
    NSException* exception = [NSException exceptionWithName:@"onnxruntime" reason:@"can't find onnxruntime session" userInfo:nil];
    @throw exception;
  }
  SessionInfo* sessionInfo = (SessionInfo*)[value pointerValue];
  
  std::vector<Ort::Value> feeds;
  std::vector<Ort::MemoryAllocation> allocations;
  feeds.reserve(sessionInfo->inputNames.size());
  for (auto inputName : sessionInfo->inputNames) {
    NSDictionary* inputTensor = [input objectForKey:[NSString stringWithUTF8String:inputName]];
    if (inputTensor == nil) {
      NSException* exception = [NSException exceptionWithName:@"onnxruntime" reason:@"can't find input" userInfo:nil];
      @throw exception;
    }
    
    Ort::Value value = [TensorHelper createInputTensor:inputTensor ortAllocator:ortAllocator allocations:allocations];
    feeds.emplace_back(std::move(value));
  }
    
  std::vector<const char*> requestedOutputs;
  requestedOutputs.reserve(output.count);
  for (NSString* outputName : output) {
    requestedOutputs.emplace_back([outputName UTF8String]);
  }
  Ort::RunOptions runOptions = [self parseRunOptions:options];
  
  auto result = sessionInfo->session->Run(runOptions,
                                          sessionInfo->inputNames.data(),
                                          feeds.data(),
                                          sessionInfo->inputNames.size(),
                                          requestedOutputs.data(),
                                          requestedOutputs.size());

  NSDictionary* resultMap = [TensorHelper createOutputTensor:requestedOutputs
                                                      values:result];
  
  return resultMap;
}

static NSDictionary* graphOptimizationLevelTable = @{
  @"disabled": @(ORT_DISABLE_ALL),
  @"basic": @(ORT_ENABLE_BASIC),
  @"extended": @(ORT_ENABLE_EXTENDED),
  @"all": @(ORT_ENABLE_ALL)
};

static NSDictionary* executionModeTable = @{
  @"sequential": @(ORT_SEQUENTIAL),
  @"parallel": @(ORT_PARALLEL)
};

- (Ort::SessionOptions)parseSessionOptions:(NSDictionary*)options {
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
    NSString* graphOptimizationLevel = [[options objectForKey:@"graphOptimizationLevel"] stringValue];
    if ([graphOptimizationLevelTable objectForKey:graphOptimizationLevel]) {
      sessionOptions.SetGraphOptimizationLevel((GraphOptimizationLevel)[[graphOptimizationLevelTable objectForKey:graphOptimizationLevel] intValue]);
    }
  }
  
  if ([options objectForKey:@"enableCpuMemArena"]) {
    BOOL enableCpuMemArena = [[options objectForKey:@"enableCpuMemArena"] boolValue];
    if (enableCpuMemArena) {
      sessionOptions.EnableCpuMemArena();
    }
    else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  if ([options objectForKey:@"enableMemPattern"]) {
    BOOL enableMemPattern = [[options objectForKey:@"enableMemPattern"] boolValue];
    if (enableMemPattern) {
      sessionOptions.EnableMemPattern();
    }
    else {
      sessionOptions.DisableMemPattern();
    }
  }
  
  if ([options objectForKey:@"executionMode"]) {
    NSString* executionMode = [[options objectForKey:@"executionMode"] stringValue];
    if ([executionModeTable objectForKey:executionMode]) {
      sessionOptions.SetExecutionMode((ExecutionMode)[[executionModeTable objectForKey:executionMode] intValue]);
    }
  }
  
  if ([options objectForKey:@"logId"]) {
    NSString* logId = [[options objectForKey:@"logId"] stringValue];
    sessionOptions.SetLogId([logId UTF8String]);
  }
  
  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    sessionOptions.SetLogSeverityLevel(logSeverityLevel);
  }

  return sessionOptions;
}

- (Ort::RunOptions)parseRunOptions:(NSDictionary*)options {
  Ort::RunOptions runOptions;
  
  if ([options objectForKey:@"logSeverityLevel"]) {
    int logSeverityLevel = [[options objectForKey:@"logSeverityLevel"] intValue];
    runOptions.SetRunLogSeverityLevel(logSeverityLevel);
  }
  
  if ([options objectForKey:@"tag"]) {
    NSString* tag = [[options objectForKey:@"tag"] stringValue];
    runOptions.SetRunTag([tag UTF8String]);
  }
  
  return runOptions;
}

- (void)dealloc {
  NSEnumerator* iterator = [sessionMap keyEnumerator];
  while (NSString* key = [iterator nextObject]) {
    NSValue* value = [sessionMap objectForKey:key];
    SessionInfo* sessionInfo = (SessionInfo*)[value pointerValue];
    delete sessionInfo;
    sessionInfo = nullptr;
  }
}

@end
