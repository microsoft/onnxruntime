// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#include "JsiMain.h"

#import <Foundation/Foundation.h>
#import <React/RCTBridge+Private.h>
#import <React/RCTUtils.h>
#import <ReactCommon/RCTTurboModule.h>
#import <jsi/jsi.h>
#include <mutex>
#include <unordered_map>

namespace {

std::mutex envsMutex;
std::unordered_map<const void*, std::shared_ptr<onnxruntimejsi::Env>>
    envs;

}  // namespace

@implementation OnnxruntimeModule

@synthesize bridge = _bridge;

RCT_EXPORT_MODULE(Onnxruntime)

- (void)setBridge:(RCTBridge*)bridge {
  _bridge = bridge;
}

/**
 * React native binding API to install onnxruntime JSI API
 */
RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(install) {
  @try {
    RCTCxxBridge* cxxBridge = (RCTCxxBridge*)_bridge;
    if (cxxBridge == nil) {
      return @false;
    }

    auto jsiRuntime = (facebook::jsi::Runtime*)cxxBridge.runtime;
    if (jsiRuntime == nil) {
      return @false;
    }
    auto& runtime = *jsiRuntime;
    auto jsiInvoker = _bridge.jsCallInvoker;

    auto newEnv = onnxruntimejsi::install(runtime, jsiInvoker);
    std::shared_ptr<onnxruntimejsi::Env> oldEnv;
    {
      std::lock_guard<std::mutex> lock(envsMutex);
      const auto moduleKey = (__bridge const void*)self;
      oldEnv = std::move(envs[moduleKey]);
      envs[moduleKey] = std::move(newEnv);
    }
    if (oldEnv) {
      oldEnv->invalidate();
    }

    return @true;
  } @catch (...) {
    return @false;
  }
}

- (void)dealloc {
  std::shared_ptr<onnxruntimejsi::Env> moduleEnv;
  {
    std::lock_guard<std::mutex> lock(envsMutex);
    auto entry = envs.find((__bridge const void*)self);
    if (entry != envs.end()) {
      moduleEnv = std::move(entry->second);
      envs.erase(entry);
    }
  }
  // dealloc is not guaranteed to run on the JS thread, so stop dispatching to the runtime and
  // release any thread blocked on it before dropping the environment.
  if (moduleEnv) {
    moduleEnv->invalidate();
  }
}

@end
