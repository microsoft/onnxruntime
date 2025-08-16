// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OnnxruntimeModule.h"
#include "JsiMain.h"

#import <Foundation/Foundation.h>
#import <React/RCTBridge+Private.h>
#import <React/RCTLog.h>

@implementation OnnxruntimeModule

static std::shared_ptr<onnxruntimejsi::Env> env;

RCT_EXPORT_MODULE(Onnxruntime)

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
    auto jsiInvoker = std::make_shared<facebook::react::CallInvoker>(
        [cxxBridge.jsInvoker getModuleRegistry()]);

    env = onnxruntimejsi::install(runtime, jsiInvoker);

    return @true;
  } @catch (...) {
    return @false;
  }
}

- (void)dealloc {
  env.reset();
}

@end
