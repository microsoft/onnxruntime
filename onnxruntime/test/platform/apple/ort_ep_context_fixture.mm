// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_ep_context_fixture.h"

#import "ort_env_internal.h"
#import "ort_session_internal.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
constexpr const char* kRegistration = "objc_encryption_test";

struct Context {
  std::string name;
  std::vector<uint8_t> bytes;
};

OrtStatus* WriteContext(void* state, const char* name, const void* data, size_t size) noexcept {
  try {
    auto& context = *static_cast<Context*>(state);
    if (!context.name.empty()) {
      throw std::runtime_error("Expected exactly one external context write");
    }
    context.name = name;
    const auto* bytes = static_cast<const uint8_t*>(data);
    context.bytes.assign(bytes, bytes + size);
    return nullptr;
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}
}  // namespace

@implementation ORTEpContextFixture {
  ORTEnv* _env;
}

- (instancetype)initWithEnv:(ORTEnv*)env pluginPath:(NSString*)pluginPath {
  self = [super init];
  if (self) {
    _env = env;
    [_env CXXAPIOrtEnv].RegisterExecutionProviderLibrary(kRegistration, pluginPath.fileSystemRepresentation);
  }
  return self;
}

- (void)configureSessionOptions:(ORTSessionOptions*)options {
  auto& nativeEnv = [_env CXXAPIOrtEnv];
  auto devices = nativeEnv.GetEpDevices();
  auto device = std::find_if(devices.begin(), devices.end(), [](const auto& candidate) {
    return std::strcmp(candidate.EpName(), kRegistration) == 0;
  });
  if (device == devices.end()) {
    throw std::runtime_error("Registered test EP device was not found");
  }
  auto& native = [options CXXAPIOrtSessionOptions];
  native.AddConfigEntry("ep.example.test_execute_ep_context", "1");
  native.AppendExecutionProvider_V2(nativeEnv, {*device}, std::unordered_map<std::string, std::string>{});
}

- (NSData*)compileModelAtPath:(NSString*)source
              outputDirectory:(NSString*)directory
               sessionOptions:(ORTSessionOptions*)options
                  contextData:(NSData**)contextData
                  contextName:(NSString**)contextName {
  auto& nativeEnv = [_env CXXAPIOrtEnv];
  Ort::ModelCompilationOptions compilation(nativeEnv, [options CXXAPIOrtSessionOptions]);
  compilation.SetInputModelPath(source.fileSystemRepresentation);
  compilation.SetEpContextEmbedMode(false);
  compilation.SetEpContextBinaryInformation([[directory stringByAppendingString:@"/"] fileSystemRepresentation],
                                            "compiled.onnx");
  compilation.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
  Context context;
  compilation.SetEpContextDataWriteFunc(WriteContext, &context);
  NSMutableData* model = [NSMutableData data];
  compilation.SetOutputModelWriteFunc(
      [](void* state, const void* data, size_t size) -> OrtStatus* {
        [(__bridge NSMutableData*)state appendBytes:data length:size];
        return nullptr;
      },
      (__bridge void*)model);
  auto status = Ort::CompileModel(nativeEnv, compilation);
  if (!status.IsOK()) {
    throw std::runtime_error(status.GetErrorMessage());
  }
  if (context.name.empty() || context.bytes.empty() || model.length == 0) {
    throw std::runtime_error("Compilation did not emit a model and external EPContext data");
  }
  *contextData = [NSData dataWithBytes:context.bytes.data() length:context.bytes.size()];
  *contextName = [NSString stringWithUTF8String:context.name.c_str()];
  return model;
}

- (void)unregister {
  [_env CXXAPIOrtEnv].UnregisterExecutionProviderLibrary(kRegistration);
}
@end
