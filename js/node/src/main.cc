// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <napi.h>

#include "ep_context_data_read_helper.h"
#include "inference_session_wrap.h"

#ifdef ORT_NODEJS_TEST_EP_CONTEXT
#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "ort_singleton_data.h"

namespace {
Napi::Value CompileTestEpContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (info.Length() != 3 || !info[0].IsString() || !info[1].IsString() || !info[2].IsBuffer()) {
    throw Napi::TypeError::New(env, "Expected plugin path, registration name, and input model buffer.");
  }
  auto* objects = OrtSingletonData::GetOrtObjects();
  if (!objects) {
    throw Napi::Error::New(env, "ONNX Runtime must be initialized before compiling the fixture.");
  }
  auto registration = info[1].As<Napi::String>().Utf8Value();
  try {
    const std::filesystem::path library(info[0].As<Napi::String>().Utf16Value());
    objects->env.RegisterExecutionProviderLibrary(registration.c_str(), library.c_str());
    try {
      auto devices = objects->env.GetEpDevices();
      auto device = std::find_if(devices.begin(), devices.end(), [&](const auto& candidate) {
        return registration == candidate.EpName();
      });
      if (device == devices.end()) {
        throw std::runtime_error("Registered example EP device was not found.");
      }
      Ort::SessionOptions options;
      options.AddConfigEntry("ep.example.test_execute_ep_context", "1");
      options.AppendExecutionProvider_V2(objects->env, {*device}, {});
      Ort::ModelCompilationOptions compile(objects->env, options);
      auto input = info[2].As<Napi::Buffer<uint8_t>>();
      compile.SetInputModelFromBuffer(input.Data(), input.Length());
      compile.SetEpContextEmbedMode(false);
      auto directory = std::filesystem::current_path();
      directory += std::filesystem::path::preferred_separator;
      const std::filesystem::path modelName("compiled.onnx");
      compile.SetEpContextBinaryInformation(directory.c_str(), modelName.c_str());
      compile.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
      struct Context {
        std::string name;
        std::vector<uint8_t> bytes;
      } context;
      compile.SetEpContextDataWriteFunc(
          [](void* state, const char* name, const void* data, size_t size) -> OrtStatus* {
            auto& context = *static_cast<Context*>(state);
            try {
              if (!context.name.empty()) {
                return Ort::GetApi().CreateStatus(ORT_FAIL, "Expected exactly one context write.");
              }
              context.name = name;
              const auto* bytes = static_cast<const uint8_t*>(data);
              context.bytes.assign(bytes, bytes + size);
              return nullptr;
            } catch (const std::exception& error) {
              return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
            }
          },
          &context);
      Ort::AllocatorWithDefaultOptions allocator;
      void* model = nullptr;
      size_t size = 0;
      compile.SetOutputModelBuffer(allocator, &model, &size);
      try {
        auto status = Ort::CompileModel(objects->env, compile);
        if (!status.IsOK()) {
          throw std::runtime_error(status.GetErrorMessage());
        }
        if (context.name.empty() || context.bytes.empty()) {
          throw std::runtime_error("Compilation did not emit external EPContext data.");
        }
        auto result = Napi::Object::New(env);
        result.Set("model", Napi::Buffer<uint8_t>::Copy(env, static_cast<const uint8_t*>(model), size));
        result.Set("context", Napi::Buffer<uint8_t>::Copy(env, context.bytes.data(), context.bytes.size()));
        result.Set("contextName", context.name);
        allocator.Free(model);
        return result;
      } catch (...) {
        if (model) {
          allocator.Free(model);
        }
        throw;
      }
    } catch (...) {
      objects->env.UnregisterExecutionProviderLibrary(registration.c_str());
      throw;
    }
  } catch (const std::exception& error) {
    throw Napi::Error::New(env, error.what());
  }
}

Napi::Value UnregisterTestEpContext(const Napi::CallbackInfo& info) {
  if (info.Length() != 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(info.Env(), "Expected an EP registration name.");
  }
  try {
    OrtSingletonData::GetOrtObjects()->env.UnregisterExecutionProviderLibrary(
        info[0].As<Napi::String>().Utf8Value().c_str());
  } catch (const std::exception& error) {
    throw Napi::Error::New(info.Env(), error.what());
  }
  return info.Env().Undefined();
}
}  // namespace
#endif

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InferenceSessionWrap::Init(env, exports);
  exports.Set("__testEpContextDataReadCallback",
              Napi::Function::New(env, TestEpContextDataReadCallback));
#ifdef ORT_NODEJS_TEST_EP_CONTEXT
  exports.Set("__testCompileEpContextModel", Napi::Function::New(env, CompileTestEpContext));
  exports.Set("__testUnregisterEpContextPlugin", Napi::Function::New(env, UnregisterTestEpContext));
#endif
  return exports;
}

NODE_API_MODULE(onnxruntime, InitAll)
