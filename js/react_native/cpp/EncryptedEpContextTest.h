// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ORT_RN_TEST_EP_CONTEXT
#include <algorithm>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "Env.h"

namespace onnxruntimejsi {

inline Ort::SessionOptions encryptionTestOptions(Ort::Env& env, const std::string& registration) {
  auto devices = env.GetEpDevices();
  auto device = std::find_if(devices.begin(), devices.end(), [&](const auto& candidate) {
    return registration == candidate.EpName();
  });
  if (device == devices.end()) {
    throw std::runtime_error("Registered encryption test EP device not found");
  }
  Ort::SessionOptions options;
  options.AddConfigEntry("ep.example.test_execute_ep_context", "1");
  options.AppendExecutionProvider_V2(env, {*device}, std::unordered_map<std::string, std::string>{});
  return options;
}

inline facebook::jsi::Value compileEncryptionTestModel(
    facebook::jsi::Runtime& runtime, Ort::Env& env,
    const facebook::jsi::Value* arguments, size_t count) {
  using namespace facebook::jsi;
  if (count != 4 || !arguments[0].isString() || !arguments[1].isString() ||
      !arguments[2].isString() || !arguments[3].isString()) {
    throw JSError(runtime, "Expected plugin path, registration, source model path, and existing output directory");
  }
  const std::filesystem::path library(arguments[0].asString(runtime).utf8(runtime));
  const auto registration = arguments[1].asString(runtime).utf8(runtime);
  const std::filesystem::path input(arguments[2].asString(runtime).utf8(runtime));
  std::filesystem::path directory(arguments[3].asString(runtime).utf8(runtime));
  directory += std::filesystem::path::preferred_separator;
  try {
    env.RegisterExecutionProviderLibrary(registration.c_str(), library.c_str());
    try {
      auto options = encryptionTestOptions(env, registration);
      Ort::ModelCompilationOptions compile(env, options);
      compile.SetInputModelPath(input.c_str());
      compile.SetEpContextEmbedMode(false);
      const std::filesystem::path modelName("compiled.onnx");
      compile.SetEpContextBinaryInformation(directory.c_str(), modelName.c_str());
      compile.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
      struct Context {
        std::string name;
        std::vector<uint8_t> bytes;
      } context;
      compile.SetEpContextDataWriteFunc(
          [](void* state, const char* name, const void* data, size_t size) -> OrtStatus* {
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
          },
          &context);
      std::vector<uint8_t> model;
      compile.SetOutputModelWriteFunc(
          [](void* state, const void* data, size_t size) -> OrtStatus* {
            try {
              auto& bytes = *static_cast<std::vector<uint8_t>*>(state);
              const auto* begin = static_cast<const uint8_t*>(data);
              bytes.insert(bytes.end(), begin, begin + size);
              return nullptr;
            } catch (const std::exception& error) {
              return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
            }
          },
          &model);
      auto status = Ort::CompileModel(env, compile);
      if (!status.IsOK()) {
        throw std::runtime_error(status.GetErrorMessage());
      }
      if (context.name.empty() ||
          std::string(context.bytes.begin(), context.bytes.end()) != "ort-test-mul-float32-v1") {
        throw std::runtime_error("Compilation did not emit the executable test context");
      }
      auto toBytes = [&](const std::vector<uint8_t>& bytes) {
        Array values(runtime, bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
          values.setValueAtIndex(runtime, i, static_cast<double>(bytes[i]));
        }
        return runtime.global().getPropertyAsFunction(runtime, "Uint8Array").callAsConstructor(runtime, values).asObject(runtime);
      };
      Object result(runtime);
      result.setProperty(runtime, "model", toBytes(model));
      result.setProperty(runtime, "context", toBytes(context.bytes));
      result.setProperty(runtime, "contextName", String::createFromUtf8(runtime, context.name));
      std::random_device random;
      std::uniform_int_distribution<unsigned int> distribution(0, 255);
      std::vector<uint8_t> key(32);
      for (auto& byte : key) {
        byte = static_cast<uint8_t>(distribution(random));
      }
      result.setProperty(runtime, "key", toBytes(key));
      return result;
    } catch (...) {
      env.UnregisterExecutionProviderLibrary(registration.c_str());
      throw;
    }
  } catch (const std::exception& error) {
    throw JSError(runtime, error.what());
  }
}
}  // namespace onnxruntimejsi
#endif
