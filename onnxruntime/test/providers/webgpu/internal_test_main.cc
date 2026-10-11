// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <filesystem>

#include <gsl/gsl>
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "onnxruntime_cxx_api.h"
#include "test/providers/webgpu/internal_test_api.h"
#include "test/util/include/telemetry_test_environment.h"

namespace {
class EmptyKernelContext final : public onnxruntime::OpKernelContext {
 public:
  EmptyKernelContext() : OpKernelContext(nullptr, onnxruntime::logging::LoggingManager::DefaultLogger(), nullptr) {}
  int InputCount() const override { return 0; }
  int OutputCount() const override { return 0; }
};
class KernelInfoProvider final : public onnxruntime::IExecutionProvider {
 public:
  explicit KernelInfoProvider(OrtEp* ep) : IExecutionProvider("WebGpuExecutionProvider"), ep_(ep) {}
  const OrtEp* GetOrtEp() const override { return ep_; }

 private:
  OrtEp* ep_;
};

struct KernelInfoHolder {
  explicit KernelInfoHolder(OrtEp* ep)
      : provider(ep), model("webgpu_internal_test", false, onnxruntime::logging::LoggingManager::DefaultLogger()), node(model.MainGraph().AddNode("test", "Identity", "", {}, {})), definition(onnxruntime::KernelDefBuilder().SetName("Identity").Provider("WebGpuExecutionProvider").SinceVersion(1).Build()), info(node, *definition, provider, initializers, values, transfers, allocators, options) {}
  KernelInfoProvider provider;
  onnxruntime::Model model;
  onnxruntime::Node& node;
  std::unique_ptr<onnxruntime::KernelDef> definition;
  std::unordered_map<int, OrtValue> initializers;
  onnxruntime::OrtValueNameIdxMap values;
  onnxruntime::DataTransferManager transfers;
  onnxruntime::AllocatorMap allocators;
  onnxruntime::ConfigOptions options;
  onnxruntime::OpKernelInfo info;
  EmptyKernelContext context;
};
}  // namespace

int main(int argc, char** argv) {
  int result = 1;
  ORT_TRY {
    onnxruntime::test::SuppressTelemetryForTests();
    Ort::Env runtime_env(ORT_LOGGING_LEVEL_WARNING, "WebGpuInternalTests");
    const auto& platform_env = onnxruntime::Env::Default();
    void* library = nullptr;
    ORT_THROW_IF_ERROR(platform_env.LoadDynamicLibrary(
        (std::filesystem::path{platform_env.GetRuntimePath()} /
         onnxruntime::ToPathString(ORT_WEBGPU_INTERNAL_TEST_LIBRARY_PATH))
            .native(),
        false, &library));
    auto unload_library = gsl::finally([&]() {
      const auto status = platform_env.UnloadDynamicLibrary(library);
      if (!status.IsOK()) {
        std::cerr << status << '\n';
        result = 1;
      }
    });

    void* entry_point = nullptr;
    ORT_THROW_IF_ERROR(platform_env.GetSymbolFromLibrary(library, "RunWebGpuInternalTests", &entry_point));
    const auto run_tests = reinterpret_cast<onnxruntime::test::RunWebGpuInternalTestsFn>(entry_point);
    const onnxruntime::test::WebGpuInternalTestHostApi host_api{
        [](OrtEp* ep, void** holder) -> const OrtKernelInfo* {
          auto value = std::make_unique<KernelInfoHolder>(ep);
          const auto* info = reinterpret_cast<const OrtKernelInfo*>(&value->info);
          *holder = value.release();
          return info;
        },
        [](void* holder) { delete static_cast<KernelInfoHolder*>(holder); },
        [](void* holder) -> OrtKernelContext* {
          return reinterpret_cast<OrtKernelContext*>(&static_cast<KernelInfoHolder*>(holder)->context);
        }};
    result = run_tests(OrtGetApiBase(), onnxruntime::logging::LoggingManager::DefaultLogger().ToExternal(), &host_api, argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { std::cerr << ex.what() << '\n'; });
  }

  google::protobuf::ShutdownProtobufLibrary();
  return result;
}
