// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/webgpu/internal_test_kernel.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/framework/op_kernel.h"

#if defined(ORT_USE_EP_API_ADAPTERS)
#include "test/providers/webgpu/internal_test_api.h"
#else
#include "core/framework/data_transfer_manager.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#endif

namespace onnxruntime::test {
namespace {
class UnusedKernel final : public WebGpuTestOpKernel {
 public:
  explicit UnusedKernel(const WebGpuTestOpKernelInfo& info) : WebGpuTestOpKernel(info) {}
  Status Compute(WebGpuTestOpKernelContext*) const override { return Status::OK(); }
};
}  // namespace

#if defined(ORT_USE_EP_API_ADAPTERS)
extern const WebGpuInternalTestHostApi* webgpu_test_host_api;
extern const OrtLogger* webgpu_test_logger;
namespace {
class TestEp final : public ep::adapter::Ep {
 public:
  explicit TestEp(std::unique_ptr<IExecutionProvider> provider)
      : Ep(std::move(provider), CPUAllocator::DefaultInstance(), nullptr) {}
};
}  // namespace
struct WebGpuTestKernel::Impl {
  explicit Impl(const ConfigOptions& options)
      : provider(WebGpuProviderFactoryCreator::Create(options)->CreateProvider()) {
    ORT_ENFORCE(webgpu_test_host_api != nullptr);
    info = std::make_unique<WebGpuTestOpKernelInfo>(webgpu_test_host_api->create_kernel_info(&provider, &host_info));
    kernel = std::make_unique<UnusedKernel>(*info);
    Ep().SetLogger(&logging::LoggingManager::DefaultLogger());
    Ep().SetEpLogger(webgpu_test_logger);
  }
  ~Impl() {
    kernel.reset();
    info.reset();
    webgpu_test_host_api->release_kernel_info(host_info);
  }
  TestEp provider;
  void* host_info{};
  std::unique_ptr<WebGpuTestOpKernelInfo> info;
  std::unique_ptr<UnusedKernel> kernel;
  WebGpuExecutionProvider& Ep() { return *static_cast<WebGpuExecutionProvider*>(provider.EpImpl()); }
};
#else
struct WebGpuTestKernel::Impl {
  explicit Impl(const ConfigOptions& options)
      : config_options(options), provider(WebGpuProviderFactoryCreator::Create(config_options)->CreateProvider()), model("webgpu_internal_test", false, DefaultLoggingManager().DefaultLogger()), node(model.MainGraph().AddNode("unused", "Identity", "", {}, {})), definition(KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build()), info(node, *definition, *provider, initializers, values, transfers, allocators, config_options), kernel(std::make_unique<UnusedKernel>(info)) {
    provider->SetLogger(&DefaultLoggingManager().DefaultLogger());
  }
  ConfigOptions config_options;
  std::unique_ptr<IExecutionProvider> provider;
  Model model;
  Node& node;
  std::unique_ptr<KernelDef> definition;
  std::unordered_map<int, OrtValue> initializers;
  OrtValueNameIdxMap values;
  DataTransferManager transfers;
  AllocatorMap allocators;
  OpKernelInfo info;
  std::unique_ptr<UnusedKernel> kernel;
  WebGpuExecutionProvider& Ep() { return *static_cast<WebGpuExecutionProvider*>(provider.get()); }
};
#endif

WebGpuTestKernel::WebGpuTestKernel(const ConfigOptions& options) : impl_(std::make_unique<Impl>(options)) {}
WebGpuTestKernel::~WebGpuTestKernel() = default;
WebGpuExecutionProvider& WebGpuTestKernel::Ep() { return impl_->Ep(); }
const WebGpuTestOpKernel& WebGpuTestKernel::Kernel() { return *impl_->kernel; }
#if defined(ORT_USE_EP_API_ADAPTERS)
OrtKernelContext* WebGpuTestKernel::Context() { return webgpu_test_host_api->kernel_context(impl_->host_info); }
#endif
}  // namespace onnxruntime::test
