// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/dnnl/dnnl_provider_factory.h"

#include <atomic>
#include <cassert>

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/dnnl/dnnl_provider_factory_creator.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/dnnl_execution_provider_info.h"

using namespace onnxruntime;

namespace onnxruntime {

struct DnnlProviderFactory : IExecutionProviderFactory {
  DnnlProviderFactory(const DnnlExecutionProviderInfo& info) : info_(info) {}
  ~DnnlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  DnnlExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  return std::make_unique<DnnlExecutionProvider>(info_);
}

struct ProviderInfo_Dnnl_Impl : ProviderInfo_Dnnl {
  void DnnlExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options,
                                                      DnnlExecutionProviderInfo& info) override {
    info = DnnlExecutionProviderInfo::FromProviderOptions(options);
  }

  std::shared_ptr<IExecutionProviderFactory>
  CreateExecutionProviderFactory(const DnnlExecutionProviderInfo& info) override {
    return std::make_shared<DnnlProviderFactory>(info);
  }
} g_info;

struct Dnnl_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int use_arena) override {
#if defined(DNNL_OPENMP)
    LoadOpenMP();
#endif  // defined(DNNL_OPENMP) && defined(_WIN32)

    // Map options to provider info
    DnnlExecutionProviderInfo info{};
    info.use_arena = use_arena;
    return std::make_shared<DnnlProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* options) override {
#if defined(DNNL_OPENMP)
    LoadOpenMP();
#endif  // defined(DNNL_OPENMP) && defined(_WIN32)
    // Cast options
    auto dnnl_options = reinterpret_cast<const OrtDnnlProviderOptions*>(options);

    // Map options to provider info
    DnnlExecutionProviderInfo info{};
    info.use_arena = dnnl_options->use_arena;
    info.threadpool_args = dnnl_options->threadpool_args;

    return std::make_shared<DnnlProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::DnnlExecutionProviderInfo::FromProviderOptions(options);
    auto& dnnl_options = *reinterpret_cast<OrtDnnlProviderOptions*>(provider_options);

    dnnl_options.use_arena = internal_options.use_arena;
    dnnl_options.threadpool_args = internal_options.threadpool_args;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtDnnlProviderOptions*>(provider_options);
    return DnnlExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

  void* GetInfo() override {
    return &g_info;
  }

 private:
  void LoadOpenMP() {
#if defined(_WIN32)
    // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
    // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
    // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
    HMODULE handle{};
#ifdef _DEBUG
    constexpr const char* dll_name = "vcomp140d.dll";
#else
    constexpr const char* dll_name = "vcomp140.dll";
#endif  // _DEBUG
    ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
    assert(handle);  // It should exist
#endif               // defined(_WIN32)
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
