// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/dnnl/dnnl_provider_factory.h"

#include <atomic>
#include <cassert>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/dnnl_provider_factory_creator.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct DnnlProviderFactory : IExecutionProviderFactory {
  DnnlProviderFactory(bool create_arena,
                      void* threadpool_args)
      : create_arena_(create_arena),
        threadpool_args_(threadpool_args) {}

  ~DnnlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
  void* threadpool_args_;
};

std::unique_ptr<IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  DNNLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  info.threadpool_args = threadpool_args_;
  return std::make_unique<DNNLExecutionProvider>(info);
}

struct Dnnl_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* options) override {
#if defined(DNNL_OPENMP) && defined(_WIN32)
    {
      // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
      // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
      // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
      HMODULE handle{};
#ifdef _DEBUG
      constexpr const char* dll_name = "vcomp140d.dll";
#else
      constexpr const char* dll_name = "vcomp140.dll";
#endif // _DEBUG
      ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
      assert(handle);  // It should exist
    }
#endif // defined(DNNL_OPENMP) && defined(_WIN32)

    // Cast options
    const OrtDnnlProviderOptions* dnnl_options = reinterpret_cast<const OrtDnnlProviderOptions*>(options);

    return std::make_shared<DnnlProviderFactory>(dnnl_options->use_arena != 0,
                                                 dnnl_options->threadpool_args);
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
