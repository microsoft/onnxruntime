// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include <atomic>
#include <cassert>
#include <omp.h>
#include <math.h>
#include "dnnl_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

struct DnnlProviderFactory : IExecutionProviderFactory {
  DnnlProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~DnnlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> DnnlProviderFactory::CreateProvider() {
  DNNLExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<DNNLExecutionProvider>(info);
}

struct Dnnl_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* options) override {
#if defined(_WIN32)
    {
      // We crash when unloading DNNL on Windows when OpenMP also unloads (As there are threads
      // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
      // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
      HMODULE handle{};
#ifdef _DEBUG
      constexpr const char* dll_name = "vcomp140d.dll";
#else
      constexpr const char* dll_name = "vcomp140.dll";
#endif
      ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
      assert(handle);  // It should exist
    }
#endif
    // Cast options
    const OrtDnnlProviderOptions* dnnl_options = reinterpret_cast<const OrtDnnlProviderOptions*>(options);

    if (dnnl_options->optimize_threads) {
      // Set up threads for optimal performance
      int num_threads = dnnl_options->onednn_threads ? dnnl_options->onednn_threads : omp_get_max_threads();
      int ort_num_threads = dnnl_options->ort_threads;
      if (!ort_num_threads) {
        // Avoid over subscription
        ort_num_threads = static_cast<int>(ceil(num_threads / 4.0f));
        num_threads -= ort_num_threads;
      }

      // Check for nullptr to avoid undefined behavior
      if (dnnl_options->ort_intra_op_threads != nullptr) {
        // Override default value
        *dnnl_options->ort_intra_op_threads = ort_num_threads;
      }

      // Set number of threads
      omp_set_num_threads(num_threads);
      fprintf(stdout, "Setting OMP to %d threads\n", num_threads);
    }

    return std::make_shared<DnnlProviderFactory>(dnnl_options->use_arena != 0);
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
