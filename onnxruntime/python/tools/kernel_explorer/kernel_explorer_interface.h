// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <pybind11/pybind11.h>

#include "core/providers/shared_library/provider_api.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/tunable/util.h"
#elif USE_ROCM
#include <hip/hip_runtime.h>
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/tunable/util.h"
#endif

#ifdef USE_CUDA
using onnxruntime::cuda::tunable::Timer;
using ExecutionProvider = onnxruntime::CUDAExecutionProvider;
using ExecutionProviderInfo = onnxruntime::CUDAExecutionProviderInfo;
using StreamT = cudaStream_t;
using TuningContextT = onnxruntime::cuda::tunable::CudaTuningContext;
#elif USE_ROCM
using onnxruntime::rocm::tunable::Timer;
using ExecutionProvider = onnxruntime::ROCMExecutionProvider;
using ExecutionProviderInfo = onnxruntime::ROCMExecutionProviderInfo;
using StreamT = hipStream_t;
using TuningContextT = onnxruntime::rocm::tunable::RocmTuningContext;
#else
#error "kernel explorer only supports CUDA or ROCM"
#endif

namespace onnxruntime {

struct TuningInfo {
  static void EnableCollect(bool b) {
    collect_enabled_ = b;
  }

  static std::vector<TuningResults> GetCollectedTuningResults() {
    return collected_tuning_results_;
  }

  static void SetMaxTuningDurationMs(int milliseconds) {
    max_tuning_duration_ms_ = milliseconds;
  }

  static bool collect_enabled_;
  static std::vector<TuningResults> collected_tuning_results_;
  static std::optional<int> max_tuning_duration_ms_;
};

/// Wrapping around Op and TunableOp
class IKernelExplorer {
 public:
  virtual void Run() = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile() {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run();
    }
    Timer timer{static_cast<Timer::TimerBase::NativeStreamT>(Stream()->GetHandle())};
    timer.Start();
    for (int i = 0; i < repeats_; i++) {
      Run();
    }
    timer.End();
    return timer.Duration() / repeats_;
  }

  virtual ~IKernelExplorer() {
    if (TuningInfo::collect_enabled_) {
      TuningInfo::collected_tuning_results_.emplace_back(this->ep_->GetTuningContext()->GetTuningResults());
    }
  }

 protected:
  ExecutionProvider* GetEp() {
    std::call_once(ep_create_once_, [this]() {
      ExecutionProviderInfo info{};
      this->ep_ = std::make_unique<ExecutionProvider>(info);
      auto allocators = this->ep_->CreatePreferredAllocators();
      for (auto& alloc : allocators) {
        this->allocators_.insert({alloc->Info().device, alloc});
      }
      auto tuning_ctx = this->ep_->GetTuningContext();
      if (nullptr != tuning_ctx) {
        tuning_ctx->RegisterAllocatorsView(&this->allocators_);
        for (const auto& tr : TuningInfo::collected_tuning_results_) {
          auto status = tuning_ctx->LoadTuningResults(tr);
          if (!status.IsOK()) {
            LOGS_DEFAULT(ERROR) << status;
          }
        }
        if (TuningInfo::max_tuning_duration_ms_.has_value()) {
          tuning_ctx->SetMaxTuningDurationMs(*TuningInfo::max_tuning_duration_ms_);
        }
      }
      stream_ = std::make_unique<onnxruntime::Stream>(nullptr, this->ep_->GetOrtDeviceByMemType(OrtMemTypeDefault));
    });
    return ep_.get();
  }

  TuningContextT* TuningContext() {
    return static_cast<TuningContextT*>(GetEp()->GetTuningContext());
  }

  onnxruntime::Stream* Stream() { return stream_.get(); }

 private:
  std::once_flag ep_create_once_;
  std::unique_ptr<ExecutionProvider> ep_{};
  std::map<OrtDevice, AllocatorPtr> allocators_;
  OrtDevice dev_;
  std::unique_ptr<onnxruntime::Stream> stream_;
  int repeats_{100};
};

class WithMaxTuningDurationMs {
 public:
  WithMaxTuningDurationMs(TuningContextT* ctx, int ms) : ctx_(ctx) {
    original_tuning_duration_ = ctx_->GetMaxTuningDurationMs();
    ctx_->SetMaxTuningDurationMs(ms);
  }

  ~WithMaxTuningDurationMs() {
    ctx_->SetMaxTuningDurationMs(original_tuning_duration_);
  }

 private:
  TuningContextT* ctx_;
  int original_tuning_duration_;
};

pybind11::module GetKernelExplorerModule();

class KernelExplorerInit {
 public:
  explicit KernelExplorerInit(void (*init_func)(pybind11::module module)) {
    init_func(GetKernelExplorerModule());
  }
};

#define KE_REGISTER_IMPL(unique_id, module_name)                                    \
  static void KeInitFunc##unique_id(pybind11::module module_name);                  \
  static const KernelExplorerInit kKeInitializer##unique_id{KeInitFunc##unique_id}; \
  void KeInitFunc##unique_id(pybind11::module module_name)

#define KE_REGISTER_(unique_id, module_name) KE_REGISTER_IMPL(unique_id, module_name)
#define KE_REGISTER(module_name) KE_REGISTER_(__COUNTER__, module_name)

}  // namespace onnxruntime
