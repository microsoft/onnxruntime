#pragma once
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
using CudaStreamHandle = cudaStream_t;


struct CudaStream : Stream {
  cudnnHandle_t cudnn_handle_{};
  cublasHandle_t cublas_handle_{};

  CudaStream(cudaStream_t stream, const IExecutionProvider* ep, bool own_flag);

  ~CudaStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  bool own_stream_{true};
};

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry, const std::string& ep_type, cudaStream_t external_stream, bool use_existing_stream);
void WaitCudaNotificationOnDevice(Stream& stream, synchronize::Notification& notification);
}
