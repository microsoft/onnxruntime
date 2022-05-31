#pragma once
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
using CudaStreamHandle = cudaStream_t;


struct CudaStream : Stream {
  CudaStream(cudaStream_t stream, const IExecutionProvider* ep);

  ~CudaStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;
};

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry);
}
