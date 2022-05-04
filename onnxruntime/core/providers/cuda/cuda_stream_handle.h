#pragma once
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
using CudaStreamHandle = cudaStream_t;

void RegisterCudaStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry);
}
