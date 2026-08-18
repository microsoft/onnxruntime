// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

template <typename T>
inline void ComputeAttentionSoftmaxInplace(T* score, size_t N, size_t D,
                                           concurrency::ThreadPool* tp, AllocatorPtr) {
  MlasComputeSoftmax(score, score, N, D, false, false, 0.0f, tp);
}

template <>
inline void ComputeAttentionSoftmaxInplace<MLFloat16>(MLFloat16* score, size_t N, size_t D,
                                                      concurrency::ThreadPool* tp, AllocatorPtr allocator) {
  ORT_ENFORCE(tp == nullptr, "No parallelized version of softmax for float16.");
  // MLAS lacks kernels for fp16 softmax, so we convert to float32 and use the float32 version.
  auto num_elements = SafeInt<size_t>(N) * D;
  void* allocated_ptr = allocator->Alloc(num_elements * sizeof(float));
  BufferUniquePtr float_buffer(allocated_ptr, BufferDeleter(allocator));
  float* ptr = reinterpret_cast<float*>(allocated_ptr);
  MlasConvertHalfToFloatBuffer(score, ptr, num_elements);
  MlasComputeSoftmax(ptr, ptr, N, D, false, false, 0.0f, tp);
  MlasConvertFloatToHalfBuffer(ptr, score, num_elements);
}

}  // namespace onnxruntime
