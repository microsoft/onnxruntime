// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {

// CUDA implementation of LabelEncoder for opset 2-3.
// Supports numeric key/value types only (int64_t, float).
// Uses sorted arrays + binary search on GPU for O(log n) per-element lookup.
template <typename TKey, typename TValue>
class CudaLabelEncoder : public CudaKernel {
 public:
  CudaLabelEncoder(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void InitializeSomeFields(const OpKernelInfo& info);

  IAllocatorUniquePtr<TKey> keys_gpu_;
  IAllocatorUniquePtr<TValue> values_gpu_;
  int64_t num_keys_;
  TValue default_value_;
  int64_t nan_key_index_;  // -1 if no NaN key

  std::string key_field_name_;
  std::string value_field_name_;
};

// CUDA implementation of LabelEncoder for opset 4+.
// Supports numeric key/value types only (int64_t, float, double).
// Uses sorted arrays + binary search on GPU for O(log n) per-element lookup.
template <typename TKey, typename TValue>
class CudaLabelEncoder_4 : public CudaKernel {
 public:
  CudaLabelEncoder_4(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void InitializeAttrFields(const OpKernelInfo& info);

  IAllocatorUniquePtr<TKey> keys_gpu_;
  IAllocatorUniquePtr<TValue> values_gpu_;
  int64_t num_keys_;
  TValue default_value_;
  int64_t nan_key_index_;  // -1 if no NaN key

  std::string key_field_name_;
  std::string value_field_name_;
};

}  // namespace cuda
}  // namespace onnxruntime
