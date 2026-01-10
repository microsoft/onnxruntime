// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
class GatherNDBase {
 public:
  struct Prepare {
    const uint8_t* input_base;
    const std::string* input_str_base;
    uint8_t* output_base;
    std::string* output_str_base;
    uint64_t bytes_per_slice;
    uint64_t element_bytes;
    uint64_t element_count_per_slice;
    std::vector<uint64_t> slice_offsets;

    Prepare() : input_base(nullptr),
                input_str_base(nullptr),
                output_base(nullptr),
                output_str_base(nullptr),
                bytes_per_slice(0),
                element_bytes(0),
                element_count_per_slice(0),
                slice_offsets(0) {}
  };  // struct Prepare

  template <typename Tind>
  Status PrepareForCompute(const TensorShape& input_shape, const Tensor* indices_tensor,
                           const int64_t bytes_per_value, Prepare& p, concurrency::ThreadPool* tp) const;
  int64_t batch_dims_;
};  // class GatherNDBase

class GatherND final : public OpKernel, protected GatherNDBase {
 public:
  explicit GatherND(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault("batch_dims", &batch_dims_, static_cast<int64_t>(0));
  }
  Status Compute(OpKernelContext* context) const override;

 private:
  Status GatherNumber(const Prepare& p, concurrency::ThreadPool* tp) const;
  Status GatherString(const Prepare& p, concurrency::ThreadPool* tp) const;
};

}  // namespace onnxruntime
