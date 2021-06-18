// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}
class ScatterNDBase {
 protected:
  struct Prepare {
    const uint8_t* input_base;
    const std::string* input_str_base;
    uint8_t* output_base;
    std::string* output_str_base;
    uint64_t bytes_to_copy;
    uint64_t element_bytes;
    uint64_t element_to_copy;
    std::vector<uint64_t> element_offsets;

    Prepare() : input_base(nullptr),
                input_str_base(nullptr),
                output_base(nullptr),
                output_str_base(nullptr),
                bytes_to_copy(0),
                element_bytes(0),
                element_to_copy(0),
                element_offsets(0) {}
  };  // struct Prepare

  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;

 public:
  // Shared between the CPU and CUDA implementation
  static Status ValidateShapes(const TensorShape& input_shape,
                               const TensorShape& indice_shape,
                               const TensorShape& update_shape);
};  // class ScatterNDBase

class ScatterND final : public OpKernel, protected ScatterNDBase {
 public:
  explicit ScatterND(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ScatterNumber(const Prepare& p, concurrency::ThreadPool* tp) const;
  Status ScatterString(const Prepare& p, concurrency::ThreadPool* tp) const;
};

}  // namespace onnxruntime
