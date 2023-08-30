// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

struct UnaryElementwisePreparation {
  const Tensor* input_tensor = nullptr;
  Tensor* output_tensor = nullptr;
};

class UnaryElementwise : public CudaKernel {
 protected:
  UnaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, UnaryElementwisePreparation* p) const;
};

template <typename T>
class Abs final : public UnaryElementwise {
 public:
  Abs(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Neg final : public UnaryElementwise {
 public:
  Neg(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Floor final : public UnaryElementwise {
 public:
  Floor(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Ceil final : public UnaryElementwise {
 public:
  Ceil(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public UnaryElementwise {
 public:
  Reciprocal(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sqrt final : public UnaryElementwise {
 public:
  Sqrt(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Log final : public UnaryElementwise {
 public:
  Log(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Exp final : public UnaryElementwise {
 public:
  Exp(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Erf final : public UnaryElementwise {
 public:
  Erf(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Not final : public UnaryElementwise {
 public:
  Not(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Round final : public UnaryElementwise {
 public:
  Round(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sin final : public UnaryElementwise {
 public:
  Sin(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Cos final : public UnaryElementwise {
 public:
  Cos(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sign final : public UnaryElementwise {
 public:
  Sign(const OpKernelInfo& info) : UnaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
