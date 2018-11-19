// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public OpKernel {
 public:
  Sub(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public OpKernel {
 public:
  Mul(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public OpKernel {
 public:
  Div(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Abs final : public OpKernel {
 public:
  Abs(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& input = *context->Input<Tensor>(0);
    auto& output = *context->Output(0, input.Shape());

    EigenMap<T>(output) = EigenMap<T>(input).cwiseAbs();
    return Status::OK();
  }
};

template <typename T>
class Neg final : public OpKernel {
 public:
  Neg(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto& input = *context->Input<Tensor>(0);
    auto& output = *context->Output(0, input.Shape());

    EigenMap<T>(output) = -EigenMap<T>(input);
    return Status::OK();
  }
};

template <typename T>
class Floor final : public OpKernel {
 public:
  Floor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Ceil final : public OpKernel {
 public:
  Ceil(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Reciprocal final : public OpKernel {
 public:
  Reciprocal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sqrt final : public OpKernel {
 public:
  Sqrt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public OpKernel {
 public:
  Pow(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Exp final : public OpKernel {
 public:
  Exp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Log final : public OpKernel {
 public:
  Log(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_6 final : public OpKernel {
 public:
  Sum_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Sum_8 final : public OpKernel {
 public:
  Sum_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min_6 final : public OpKernel {
 public:
  Min_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Min_8 final : public OpKernel {
 public:
  Min_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Max_6 final : public OpKernel {
 public:
  Max_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Max_8 final : public OpKernel {
 public:
  Max_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Not final : public OpKernel {
 public:
  Not(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class And final : public OpKernel {
 public:
  And(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Or final : public OpKernel {
 public:
  Or(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class Xor final : public OpKernel {
 public:
  Xor(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public OpKernel {
 public:
  Equal(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public OpKernel {
 public:
  Less(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Greater final : public OpKernel {
 public:
  Greater(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_6 final : public OpKernel {
 public:
  Mean_6(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Mean_8 final : public OpKernel {
 public:
  Mean_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Affine final : public OpKernel {
 public:
  Affine(const OpKernelInfo& info) : OpKernel(info) {
    // Either model-supplied or default values should be returned for alpha and beta
    ONNXRUNTIME_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ONNXRUNTIME_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public OpKernel {
 public:
  PRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Expand_8 final : public OpKernel {
 public:
  Expand_8(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class Scale final : public OpKernel {
 public:
  Scale(const OpKernelInfo& info) : OpKernel(info) {
    ONNXRUNTIME_ENFORCE(info.GetAttr("scale", &scale_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float scale_;
};

}  // namespace onnxruntime
