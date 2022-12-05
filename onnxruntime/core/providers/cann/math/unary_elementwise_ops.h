// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

class UnaryElementwise : public CannKernel {
 protected:
  explicit UnaryElementwise(const OpKernelInfo& info) : CannKernel(info) {
    op_name_ = Info().GetKernelDef().OpName();
  }
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  template <typename T>
  Status Prepare(OpKernelContext* ctx, CannPreparation& prepare) const;

 private:
  std::string op_name_;
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

}  // namespace cann
}  // namespace onnxruntime
