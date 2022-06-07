// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/random_generator.h"
#include "core/providers/cuda/cuda_kernel.h"
#include <optional>

namespace onnxruntime {
namespace cuda {

class RandomBase {
 protected:
  explicit RandomBase(const OpKernelInfo& info) {
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_.emplace(static_cast<uint64_t>(seed));
    }

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(gsl::narrow<int>(dtype)) &&
                      dtype != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                  "Invalid dtype of ", dtype);
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
    }
  }

 protected:

  void SetDTypeIfUndefined(ONNX_NAMESPACE::TensorProto::DataType dtype) noexcept {
    if (dtype_ == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      dtype_ = dtype;
    }
  }

  ONNX_NAMESPACE::TensorProto::DataType GetDType() const noexcept { return dtype_; }

  PhiloxGenerator& GetPhiloxGenerator() const {
    return (generator_.has_value()) ? *generator_ : PhiloxGenerator::Default();
  }

 private:

  ONNX_NAMESPACE::TensorProto::DataType dtype_ =
      ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;  // optional and may be inferred

  // This member is thread-safe, ensuring proper synchronization
  mutable std::optional<PhiloxGenerator> generator_;
};

class RandomNormalBase : public RandomBase {
 protected:
  RandomNormalBase(const OpKernelInfo& info) : RandomBase(info) {
    ORT_THROW_IF_ERROR(info.GetAttr<float>("scale", &scale_));
    ORT_THROW_IF_ERROR(info.GetAttr<float>("mean", &mean_));
  }

  Status ComputeNormal(const CudaKernel& cuda_kernel, OpKernelContext& ctx, const TensorShape& shape, int dtype) const;

 private:
  float scale_;
  float mean_;
};

class RandomNormal final : public CudaKernel, protected RandomNormalBase {
 public:
  explicit RandomNormal(const OpKernelInfo& info) : CudaKernel(info), RandomNormalBase(info) {
    SetDTypeIfUndefined(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    std::vector<int64_t> shape;
    ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("shape", shape));
    shape_ = TensorShape(shape);
  }

  Status ComputeInternal(OpKernelContext* p_ctx) const override {
    return ComputeNormal(*this, *p_ctx, shape_, GetDType());
  }

 private:
  TensorShape shape_;
};

class RandomNormalLike final : public CudaKernel, protected RandomNormalBase {
 public:
  explicit RandomNormalLike(const OpKernelInfo& info) : CudaKernel(info), RandomNormalBase(info) {}
  Status ComputeInternal(OpKernelContext* p_ctx) const override;
};

class RandomUniformBase : public RandomBase {
 protected:
  explicit RandomUniformBase(const OpKernelInfo& info) : RandomBase(info) {
    float low, high;
    ORT_THROW_IF_ERROR(info.GetAttr<float>("low", &low));
    ORT_THROW_IF_ERROR(info.GetAttr<float>("high", &high));
    from_ = low;
    range_ = high - low;
  }

  Status ComputeUniform(const CudaKernel& cuda_kernel, OpKernelContext& ctx, const TensorShape& shape, int dtype) const;

 private:
  float range_;
  float from_;
};

class RandomUniform final : public CudaKernel, protected RandomUniformBase {
 public:
  explicit RandomUniform(const OpKernelInfo& info) : CudaKernel(info), RandomUniformBase(info) {
    SetDTypeIfUndefined(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    std::vector<int64_t> shape;
    ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("shape", shape));
    shape_ = TensorShape(shape);
  }

  Status ComputeInternal(OpKernelContext* p_ctx) const override {
    return ComputeUniform(*this, *p_ctx, shape_, GetDType());
  }

 private:
  TensorShape shape_;
};

class RandomUniformLike final : public CudaKernel, protected RandomUniformBase {
 public:
  explicit RandomUniformLike(const OpKernelInfo& info) : CudaKernel(info), RandomUniformBase(info) {}
  Status ComputeInternal(OpKernelContext* p_ctx) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
