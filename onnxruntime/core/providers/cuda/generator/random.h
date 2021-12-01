// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#include "core/providers/cuda/generator/random_impl.h"

namespace onnxruntime {
namespace cuda {

#define RANDOM_COMPUTE_IMPL(name)                                                                        \
  template <typename T>                                                                                  \
  struct name##ComputeImpl {                                                                             \
    void operator()(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const float alpha, \
                    const float beta, PhiloxGenerator& generator, Tensor& Y) const {                     \
      typedef typename ToCudaType<T>::MappedType CudaT;                                                  \
      CudaT* Y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());                             \
      name##KernelImpl<CudaT>(prop, stream, N, alpha, beta, generator, Y_data);                          \
    }                                                                                                    \
  };

RANDOM_COMPUTE_IMPL(RandomNormal)
RANDOM_COMPUTE_IMPL(RandomUniform)

#undef RANDOM_COMPUTE_IMPL

class RandomBase : public CudaKernel {
 protected:
  RandomBase(const OpKernelInfo& info) : CudaKernel(info) {
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
      ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) &&
                      dtype_ != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                  "Invalid dtype of ", dtype_);
    }
  }

 protected:
  std::unique_ptr<PhiloxGenerator> generator_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ =
      ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;  // optional and may be inferred
};

class RandomNormalBase : public RandomBase {
 protected:
  RandomNormalBase(const OpKernelInfo& info) : RandomBase(info) {
    ORT_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("mean", &mean_).IsOK());
  }

  Status Compute(OpKernelContext* p_ctx, const TensorShape& shape, int dtype) const;

 protected:
  float scale_;
  float mean_;
};

class RandomNormal final : public RandomNormalBase {
 public:
  explicit RandomNormal(const OpKernelInfo& info) : RandomNormalBase(info) {
    if (dtype_ == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      dtype_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
    std::vector<int64_t> shape;
    ORT_ENFORCE(info.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status ComputeInternal(OpKernelContext* p_ctx) const override;

 private:
  TensorShape shape_;
};

class RandomNormalLike final : public RandomNormalBase {
 public:
  explicit RandomNormalLike(const OpKernelInfo& info) : RandomNormalBase(info) {}
  Status ComputeInternal(OpKernelContext* p_ctx) const override;
};

class RandomUniformBase : public RandomBase {
 protected:
  RandomUniformBase(const OpKernelInfo& info) : RandomBase(info) {
    float low, high;
    ORT_ENFORCE(info.GetAttr<float>("low", &low).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("high", &high).IsOK());
    from_ = low;
    range_ = high - low;
  }

  Status Compute(OpKernelContext* p_ctx, const TensorShape& shape, int dtype) const;

 protected:
  float range_;
  float from_;
};

class RandomUniform final : public RandomUniformBase {
 public:
  explicit RandomUniform(const OpKernelInfo& info) : RandomUniformBase(info) {
    if (dtype_ == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      dtype_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
    std::vector<int64_t> shape;
    ORT_ENFORCE(info.GetAttrs<int64_t>("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status ComputeInternal(OpKernelContext* p_ctx) const override;

 private:
  TensorShape shape_;
};

class RandomUniformLike final : public RandomUniformBase {
 public:
  explicit RandomUniformLike(const OpKernelInfo& info) : RandomUniformBase(info) {}
  Status ComputeInternal(OpKernelContext* p_ctx) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
