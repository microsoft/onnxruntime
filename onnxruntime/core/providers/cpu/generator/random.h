// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include "core/common/gsl.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/random_seed.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

template <typename OutputType>
Status MultinomialComputeShared(AllocatorPtr& alloc,
                                const Tensor& X,
                                const int64_t batch_size,
                                const int64_t num_classes,
                                const int64_t num_samples,
                                std::default_random_engine& generator,
                                Tensor& Y);

class RandomNormal final : public OpKernel {
 public:
  RandomNormal(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("mean", &mean_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
    } else {
      // node index is added to the global seed to avoid two nodes generating the same sequence of random data
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(utils::GetRandomSeed() + info.node().Index())};
    }

    int64_t dtype;
    ORT_ENFORCE(info.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
    ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", dtype_);

    TensorShapeVector shape;
    ORT_ENFORCE(info.GetAttrs("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;

  // generator_ is updated with every call to Compute().
  // use generator_mutex_ to ensure Compute() can be called concurrently.
  // this is to ensure that a model with random generators is deterministic and still can be executed in parallel.
  mutable std::default_random_engine generator_;
  mutable onnxruntime::OrtMutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomNormalLike final : public OpKernel {
 public:
  RandomNormalLike(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("mean", &mean_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
    } else {
      // node index is added to the global seed to avoid two nodes generating the same sequence of random data
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(utils::GetRandomSeed() + info.node().Index())};
    }

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
      ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float mean_;
  float scale_;

  // see comments for generator_ and generator_mutex_ in RandomNormal class.
  mutable std::default_random_engine generator_;
  mutable onnxruntime::OrtMutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;  // optional and may be inferred
};

class RandomUniform final : public OpKernel {
 public:
  RandomUniform(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("high", &high_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("low", &low_).IsOK());

    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
    } else {
      // node index is added to the global seed to avoid two nodes generating the same sequence of random data
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(utils::GetRandomSeed() + info.node().Index())};
    }

    int64_t dtype;
    ORT_ENFORCE(info.GetAttr<int64_t>("dtype", &dtype).IsOK());
    dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
    ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", dtype_);

    TensorShapeVector shape;
    ORT_ENFORCE(info.GetAttrs("shape", shape).IsOK());
    shape_ = TensorShape(shape);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;

  // see comments for generator_ and generator_mutex_ in RandomNormal class.
  mutable std::default_random_engine generator_;
  mutable onnxruntime::OrtMutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_;
  TensorShape shape_;
};

class RandomUniformLike final : public OpKernel {
 public:
  RandomUniformLike(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("high", &high_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("low", &low_).IsOK());
    // read optional seed attribute and generate if not provided
    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
    } else {
      // node index is added to the global seed to avoid two nodes generating the same sequence of random data
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(utils::GetRandomSeed() + info.node().Index())};
    }

    int64_t dtype;
    if (info.GetAttr<int64_t>("dtype", &dtype).IsOK()) {
      dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype);
      ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(dtype_) && dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                  "Invalid dtype of ", dtype_);
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  float high_;
  float low_;

  // see comments for generator_ and generator_mutex_ in RandomNormal class.
  mutable std::default_random_engine generator_;
  mutable onnxruntime::OrtMutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType dtype_ = ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_UNDEFINED;  // optional and may be inferred
};

class Multinomial final : public OpKernel {
 public:
  Multinomial(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("sample_size", &num_samples_).IsOK());

    float seed = 0.f;
    if (info.GetAttr<float>("seed", &seed).IsOK()) {
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(seed)};
    } else {
      // node index is added to the global seed to avoid two nodes generating the same sequence of random data
      generator_ = std::default_random_engine{gsl::narrow_cast<uint32_t>(utils::GetRandomSeed() + info.node().Index())};
    }

    int64_t output_dtype_tmp;
    if (!info.GetAttr<int64_t>("dtype", &output_dtype_tmp).IsOK()) {
      output_dtype_ = ONNX_NAMESPACE::TensorProto_DataType_INT32;  // default is INT32 as per spec
    } else {
      output_dtype_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(output_dtype_tmp);
    }
    ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(output_dtype_) && output_dtype_ != ONNX_NAMESPACE::TensorProto::UNDEFINED,
                "Invalid dtype of ", output_dtype_);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  int64_t num_samples_;

  // see comments for generator_ and generator_mutex_ in RandomNormal class.
  mutable std::default_random_engine generator_;
  mutable onnxruntime::OrtMutex generator_mutex_;
  ONNX_NAMESPACE::TensorProto::DataType output_dtype_;
};
}  // namespace onnxruntime
