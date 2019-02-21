// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

std::vector<VectorInt64> InferOutputShapes(OpKernelInfo info) {
  std::vector<VectorInt64> output_tensor_shapes = {};

  auto& node = info.node();
  auto output_defs = node.OutputDefs();
  auto outputCount = output_defs.size();

  for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
    output_tensor_shapes.push_back({});
    if (!output_defs[outputIndex]->Exists())
      continue;

    auto shape = output_defs[outputIndex]->Shape();
    for (auto dim : shape->dim()) {
      output_tensor_shapes[outputIndex].push_back(dim.dim_value());
    }
  }
  return output_tensor_shapes;
}

template <typename T>
class SinGrad final : public OpKernel {
 public:
  explicit SinGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SinGrad);
};

template <typename T>
class MulGrad final : public OpKernel {
 public:
  explicit MulGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MulGrad);
};

template <typename T>
class ConvGrad final : public OpKernel {
 public:
  explicit ConvGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvGrad);
};

template <typename T>
class FlattenGrad final : public OpKernel {
 public:
  explicit FlattenGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FlattenGrad);
};

template <typename T>
class UnsqueezeGrad final : public OpKernel {
 public:
  explicit UnsqueezeGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(UnsqueezeGrad);
};

template <typename T>
class ReluGrad final : public OpKernel {
 public:
  explicit ReluGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReluGrad);
};

template <typename T>
class AddGrad final : public OpKernel {
 public:
  explicit AddGrad(const OpKernelInfo& info) : OpKernel(info) {
    output_tensor_shapes_ = InferOutputShapes(info);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AddGrad);

  std::vector<VectorInt64> output_tensor_shapes_;
};

template <typename T>
class MatMulGrad final : public OpKernel {
 public:
  explicit MatMulGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulGrad);
};

template <typename T>
class SubGrad final : public OpKernel {
 public:
  explicit SubGrad(const OpKernelInfo& info) : OpKernel(info) {
    output_tensor_shapes_ = InferOutputShapes(info);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SubGrad);

  std::vector<VectorInt64> output_tensor_shapes_;
};

template <typename T>
class PowGrad final : public OpKernel {
 public:
  explicit PowGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PowGrad);
};

template <typename T>
class ReduceMeanGrad final : public OpKernel {
 public:
  explicit ReduceMeanGrad(const OpKernelInfo& info) : OpKernel(info) {
    output_tensor_shapes_ = InferOutputShapes(info);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReduceMeanGrad);

  std::vector<VectorInt64> output_tensor_shapes_;
};

template <typename T>
class SigmoidGrad final : public OpKernel {
 public:
  explicit SigmoidGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SigmoidGrad);
};

template <typename T>
class SoftmaxGrad final : public OpKernel {
 public:
  explicit SoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxGrad);
  int64_t axis_;
};
}  // namespace contrib
}  // namespace onnxruntime
