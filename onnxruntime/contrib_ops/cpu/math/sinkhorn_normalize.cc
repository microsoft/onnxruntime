// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/framework/op_kernel.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace contrib {
namespace {

void NormalizeAxis(float* matrix, int64_t order, float epsilon, bool by_row) {
  for (int64_t outer = 0; outer < order; ++outer) {
    float sum = 0.0f;
    for (int64_t inner = 0; inner < order; ++inner) {
      const int64_t index = by_row ? outer * order + inner : inner * order + outer;
      sum += matrix[index];
    }

    const float denominator = sum + epsilon;
    for (int64_t inner = 0; inner < order; ++inner) {
      const int64_t index = by_row ? outer * order + inner : inner * order + outer;
      matrix[index] /= denominator;
    }
  }
}

void NormalizeSinkhorn(float* matrix, int64_t order, int64_t iterations, float epsilon) {
  NormalizeAxis(matrix, order, epsilon, false);
  for (int64_t iteration = 1; iteration < iterations; ++iteration) {
    NormalizeAxis(matrix, order, epsilon, true);
    NormalizeAxis(matrix, order, epsilon, false);
  }
}

class SinkhornNormalize final : public OpKernel {
 public:
  explicit SinkhornNormalize(const OpKernelInfo& info) : OpKernel(info) {
    iterations_ = info.GetAttrOrDefault<int64_t>("iterations", 1);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(iterations_ >= 1, "iterations must be at least 1, got ", iterations_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(input != nullptr, "SinkhornNormalize requires an input.");

    const TensorShape& shape = input->Shape();
    const auto& dims = shape.GetDims();
    ORT_RETURN_IF_NOT(dims.size() >= 2, "Input is expected to have at least 2 dimensions, got ", dims.size());

    const int64_t order = dims[dims.size() - 1];
    ORT_RETURN_IF_NOT(dims[dims.size() - 2] == order,
                      "The last two dimensions must be equal, got ", dims[dims.size() - 2], " and ", order);
    ORT_RETURN_IF_NOT(order >= 1, "The matrix order must be at least 1, got ", order);

    Tensor* output = context->Output(0, shape);
    const int64_t element_count = shape.Size();
    std::copy_n(input->Data<float>(), static_cast<size_t>(element_count), output->MutableData<float>());

    const int64_t matrix_size = order * order;
    const int64_t num_matrices = element_count / matrix_size;
    float* output_data = output->MutableData<float>();
    for (int64_t matrix = 0; matrix < num_matrices; ++matrix) {
      NormalizeSinkhorn(output_data + matrix * matrix_size, order, iterations_, epsilon_);
    }

    return Status::OK();
  }

 private:
  int64_t iterations_{};
  float epsilon_{};
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    SinkhornNormalize, kMSDomain, 1, kCpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SinkhornNormalize);

}  // namespace contrib
}  // namespace onnxruntime
