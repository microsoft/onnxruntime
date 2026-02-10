// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Softmax kernel implementation for Telum EP
 *
 * Uses zDNN's zdnn_softmax. zDNN softmax only supports the vector dimension in dim1 of a ZDNN_3DS tensor.
 *
 * Current Telum EP implementation supports ONNX Softmax where axis == last dimension only.
 */
class Softmax final : public TelumKernel {
 public:
  explicit Softmax(const OpKernelInfo& info) : TelumKernel(info) {
    // For ONNX Softmax (since_version 13), default axis is -1.
    axis_ = -1;
    (void)info.GetAttr<int64_t>("axis", &axis_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(X != nullptr, "Input is null");

    ORT_RETURN_IF_ERROR(ValidateStaticShape(X->Shape()));

    const auto& x_shape = X->Shape();
    const auto& dims = x_shape.GetDims();
    ORT_RETURN_IF_NOT(!dims.empty(), "Softmax requires rank >= 1");

    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t axis = HandleNegativeAxis(axis_, rank);

    // zDNN softmax does not support arbitrary axis. We only support axis == last dim.
    if (axis != rank - 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Telum EP: Softmax supports axis==last dimension only. ",
                             "Got axis=", axis_, " (normalized to ", axis, "), rank=", rank);
    }

    const int64_t vector_len = dims.back();
    int64_t batch = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
      batch *= dims[i];
    }

    // Output has the same shape as input.
    Tensor* Y = context->Output(0, x_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Coerce input to ZDNN_3DS where:
    // - dim1 is the vector dimension
    // - dim2 and dim4 are batching dimensions
    // We use (dim4=batch, dim2=1) so we process `batch` vectors of length `vector_len`.
    const TensorShape logical_shape({batch, 1, vector_len});

    zdnn_ztensor zdnn_x, zdnn_y;
    ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(*X, logical_shape, zdnn_x, ZDNN_3DS));
    ZTensorGuard guard_x(&zdnn_x);

    ORT_RETURN_IF_ERROR(TensorConverter::InitZTensorForOutputWithShape(*Y, logical_shape, zdnn_y, ZDNN_3DS));
    ZTensorGuard guard_y(&zdnn_y);

    // Let zDNN allocate internal temporary storage by passing save_area = nullptr.
    const zdnn_status status = zdnn_softmax(&zdnn_x, nullptr, SOFTMAX_ACT_NONE, &zdnn_y);
    ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_softmax"));

    ORT_RETURN_IF_ERROR(ConvertFromZTensor(zdnn_y, *Y));

    return Status::OK();
  }

 private:
  int64_t axis_{-1};
};

ONNX_OPERATOR_KERNEL_EX(
    Softmax,
    kOnnxDomain,
    13,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()}),
    Softmax);

}  // namespace telum
}  // namespace onnxruntime
