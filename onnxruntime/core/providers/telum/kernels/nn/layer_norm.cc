// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../telum_kernel_common.h"

#include <cmath>
#include <vector>

#include "core/framework/float16.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief LayerNormalization kernel implementation for Telum EP
 *
 * This implementation uses zDNN's `zdnn_moments` + `zdnn_layernorm` to compute the normalization,
 * and applies the ONNX scale/bias vectors on CPU (zDNN layernorm only supports scalar gamma/beta).
 *
 * Current support:
 * - Static shapes only
 * - axis == last dimension only
 * - scale shape: [C]
 * - bias shape: [C] (optional input)
 *
 * Outputs:
 * - Y: same type/shape as X
 * - Mean and InvStdDev: float outputs (optional, but must be computed for zDNN layernorm anyway)
 */
class LayerNormalization final : public TelumKernel {
 public:
  explicit LayerNormalization(const OpKernelInfo& info) : TelumKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* Scale = context->Input<Tensor>(1);
    const Tensor* Bias = context->Input<Tensor>(2);  // optional

    ORT_RETURN_IF_NOT(X != nullptr, "Input X is null");
    ORT_RETURN_IF_NOT(Scale != nullptr, "Input Scale is null");

    ORT_RETURN_IF_ERROR(ValidateStaticShape(X->Shape()));
    ORT_RETURN_IF_ERROR(ValidateStaticShape(Scale->Shape()));
    if (Bias != nullptr) {
      ORT_RETURN_IF_ERROR(ValidateStaticShape(Bias->Shape()));
    }

    const auto& x_shape = X->Shape();
    const auto& dims = x_shape.GetDims();
    ORT_RETURN_IF_NOT(!dims.empty(), "LayerNormalization requires rank >= 1");

    const int64_t rank = static_cast<int64_t>(dims.size());
    const int64_t axis = HandleNegativeAxis(axis_, rank);

    // zDNN moments/layernorm require 4D NHWC and we map normalization over the last dimension (C).
    if (axis != rank - 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Telum EP: LayerNormalization supports axis==last dimension only. ",
                             "Got axis=", axis_, " (normalized to ", axis, "), rank=", rank);
    }

    const int64_t C = dims.back();
    ORT_RETURN_IF_NOT(C >= 0, "Invalid last dimension size");

    // Validate scale/bias shapes: require 1D [C] for now.
    {
      const auto& s_dims = Scale->Shape().GetDims();
      if (!(s_dims.size() == 1 && s_dims[0] == C)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Telum EP: LayerNormalization currently requires Scale shape [C]. ",
                               "Got ", Scale->Shape().ToString(), " with C=", C);
      }
      if (Bias != nullptr) {
        const auto& b_dims = Bias->Shape().GetDims();
        if (!(b_dims.size() == 1 && b_dims[0] == C)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                                 "Telum EP: LayerNormalization currently requires Bias shape [C] when provided. ",
                                 "Got ", Bias->Shape().ToString(), " with C=", C);
        }
      }
    }

    int64_t N = 1;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
      N *= dims[i];
    }

    // Output Y has same shape/type as X.
    Tensor* Y = context->Output(0, x_shape);
    ORT_RETURN_IF_NOT(Y != nullptr, "Failed to allocate output tensor");

    // Mean/InvStdDev output shape: same rank as X with dims [axis..] set to 1.
    std::vector<int64_t> mean_inv_dims;
    mean_inv_dims.reserve(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      mean_inv_dims.push_back((static_cast<int64_t>(i) < axis) ? dims[i] : 1);
    }
    const TensorShape mean_inv_shape(mean_inv_dims);

    float* mean_out = nullptr;
    if (Tensor* mean = context->Output(1, mean_inv_shape); mean != nullptr) {
      mean_out = mean->MutableData<float>();
    }

    float* inv_std_out = nullptr;
    if (Tensor* inv_std = context->Output(2, mean_inv_shape); inv_std != nullptr) {
      inv_std_out = inv_std->MutableData<float>();
    }

    // Map X to NHWC: [N, 1, 1, C]. This normalizes over C for each of the N instances.
    const TensorShape logical_x({N, 1, 1, C});
    const TensorShape logical_moments({N, 1, 1, 1});

    zdnn_ztensor z_x, z_mean, z_var, z_y;
    ORT_RETURN_IF_ERROR(TensorConverter::ConvertToZTensorWithShape(*X, logical_x, z_x, ZDNN_NHWC));
    ZTensorGuard guard_x(&z_x);

    // moments outputs must be same type/format as input.
    ORT_RETURN_IF_ERROR(TensorConverter::InitZTensorForOutputWithShapeAndType(X->GetElementType(),
                                                                              logical_moments, z_mean, ZDNN_NHWC));
    ZTensorGuard guard_mean(&z_mean);

    ORT_RETURN_IF_ERROR(TensorConverter::InitZTensorForOutputWithShapeAndType(X->GetElementType(),
                                                                              logical_moments, z_var, ZDNN_NHWC));
    ZTensorGuard guard_var(&z_var);

    {
      const zdnn_status status = zdnn_moments(&z_x, MOMENTS_BESSEL_POPULATION, &z_mean, &z_var);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_moments"));
    }

    ORT_RETURN_IF_ERROR(TensorConverter::InitZTensorForOutputWithShape(*Y, logical_x, z_y, ZDNN_NHWC));
    ZTensorGuard guard_y(&z_y);

    // zDNN layernorm only supports scalar beta/gamma. Use beta=0, gamma=1 to compute the normalized values.
    {
      const zdnn_status status = zdnn_layernorm(&z_x, &z_mean, &z_var,
                                                /*beta=*/0.0f, /*gamma=*/1.0f, /*epsilon=*/epsilon_,
                                                &z_y);
      ORT_RETURN_IF_ERROR(CheckStatus(status, "zdnn_layernorm"));
    }

    ORT_RETURN_IF_ERROR(ConvertFromZTensor(z_y, *Y));

    // Apply ONNX scale/bias vectors (length C) on CPU: y = y * scale + bias.
    ORT_RETURN_IF_ERROR(ApplyScaleBiasInPlace(*Y, *Scale, Bias, N, C));

    // Export Mean/InvStdDev outputs if requested.
    if (mean_out != nullptr || inv_std_out != nullptr) {
      ORT_RETURN_IF_ERROR(WriteMeanInvStd(*X, z_mean, z_var, mean_out, inv_std_out, static_cast<size_t>(N), epsilon_));
    }

    return Status::OK();
  }

 private:
  int64_t axis_{-1};
  float epsilon_{1e-5f};

  static Status ApplyScaleBiasInPlace(Tensor& y,
                                      const Tensor& scale,
                                      const Tensor* bias,
                                      int64_t N,
                                      int64_t C) {
    ORT_RETURN_IF_NOT(y.Shape().Size() == N * C, "Unexpected output size for LayerNormalization");

    const int32_t ort_type = y.GetElementType();

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      float* y_data = y.MutableData<float>();
      const float* s_data = scale.Data<float>();
      const float* b_data = bias ? bias->Data<float>() : nullptr;

      for (int64_t i = 0; i < N; ++i) {
        const int64_t base = i * C;
        for (int64_t j = 0; j < C; ++j) {
          float v = y_data[base + j] * s_data[j];
          if (b_data) v += b_data[j];
          y_data[base + j] = v;
        }
      }
      return Status::OK();
    }

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      MLFloat16* y_data = y.MutableData<MLFloat16>();
      const MLFloat16* s_data = scale.Data<MLFloat16>();
      const MLFloat16* b_data = bias ? bias->Data<MLFloat16>() : nullptr;

      for (int64_t i = 0; i < N; ++i) {
        const int64_t base = i * C;
        for (int64_t j = 0; j < C; ++j) {
          float v = static_cast<float>(y_data[base + j]) * static_cast<float>(s_data[j]);
          if (b_data) v += static_cast<float>(b_data[j]);
          y_data[base + j] = MLFloat16(v);
        }
      }
      return Status::OK();
    }

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      BFloat16* y_data = y.MutableData<BFloat16>();
      const BFloat16* s_data = scale.Data<BFloat16>();
      const BFloat16* b_data = bias ? bias->Data<BFloat16>() : nullptr;

      for (int64_t i = 0; i < N; ++i) {
        const int64_t base = i * C;
        for (int64_t j = 0; j < C; ++j) {
          float v = static_cast<float>(y_data[base + j]) * static_cast<float>(s_data[j]);
          if (b_data) v += static_cast<float>(b_data[j]);
          y_data[base + j] = BFloat16(v);
        }
      }
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported LayerNormalization type for Telum EP");
  }

  static Status WriteMeanInvStd(const Tensor& x,
                                const zdnn_ztensor& z_mean,
                                const zdnn_ztensor& z_var,
                                float* mean_out,
                                float* inv_std_out,
                                size_t N,
                                float epsilon) {
    // Convert zDNN mean/var tensors back to original layout/type, then export as float outputs.
    const int32_t ort_type = x.GetElementType();

    auto write = [&](const auto* mean_buf, const auto* var_buf) -> Status {
      for (size_t i = 0; i < N; ++i) {
        const float m = static_cast<float>(mean_buf[i]);
        const float v = static_cast<float>(var_buf[i]);
        if (mean_out) mean_out[i] = m;
        if (inv_std_out) inv_std_out[i] = 1.0f / std::sqrt(v + epsilon);
      }
      return Status::OK();
    };

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      std::vector<float> mean_buf(N), var_buf(N);
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_mean, mean_buf.data()), "zdnn_transform_origtensor(mean)"));
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_var, var_buf.data()), "zdnn_transform_origtensor(var)"));
      return write(mean_buf.data(), var_buf.data());
    }

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      std::vector<MLFloat16> mean_buf(N), var_buf(N);
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_mean, mean_buf.data()), "zdnn_transform_origtensor(mean)"));
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_var, var_buf.data()), "zdnn_transform_origtensor(var)"));
      return write(mean_buf.data(), var_buf.data());
    }

    if (ort_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      std::vector<BFloat16> mean_buf(N), var_buf(N);
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_mean, mean_buf.data()), "zdnn_transform_origtensor(mean)"));
      ORT_RETURN_IF_ERROR(CheckZDNNStatus(zdnn_transform_origtensor(&z_var, var_buf.data()), "zdnn_transform_origtensor(var)"));
      return write(mean_buf.data(), var_buf.data());
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported LayerNormalization type for Telum EP");
  }
};

ONNX_OPERATOR_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    17,
    kTelumExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                             DataTypeImpl::GetTensorType<MLFloat16>(),
                             DataTypeImpl::GetTensorType<BFloat16>()})
        .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()),
    LayerNormalization);

}  // namespace telum
}  // namespace onnxruntime
