// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/rms_norm_rotary.h"
#include <cmath>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
ONNX_OPERATOR_KERNEL_EX(
    RMSNormRotary, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 5),
    RMSNormRotary);

RMSNormRotary::RMSNormRotary(const OpKernelInfo& info)
    : CudaKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-6f)),
      input_start_(info.GetAttrOrDefault<int64_t>("input_start", 0)),
      input_width_(info.GetAttrOrDefault<int64_t>("input_width", 0)),
      rotary_dim_(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0)),
      fp32_rotary_(info.GetAttrOrDefault<int64_t>("use_fp32_arithmetic", 1) == 1),
      flatten_batch_(info.GetAttrOrDefault<int64_t>("flatten_batch", 0) == 1) {
  const auto arithmetic = info.GetAttrOrDefault<int64_t>("use_fp32_arithmetic", 1);
  const auto flatten = info.GetAttrOrDefault<int64_t>("flatten_batch", 0);
  ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ >= 0 && input_start_ >= 0 && input_width_ >= 0 &&
                  rotary_dim_ > 0 && rotary_dim_ % 2 == 0 && (arithmetic == 0 || arithmetic == 1) &&
                  (flatten == 0 || flatten == 1),
              "RMSNormRotary requires finite nonnegative epsilon, nonnegative slice, positive even rotary width and mode 0/1");
}

Status RMSNormRotary::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  const auto* positions = context->Input<Tensor>(2);
  const auto* cosine = context->Input<Tensor>(3);
  const auto* sine = context->Input<Tensor>(4);
  const auto* mode = context->Input<Tensor>(5);
  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 4 && shape[2] > 0 && shape[3] > 0,
                    "RMSNormRotary input must have shape [B,S,heads,stride] with positive heads/stride");
  const int64_t width = input_width_ == 0 ? shape[3] : input_width_;
  ORT_RETURN_IF_NOT(width > 0 && width <= 1024 && input_start_ <= shape[3] && width <= shape[3] - input_start_ &&
                        rotary_dim_ <= width && scale->Shape() == TensorShape({width}),
                    "RMSNormRotary slice, rotary width or scale shape is incompatible");
  const auto& cache = cosine->Shape();
  ORT_RETURN_IF_NOT(cache.NumDimensions() == 2 && cache[0] > 0 && cache[1] == rotary_dim_ / 2 &&
                        sine->Shape() == cache,
                    "RMSNormRotary cosine/sine must have matching shape [cache_rows,rotary_width/2]");
  const bool scalar = positions->Shape().Size() == 1;
  const int64_t output_batch = flatten_batch_ ? 1 : shape[0];
  const int64_t output_sequence = flatten_batch_ ? static_cast<int64_t>(SafeInt<int64_t>(shape[0]) * shape[1]) : shape[1];
  ORT_RETURN_IF_NOT(scalar || positions->Shape() == TensorShape({output_batch, output_sequence}),
                    "RMSNormRotary positions must contain one base position or have shape [B,S]");
  ORT_RETURN_IF_NOT(!scalar || output_sequence <= cache[0], "RMSNormRotary sequence exceeds cache capacity");
  bool dense = false;
  if (mode != nullptr) {
    ORT_RETURN_IF_NOT(mode->Shape().Size() == 1 && (mode->Data<int32_t>()[0] == 0 || mode->Data<int32_t>()[0] == 1),
                      "RMSNormRotary prefill_numerics must contain one int32 value 0 or 1");
    dense = mode->Data<int32_t>()[0] == 1;
  }
  const TensorShape norm_shape({shape[0], shape[1], shape[2], width});
  const int groups = SafeInt<int>(norm_shape.Size() / width);
  const int heads = SafeInt<int>(shape[2]), stride = SafeInt<int>(shape[3]);
  const int sequence = SafeInt<int>(output_sequence);
  const int dim = SafeInt<int>(width), rotary = SafeInt<int>(rotary_dim_), cache_rows = SafeInt<int>(cache[0]);
  auto* output = context->Output(0, TensorShape({output_batch, output_sequence, SafeInt<int64_t>(shape[2]) * width}));
  auto* normalized = context->OutputCount() > 1 ? context->Output(1, norm_shape) : nullptr;
  if (groups == 0) return Status::OK();
  const auto* x = reinterpret_cast<const uint16_t*>(input->Data<BFloat16>()) + input_start_;
  return LaunchRMSNormRotary(
      Stream(context), x, scale->Data<float>(), positions->Data<int64_t>(),
      reinterpret_cast<const uint16_t*>(cosine->Data<BFloat16>()),
      reinterpret_cast<const uint16_t*>(sine->Data<BFloat16>()),
      reinterpret_cast<uint16_t*>(output->MutableData<BFloat16>()),
      normalized ? normalized->MutableData<float>() : nullptr,
      groups, heads, sequence, stride, dim, rotary, cache_rows, scalar, epsilon_, dense, fp32_rotary_);
}

}  // namespace onnxruntime::contrib::cuda
