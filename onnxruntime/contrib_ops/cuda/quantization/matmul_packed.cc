// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <string>

#include "contrib_ops/cuda/quantization/matmul_packed_projection.cuh"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulPacked final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulPacked(const OpKernelInfo& info) : CudaKernel(info) {
    std::string quant_type;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("quant_type", &quant_type));
    ORT_ENFORCE(quant_type == "iq2_xxs" || quant_type == "q8_0" || quant_type == "nvfp4",
                "MatMulPacked requires quant_type='iq2_xxs', 'q8_0' or 'nvfp4'.");
    const bool iq2 = quant_type == "iq2_xxs";
    const bool nvfp4 = quant_type == "nvfp4";
    format_ = iq2     ? PackedProjectionFormat::IQ2XXSGGUF
              : nvfp4 ? PackedProjectionFormat::NVFP4GGUF
                      : PackedProjectionFormat::Q8GGUF;
    ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("K", &k_));
    ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("N", &n_));
    ORT_ENFORCE(k_ > 0 && k_ <= std::numeric_limits<int32_t>::max() && k_ % (iq2 ? 256 : 64) == 0,
                "MatMulPacked requires positive int32 K divisible by 256 (IQ2_XXS) or 64 (Q8_0).");
    ORT_ENFORCE(n_ > 0 && n_ <= std::numeric_limits<int32_t>::max() - 32,
                "MatMulPacked requires positive N <= INT32_MAX-32.");
    ORT_ENFORCE(!nvfp4 || k_ == 4096 || k_ == 8192, "MatMulPacked NVFP4 requires K=4096 or K=8192.");
    packed_row_bytes_ = iq2 ? (k_ / 256) * 66 : nvfp4 ? (k_ / 64) * 36
                                                      : (k_ / 32) * 34;
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* a = context->Input<Tensor>(0);
    const auto* b = context->Input<Tensor>(1);
    const auto* global_scale = context->InputCount() > 2 ? context->Input<Tensor>(2) : nullptr;
    const bool nvfp4 = format_ == PackedProjectionFormat::NVFP4GGUF;
    ORT_RETURN_IF_NOT(nvfp4 == (global_scale != nullptr),
                      "MatMulPacked global_scale is required for NVFP4 and must be absent for other codecs.");
    ORT_RETURN_IF_NOT(!global_scale || (global_scale->Shape().NumDimensions() <= 1 &&
                                        global_scale->Shape().Size() == 1),
                      "MatMulPacked global_scale must be a scalar or a one-element vector.");
    const auto& shape = a->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1 && shape.GetDims().back() == k_,
                      "MatMulPacked A must have shape [..., K].");
    ORT_RETURN_IF_NOT(b->Shape().NumDimensions() == 2 && b->Shape()[0] == n_ &&
                          b->Shape()[1] == packed_row_bytes_,
                      "MatMulPacked B must have shape [N, packed_row_bytes] for the selected codec.");
    const int64_t rows = shape.Size() / k_;
    ORT_RETURN_IF_NOT(rows >= 1 && rows <= 8, "MatMulPacked CUDA supports 1 through 8 activation rows.");
    ORT_RETURN_IF_NOT(GetDeviceProp().major >= 8, "MatMulPacked requires CUDA SM80 or later.");
    auto output_shape = shape.AsShapeVector();
    output_shape.back() = n_;
    auto* y = context->Output(0, TensorShape(output_shape));
    CUDA_RETURN_IF_ERROR(LaunchPackedProjection(
        format_, a->DataRaw(), b->Data<uint8_t>(), nullptr,
        global_scale ? global_scale->Data<float>() : nullptr,
        false, static_cast<int>(rows), static_cast<int>(n_), static_cast<int>(k_),
        y->MutableDataRaw(), Stream(context)));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulPacked);
  int64_t k_;
  int64_t n_;
  int64_t packed_row_bytes_;
  PackedProjectionFormat format_;
};

ONNX_OPERATOR_KERNEL_EX(
    MatMulPacked, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("TW", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TF", DataTypeImpl::GetTensorType<float>()),
    MatMulPacked);

}  // namespace onnxruntime::contrib::cuda
