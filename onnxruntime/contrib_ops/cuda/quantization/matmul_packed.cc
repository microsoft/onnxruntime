// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed.h"

#include <algorithm>
#include <string>
#include <type_traits>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}
}  // namespace

#define REGISTER_MATMUL_PACKED_CUDA(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      MatMulPacked,                                                        \
      kMSDomain, 1,                                                        \
      T,                                                                   \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())    \
          .TypeConstraint("Tscale", DataTypeImpl::GetTensorType<float>()), \
      MatMulPacked<T>);

REGISTER_MATMUL_PACKED_CUDA(MLFloat16)
REGISTER_MATMUL_PACKED_CUDA(BFloat16)

#undef REGISTER_MATMUL_PACKED_CUDA

template <typename T>
MatMulPacked<T>::MatMulPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("K", &k_).IsOK() && k_ > 0, "MatMulPacked requires K > 0");
  ORT_ENFORCE(info.GetAttr<int64_t>("N", &n_).IsOK() && n_ > 0, "MatMulPacked requires N > 0");
  ORT_ENFORCE(k_ <= kMatMulPackedMaxDimension && n_ <= kMatMulPackedMaxDimension,
              "MatMulPacked requires K and N <= ", kMatMulPackedMaxDimension);
  ORT_ENFORCE(k_ % 32 == 0, "MatMulPacked requires K to be a multiple of 32");
  ORT_ENFORCE(info.GetAttr<int64_t>("packed_type", &ggml_type_).IsOK() && ggml_type_ >= 0,
              "MatMulPacked requires packed_type >= 0");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(ggml_type_) || ggml_type_ == gguf_quant::kGgmlTypeNVFP4,
              "MatMulPacked: unsupported packed_type=", ggml_type_);
  ORT_ENFORCE(gguf_quant::GgmlRowBytes(static_cast<int>(ggml_type_), k_) > 0,
              "MatMulPacked: invalid K=", k_, " for packed_type=", ggml_type_);
  if constexpr (std::is_same_v<T, BFloat16>) {
    ORT_ENFORCE(GetDeviceProp().major >= 8, "MatMulPacked BF16 requires compute capability 8.0 or higher");
    ORT_THROW_IF_ERROR(GetMatMulPackedBf16KernelSupport(GetDeviceId(), bf16_tc_supported_));
  }
  const std::string policy = info.GetConfigOptions().GetConfigOrDefault(
      kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv, "0");
  ORT_ENFORCE(policy == "0" || policy == "1",
              kOrtSessionOptionsCudaMatMulPackedAllowUnroundedGemv, " must be 0 or 1");
  allow_unrounded_gemv_ = policy == "1";
  std::string act = info.GetAttrOrDefault<std::string>("activation_type", "none");
  if (act == "none" || act.empty()) {
    activation_type_ = 0;
  } else if (act == "relu2" || act == "relu_squared") {
    activation_type_ = 1;
  } else if (act == "silu" || act == "swish") {
    activation_type_ = 2;
  } else {
    ORT_THROW("MatMulPacked: unsupported activation_type=", act);
  }
}

template <typename T>
Status MatMulPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  const Tensor* global_scale = context->InputCount() > 2 ? context->Input<Tensor>(2) : nullptr;
  ORT_RETURN_IF_NOT(a != nullptr && b != nullptr, "MatMulPacked requires inputs A and B");
  ORT_RETURN_IF_NOT(!global_scale ||
                        (ggml_type_ == gguf_quant::kGgmlTypeNVFP4 &&
                         (global_scale->Shape().NumDimensions() == 0 ||
                          global_scale->Shape() == TensorShape({1}))),
                    "MatMulPacked global_scale requires NVFP4 and a scalar or [1] shape");

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "MatMulPacked input A must have rank >= 1");
  const size_t last_axis = a_shape.NumDimensions() - 1;
  ORT_RETURN_IF_NOT(a_shape[last_axis] == k_, "MatMulPacked A last dimension must equal K");

  const auto& b_shape = b->Shape();
  ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 2, "MatMulPacked weight B must be 2D [N, row_bytes]");
  ORT_RETURN_IF_NOT(b_shape[0] == n_, "MatMulPacked B dim 0 (", b_shape[0], ") != N (", n_, ")");
  const int64_t expected_rb = gguf_quant::GgmlRowBytes(static_cast<int>(ggml_type_), k_);
  ORT_RETURN_IF_NOT(expected_rb > 0 && b_shape[1] == expected_rb,
                    "MatMulPacked weight B row_bytes=", b_shape[1], " != expected=", expected_rb);

  const int64_t m = a_shape.SizeToDimension(last_axis);
  ORT_RETURN_IF_NOT(m >= 0 && m <= std::numeric_limits<int>::max(),
                    "MatMulPacked requires M in 0..", std::numeric_limits<int>::max());
  ORT_RETURN_IF_NOT((n_ + 7) / 8 <= GetDeviceProp().maxGridSize[0],
                    "MatMulPacked N exceeds the CUDA grid limit");
  const int64_t scratch_rows = std::min<int64_t>(n_, kMatMulPackedPrefillNTile);
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(m, k_, sizeof(T)) &&
                        PackedTensorBytesFit(n_, expected_rb, 1) &&
                        PackedTensorBytesFit(m, n_, sizeof(T)) &&
                        (m <= 8 || PackedTensorBytesFit(scratch_rows, k_, sizeof(T))),
                    "MatMulPacked tensor byte size exceeds the supported address range");
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;

  MatMulPackedParams params{};
  params.b_data = b->Data<uint8_t>();
  params.global_scale = global_scale ? global_scale->Data<float>() : nullptr;
  params.allow_unrounded_gemv = allow_unrounded_gemv_;
  params.k = static_cast<int>(k_);
  params.n = static_cast<int>(n_);
  params.row_bytes = expected_rb;
  params.ggml_type = static_cast<int>(ggml_type_);
  params.activation_type = activation_type_;
  params.bf16_tc_supported = bf16_tc_supported_;
  params.fp32_prefill = !is_bf16 && ggml_type_ != gguf_quant::kGgmlTypeF16;

  size_t workspace_bytes = 0;
  ORT_RETURN_IF_ERROR(GetMatMulPackedWorkspaceBytes(params, m, workspace_bytes));

  TensorShapeVector y_dims = a_shape.AsShapeVector();
  y_dims[last_axis] = n_;
  Tensor* y = context->Output(0, TensorShape(y_dims));

  if (m == 0) return Status::OK();

  IAllocatorUniquePtr<uint8_t> scratch_w;
  if (workspace_bytes > 0) {
    scratch_w = GetScratchBuffer<uint8_t>(workspace_bytes, GetComputeStream(context));
  }

  return LaunchMatMulPackedCuda(
      GetCublasHandle(context),
      Stream(context),
      is_bf16,
      a->DataRaw(),
      params,
      scratch_w ? scratch_w.get() : nullptr,
      y->MutableDataRaw(),
      m);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
