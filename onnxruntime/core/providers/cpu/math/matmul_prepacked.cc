// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul_prepacked.h"
#include "core/providers/cpu/math/matmul_helper.h"

#include "core/mlas/inc/mlas.h"
#include "core/optimizer/matmul_prepacking.h"

namespace onnxruntime {


ONNX_OPERATOR_TYPED_KERNEL_EX(
    PackForGemm,
    kOnnxRuntimeDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PackForGemm<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulPrepacked,
    kOnnxRuntimeDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMulPrepacked<float>);

template<typename TEnum>
void GetEnumAttr(const OpKernelInfo& info, const std::string& name, TEnum* enum_value) {
  int64_t integral_representation;
  ORT_ENFORCE(info.GetAttr(name, &integral_representation).IsOK());
  *enum_value = static_cast<TEnum>(integral_representation);
}

template <typename T>
PackForGemm<T>::PackForGemm(const OpKernelInfo& info)
  : OpKernel(info)
{
  gemm_params_ = onnxruntime::make_unique<MLAS_GEMM_PARAMETERS>();
  ORT_ENFORCE(GemmParamsFromNodeAttributes(info, *gemm_params_).IsOK());
}

template <typename T>
PackForGemm<T>::~PackForGemm() = default;

template <typename T>
Status PackForGemm<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const T* input_data = input->Data<T>();
  const auto& input_shape = input->Shape();
  ORT_ENFORCE(input_shape.NumDimensions() >= 2);

  const size_t N = input_shape[input_shape.NumDimensions() - 1];
  const size_t K = input_shape[input_shape.NumDimensions() - 2];
  const size_t input_matrix_size = N*K;

  auto output_shape = input_shape.GetDims();
  output_shape.pop_back();
  output_shape.back() = gemm_params_->PackedSize;
  Tensor* output = ctx->Output(0, output_shape);
  T* output_data = output->MutableData<T>();

  const auto num_iterations = input->Shape().SizeToDimension(input->Shape().NumDimensions() - 2);
  for (int i = 0; i < num_iterations; ++i, input_data += input_matrix_size, output_data += gemm_params_->PackedSize) {
    auto result = MlasGemmPackMatrix(*gemm_params_, input_data, CblasNoTrans, N, output_data, gemm_params_->PackedSize);
    ORT_ENFORCE(result == gemm_params_->PackedSize);
  }

  return Status::OK();
}

template <typename T>
MatMulPrepacked<T>::MatMulPrepacked(const OpKernelInfo& info)
  : OpKernel(info)
{
  gemm_params_ = onnxruntime::make_unique<MLAS_GEMM_PARAMETERS>();
  ORT_ENFORCE(GemmParamsFromNodeAttributes(info, *gemm_params_).IsOK());
}

template <typename T>
MatMulPrepacked<T>::~MatMulPrepacked() = default;

template <typename T>
Status MatMulPrepacked<T>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* A = ctx->Input<Tensor>(0);
  const auto* PackedB = ctx->Input<Tensor>(1);
  const auto* OrigB = ctx->Input<Tensor>(2);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), PackedB->Shape(), false, false, gemm_params_->N, gemm_params_->K));

  MatMulComputeHelper orig_helper;
  if (OrigB != nullptr) {
    ORT_RETURN_IF_ERROR(orig_helper.Compute(A->Shape(), OrigB->Shape(), false, false));
  }

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  const size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    const T* orig_B = nullptr;
    if (OrigB != nullptr) {
      orig_B = OrigB->Data<T>() + orig_helper.LeftOffsets()[i];
    }
    MlasGemm(
      CblasNoTrans,
      CblasNoTrans,
      gemm_params_.get(),
      helper.M(),
      helper.N(),
      helper.K(),
      1.0f,
      A->Data<T>() + helper.LeftOffsets()[i],
      helper.K(),
      orig_B,
      helper.N(),
      PackedB->Data<T>() + helper.RightOffsets()[i],
      0.0f,
      Y->MutableData<T>() + helper.OutputOffsets()[i],
      helper.N(),
      thread_pool);
  }

  return Status::OK();
}

}
