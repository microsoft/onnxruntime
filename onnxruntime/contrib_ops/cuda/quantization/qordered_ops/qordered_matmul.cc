// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qordered_matmul.h"
#include "qordered_matmul_utils.h"

#include <functional>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    QOrderedMatMul,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_A
        .InputMemoryType(OrtMemTypeCPUInput, 3)   // scale_B
        .InputMemoryType(OrtMemTypeCPUInput, 4)   // scale_Y
        .InputMemoryType(OrtMemTypeCPUInput, 7),  // scale_C
    QOrderedMatMul);

static Status ParseRowMajorTensorMetadata(const Tensor& input_tensor, int64_t& rows,
                                          int64_t& cols, int64_t& batch_count, int64_t& element_count) {
  const auto& dims = input_tensor.Shape().GetDims();

  cols = dims.back();
  rows = (dims.size() <= 1 ? 1LL : dims[dims.size() - 2]);
  batch_count = (dims.size() <= 2
                     ? 1LL
                     : std::accumulate(dims.begin(), dims.begin() + (dims.size() - 2),
                                       1LL, std::multiplies<int64_t>()));

  element_count = cols * rows * batch_count;

  return Status::OK();
}

QOrderedMatMul::QOrderedMatMul(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr("order_A", &order_A_).IsOK());

  ORT_ENFORCE(info.GetAttr("order_B", &order_B_).IsOK());

  ORT_ENFORCE(info.GetAttr("order_Y", &order_Y_).IsOK());

  ORT_ENFORCE(order_B_ == CUBLASLT_ORDER_COL &&
                  order_A_ == CUBLASLT_ORDER_ROW &&
                  order_Y_ == CUBLASLT_ORDER_ROW,
              "QOrderedMatMul: Input 1 and output's data ordering should be ROW_MAJOR and "
              "Input 2's data ordering should be COL_MAJOR");
}

Status QOrderedMatMul::QOrderedMatMul::ComputeInternal(OpKernelContext* context) const {
  int64_t rows_A = 0, cols_A = 0, batch_A = 1, elements_A = 0;
  int64_t rows_B = 0, cols_B = 0, batch_B = 1, elements_B = 0;
  int64_t rows_C = 0, cols_C = 0, batch_C = 1, elements_C = 0;

  const Tensor& tensor_A = *context->Input<Tensor>(0);
  const Tensor& tensor_B = *context->Input<Tensor>(2);

  // Support General case only. No broadcasting, is handled now.
  ORT_ENFORCE(tensor_A.Shape().NumDimensions() == 2 || tensor_A.Shape().NumDimensions() == 3);
  ORT_ENFORCE(tensor_B.Shape().NumDimensions() == 2 || tensor_B.Shape().NumDimensions() == 3);

  ORT_RETURN_IF_ERROR(ParseRowMajorTensorMetadata(tensor_A, rows_A, cols_A, batch_A, elements_A));
  ORT_RETURN_IF_ERROR(ParseRowMajorTensorMetadata(tensor_B, rows_B, cols_B, batch_B, elements_B));

  const float* scale_A = context->Input<Tensor>(1)->Data<float>();
  const float* scale_B = context->Input<Tensor>(3)->Data<float>();
  const float* scale_Y = context->Input<Tensor>(4)->Data<float>();
  ORT_ENFORCE(*scale_Y > 0.0f && *scale_A > 0.0f && *scale_B > 0.0f);

  const Tensor* tensor_bias = context->Input<Tensor>(5);
  ORT_ENFORCE(tensor_bias == nullptr ||
              (tensor_bias->Shape().NumDimensions() == 1 && tensor_bias->Shape()[0] == cols_B));

  const float* bias = (tensor_bias == nullptr) ? nullptr : tensor_bias->Data<float>();

  ORT_ENFORCE(batch_A == batch_B || batch_B == 1, "Batch count for matrix A and matrix B does not match");
  ORT_ENFORCE(cols_A == rows_B, "MatMul shape mis-match");

  TensorShape output_shape(tensor_A.Shape());
  output_shape[output_shape.NumDimensions() - 1] = cols_B;

  constexpr float zero = 0.0f;
  const float* scale_C = &zero;

  const int8_t* C = nullptr;
  const Tensor* tensor_C = context->Input<Tensor>(6);

  if (tensor_C != nullptr) {
    ORT_ENFORCE(tensor_C->Shape().NumDimensions() == 2 || tensor_C->Shape().NumDimensions() == 3);
    ORT_RETURN_IF_ERROR(ParseRowMajorTensorMetadata(*tensor_C, rows_C, cols_C, batch_C, elements_C));

    ORT_ENFORCE(batch_C == batch_A || batch_C == 1);
    ORT_ENFORCE(rows_C == rows_A && cols_C == cols_B);

    const Tensor* tensor_scale_C = context->Input<Tensor>(7);
    ORT_ENFORCE(tensor_scale_C != nullptr);
    scale_C = tensor_scale_C->Data<float>();

    C = tensor_C->Data<int8_t>();
  }

  Tensor* tensor_Y = context->Output(0, output_shape);
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream();
  auto& device_prop = GetDeviceProp();

  const float alpha = *scale_A * *scale_B / *scale_Y;
  const float beta = *scale_C / *scale_Y;

  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      static_cast<int32_t>(batch_A), rows_A, cols_B, cols_A,
                                      &alpha, tensor_A.Data<int8_t>(), tensor_B.Data<int8_t>(),
                                      static_cast<int32_t>(batch_B), bias,
                                      &beta, C, static_cast<int32_t>(batch_C),
                                      tensor_Y->MutableData<int8_t>(), static_cast<cublasLtOrder_t>(order_B_)));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
