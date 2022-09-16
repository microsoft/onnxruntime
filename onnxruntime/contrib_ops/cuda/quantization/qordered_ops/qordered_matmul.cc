// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_matmul.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_matmul_utils.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"

#include <functional>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int QOrderedMatMulScaleA = 1;
constexpr int QOrderedMatMulScaleB = 3;
constexpr int QOrderedMatMulScaleC = 7;
constexpr int QOrderedMatMulScaleY = 4;

ONNX_OPERATOR_KERNEL_EX(
    QOrderedMatMul,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, QOrderedMatMulScaleA)   // scale_A
        .InputMemoryType(OrtMemTypeCPUInput, QOrderedMatMulScaleY)   // scale_Y
        .InputMemoryType(OrtMemTypeCPUInput, QOrderedMatMulScaleC),  // scale_C
    QOrderedMatMul);

QOrderedMatMul::QOrderedMatMul(const OpKernelInfo& info) : CudaKernel(info) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

  order_A_ = GetCublasLtOrderAttr(info, "order_A");
  order_B_ = GetCublasLtOrderAttr(info, "order_B");
  order_Y_ = GetCublasLtOrderAttr(info, "order_Y");
  if (order_B_ == CUBLASLT_ORDER_COL) {
    ORT_ENFORCE(order_A_ == CUBLASLT_ORDER_ROW && order_Y_ == CUBLASLT_ORDER_ROW,
                "When order_B is ORDER_COL, other matrix must be ORDER_ROW");
  } else {
    ORT_ENFORCE(order_B_ == CUBLASLT_ORDER_COL4_4R2_8C || order_B_ == CUBLASLT_ORDER_COL32_2R_4R4,
                "If order_B is not ORDER_COL, it must be either ORDER_COL4_4R2_8C or ORDER_COL32_2R_4R4");
    ORT_ENFORCE(order_Y_ == CUBLASLT_ORDER_COL32 && order_A_ == CUBLASLT_ORDER_COL32,
                "If order_B is not ORDER_COL, Only CUBLASLT_ORDER_COL32 is supported for order_A and order_Y");
  }
  const_scale_A_ = const_scale_B_ = const_scale_C_ = const_scale_Y_ = 0.0;
  origin_scale_B_vector_ = nullptr;

#else

  ORT_ENFORCE(false, "Higher CUDA_VERSION is needed!")

#endif
}

Status QOrderedMatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                               /*out*/ bool& is_packed,
                               /*out*/ PrePackedWeights* /* prepacked_weights */) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

  is_packed = false;
  if (order_B_ == CUBLASLT_ORDER_COL) {
    if (input_idx == QOrderedMatMulScaleA) {
      ORT_ENFORCE(tensor.Shape().IsScalar(), "scale_A_ must be scala!");
      const_scale_A_ = *tensor.Data<float>();
      ORT_ENFORCE(const_scale_A_ > 0.0f, "scale_A_ must > 0.0f");
    }

    if (input_idx == QOrderedMatMulScaleB) {
      if (tensor.Shape().IsScalar()) {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(&const_scale_B_, tensor.Data<float>(), sizeof(float), cudaMemcpyDeviceToHost));
        ORT_ENFORCE(const_scale_B_ > 0.0f, "scale_B_ must > 0.0f if scalar");
      } else {
        ORT_ENFORCE(tensor.Shape().NumDimensions() == 1, "scale_b_ must be 1d array if not scalar!");
        scale_b_size_ = gsl::narrow_cast<int>(tensor.Shape()[0]);
        origin_scale_B_vector_ = tensor.Data<float>();
      }
    }

    if (input_idx == QOrderedMatMulScaleY) {
      ORT_ENFORCE(tensor.Shape().IsScalar(), "scale_Y_ must be scala!");
      const_scale_Y_ = *tensor.Data<float>();
      ORT_ENFORCE(const_scale_Y_ > 0.0f, "scale_Y_ must > 0.0f");
      if (origin_scale_B_vector_) {
        calculated_alpha_ = BufferUniquePtr(alloc->Alloc(scale_b_size_ * sizeof(float)), BufferDeleter(alloc));
        float rescale = static_cast<float>((double)const_scale_A_ / const_scale_Y_);
        CUBLAS_RETURN_IF_ERROR(cublasSscal(CublasHandle(), scale_b_size_, &rescale, (float*)calculated_alpha_.get(), 1));
      }
    }
  }

#else

  ORT_UNUSED_PARAMETER(tensor);
  ORT_UNUSED_PARAMETER(input_idx);
  ORT_UNUSED_PARAMETER(alloc);
  ORT_UNUSED_PARAMETER(is_packed);
  ORT_ENFORCE(false, "Higher CUDA_VERSION is needed!")

#endif

  return Status::OK();
}

Status QOrderedMatMul::ComputeInternal(OpKernelContext* context) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

  ORT_ENFORCE(order_B_ == CUBLASLT_ORDER_COL, "COL32 related order processing will be implemented later!");

  int64_t rowsA = 0, colsA = 0, batchA = 1, elementsA = 0;
  int64_t rowsB = 0, colsB = 0, batchB = 1, elementsB = 0;
  int64_t rowsC = 0, colsC = 0, batchC = 1, elementsC = 0;

  const Tensor& tensor_A = *context->Input<Tensor>(0);
  const Tensor& tensor_B = *context->Input<Tensor>(2);

  // Support General case only. Broadcasting is not supported now.
  ORT_ENFORCE(tensor_A.Shape().NumDimensions() == 2 || tensor_A.Shape().NumDimensions() == 3);
  ORT_ENFORCE(tensor_B.Shape().NumDimensions() == 2 || tensor_B.Shape().NumDimensions() == 3);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(tensor_A, (cublasLtOrder_t)order_A_, (cublasLtOrder_t)order_A_, rowsA, colsA, batchA, elementsA));
  ORT_RETURN_IF_ERROR(CheckTensorOrder(tensor_B, (cublasLtOrder_t)order_B_, (cublasLtOrder_t)order_B_, rowsB, colsB, batchB, elementsB));
  ORT_ENFORCE(const_scale_A_ > 0.0f && const_scale_Y_ > 0.0f, "scale_A and scale_Y must be constant value");
  ORT_ENFORCE(const_scale_B_ > 0.0f || calculated_alpha_.get() != nullptr, "scale_B_ must be constant!");
  ORT_ENFORCE(calculated_alpha_.get() == nullptr || scale_b_size_ == colsB, "if not scalar, scale_B_ must be of size colsB!");

  const Tensor* tensor_bias = context->Input<Tensor>(5);
  ORT_ENFORCE(tensor_bias == nullptr || (tensor_bias->Shape().NumDimensions() == 1 && tensor_bias->Shape()[0] == colsB));
  const float* bias = (tensor_bias == nullptr) ? nullptr : tensor_bias->Data<float>();

  ORT_ENFORCE(batchA == batchB || batchB == 1, "batch count for matrix A and matrix B does not match");
  ORT_ENFORCE(colsA == rowsB, "Sahpe mis-match");
  TensorShape shapeY(tensor_A.Shape());
  shapeY[shapeY.NumDimensions() - 1] = colsB;

  const float zero = 0.0f;
  const int8_t* C = nullptr;
  const float* scaleC = &zero;
  const Tensor* tensor_C = context->Input<Tensor>(6);
  if (tensor_C != nullptr) {
    ORT_ENFORCE(tensor_C->Shape().NumDimensions() == 2 || tensor_C->Shape().NumDimensions() == 3);
    ORT_RETURN_IF_ERROR(CheckTensorOrder(*tensor_C, (cublasLtOrder_t)order_A_, (cublasLtOrder_t)order_A_, rowsC, colsC, batchC, elementsC));
    ORT_ENFORCE(batchC == batchA || batchC == 1);
    ORT_ENFORCE(rowsC == rowsA && colsC == colsB);
    const Tensor* tensor_scaleC = context->Input<Tensor>(7);
    ORT_ENFORCE(tensor_scaleC != nullptr);
    scaleC = tensor_scaleC->Data<float>();
    C = tensor_C->Data<int8_t>();
  }

  Tensor* tensor_Y = context->Output(0, shapeY);
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream();
  auto& device_prop = GetDeviceProp();

  float alpha_value = 0.0;
  const float* alpha = &alpha_value;
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  if (const_scale_B_ == 0.0f) {
    alpha = (const float*)calculated_alpha_.get();
    pointer_mode = (cublasLtPointerMode_t)4; //CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST, 11.4.2 header needed
  } else {
    alpha_value = const_scale_A_ * const_scale_B_ / const_scale_Y_;
  }
  const float beta = *scaleC / const_scale_Y_;
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      gsl::narrow<int32_t>(batchA), rowsA, colsB, colsA,
                                      alpha, tensor_A.Data<int8_t>(), tensor_B.Data<int8_t>(), gsl::narrow<int32_t>(batchB),
                                      bias, &beta, C, gsl::narrow<int32_t>(batchC),
                                      tensor_Y->MutableData<int8_t>(), (cublasLtOrder_t)order_B_,
                                      pointer_mode));


#else

  ORT_UNUSED_PARAMETER(context);
  ORT_ENFORCE(false, "Higher CUDA_VERSION is needed!")

#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
