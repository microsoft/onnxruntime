// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/layer_norm.h"
#include "contrib_ops/cuda/layer_norm_impl.h"

#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include "qorder_layer_norm.h"
#include "qorder_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedLayerNormalization,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_X
        .InputMemoryType(OrtMemTypeCPUInput, 4),  // scale_Y
    QOrderedLayerNormalization);

QOrderedLayerNormalization::QOrderedLayerNormalization(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
  const cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
  GetCublasLtOrderAttr(op_kernel_info, "order_X", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
  GetCublasLtOrderAttr(op_kernel_info, "order_Y", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
}

//   // TODO: quantize the gamma and beta or not
//   ONNX_CONTRIB_OPERATOR_SCHEMA(QOrderedLayerNormalization)
//       .SetDomain(kOnnxDomain)
//       .SinceVersion(1)
//       .SetDoc("QOrderedLayerNormalization")
//       .Attr("axis",
//             "The first normalization dimension: normalization will be performed along dimensions axis : rank(inputs).",
//             AttributeProto::INT, static_cast<int64_t>(-1))
//       .Attr("epsilon",
//             "The epsilon value to use to avoid division by zero.",
//             AttributeProto::FLOAT, 1e-5f)
//       .Attr("order_X", "cublasLt order of input X", AttributeProto::INT)
//       .Attr("order_Y", "cublasLt order of matrix Y", AttributeProto::INT)
//       .AllowUncheckedAttributes()
//       .Input(0, "X", "Input data tensor from the previous layer.", "Q")
//       .Input(1, "scale_X", "scale of the quantized X", "S")
//       .Input(2, "scale", "Scale tensor.", "F")
//       .Input(3, "B", "Bias tensor.", "F", OpSchema::Optional)
//       .Input(4, "scale_Y", "scale of the quantized X", "S")
//       .Output(0, "Y", "Output data tensor.", "Q")
//       .TypeConstraint(
//           "F",
//           {"tensor(float16)"},
//           "Constrain input scale and bias could be float16 tensors.")
//       .TypeConstraint(
//           "S",
//           {"tensor(float)"},
//           "quantization scale must be float tensors.")
//       .TypeConstraint(
//           "Q",
//           {"tensor(int8)"},
//           "quantization tensor must be int8 tensors.")
//       .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
//         propagateShapeAndTypeFromFirstInput(ctx);
//         propagateElemTypeFromInputToOutput(ctx, 0, 0);
//       });

Status QOrderedLayerNormalization::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<int8_t>::MappedType CudaQ;
  typedef typename ToCudaType<MLFloat16>::MappedType CudaF;

  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(2);
  const Tensor* bias = ctx->Input<Tensor>(3);

  auto X_data = reinterpret_cast<const CudaQ*>(X->Data<int8_t>());
  auto scale_data = reinterpret_cast<const CudaF*>(scale->Data<MLFloat16>());
  auto bias_data = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaF*>(bias->Data<MLFloat16>());

  const TensorShape& x_shape = X->Shape();
  ORT_ENFORCE(x_shape.GetDims().size() == 3, "input shape must be {batch, rows, cols}");
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  int64_t element_count = x_shape.Size();

  int batch = gsl::narrow<int>(x_shape.GetDims()[0]);
  int64_t rows = gsl::narrow<int>(x_shape.GetDims()[1]);
  int64_t cols = gsl::narrow<int>(x_shape.GetDims()[2]);
  int n1 = gsl::narrow<int>(x_shape.SizeToDimension(axis));
  int n2 = gsl::narrow<int>(x_shape.SizeFromDimension(axis));
  ORT_ENFORCE(n2 != 1, "n2 should not be 1");

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  auto stream = Stream();
  auto cublasLt = CublasLtHandle();
  auto Y_data = reinterpret_cast<CudaQ*>(Y->MutableData<int8_t>());
  const float* scale_x = ctx->Input<Tensor>(1)->Data<float>();
  const float* scale_y = ctx->Input<Tensor>(4)->Data<float>();

  // TODO: Write specific kernel for all these rather than QOrder/DQOrder
  auto fp16_buffer = GetScratchBuffer<CudaF>(2 * element_count * sizeof(MLFloat16));
  ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, batch, rows, cols, CUDA_R_8I, X_data, CUBLASLT_ORDER_COL32, fp16_buffer.get(), CUBLASLT_ORDER_ROW));
  CudaF half_scale_x = ToCudaType<MLFloat16>::FromFloat(*scale_x);
  ORT_RETURN_IF_ERROR(CudaDequantizeLinear(stream, (const CudaQ*)(fp16_buffer.get()), fp16_buffer.get() + element_count,
                                           &half_scale_x, (const int8_t*)nullptr, element_count));

  typedef typename ToCudaType<float>::MappedType CudaU;
  HostApplyLayerNorm<CudaF, CudaU, false>(GetDeviceProp(), Stream(), fp16_buffer.get(), nullptr, nullptr,
                                          fp16_buffer.get() + element_count, n1, n2, epsilon_, scale_data, bias_data);

  CudaF half_scale_y = ToCudaType<MLFloat16>::FromFloat(*scale_y);
  ORT_RETURN_IF_ERROR(CudaQuantizeLinear(stream, fp16_buffer.get(), (CudaQ*)(fp16_buffer.get() + element_count),
                                         &half_scale_y, (const int8_t*)nullptr, element_count));
  ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, batch, rows, cols, CUDA_R_8I,
                              fp16_buffer.get() + element_count, CUBLASLT_ORDER_ROW, Y_data, CUBLASLT_ORDER_COL32));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
