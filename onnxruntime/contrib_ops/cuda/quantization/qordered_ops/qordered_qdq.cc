// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_shape.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"
#include "gsl/gsl"

#include <numeric>
#include <functional>
#include <sstream>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QuantizeWithOrder,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("F", BuildKernelDefConstraints<float, MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // scale_A
    QuantizeWithOrder);

ONNX_OPERATOR_KERNEL_EX(
    DequantizeWithOrder,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", BuildKernelDefConstraints<float, MLFloat16>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // scale_A
    DequantizeWithOrder);

void UpdateTileRequire(cublasLtOrder_t order, int64_t& row_tile, int64_t& col_tile) {
  switch (order) {
    case CUBLASLT_ORDER_ROW:
    case CUBLASLT_ORDER_COL:
      break;
    case CUBLASLT_ORDER_COL32:
      col_tile = std::max(col_tile, int64_t{32});
      break;
    case CUBLASLT_ORDER_COL4_4R2_8C:
      col_tile = std::max(col_tile, int64_t{32});
      row_tile = std::max(row_tile, int64_t{8});
      break;
    case CUBLASLT_ORDER_COL32_2R_4R4:
      col_tile = std::max(col_tile, int64_t{32});
      row_tile = std::max(row_tile, int64_t{32});
      break;
  }
}

Status CheckTensorOrder(const Tensor& input_tensor, cublasLtOrder_t input_order, cublasLtOrder_t output_order,
                        int64_t& rows, int64_t& cols, int64_t& batchCount, int64_t& elementCount) {
  const auto dims = input_tensor.Shape().GetDims();
  cols = dims.back();
  rows = (dims.size() <= 1 ? 1LL : dims[dims.size() - 2]);
  batchCount = (dims.size() <= 2 ? 1LL : std::accumulate(dims.begin(), dims.begin() + (dims.size() - 2), 1LL, std::multiplies<int64_t>()));
  elementCount = cols * rows * batchCount;
  int64_t row_tile = 1, col_tile = 1;
  UpdateTileRequire(input_order, row_tile, col_tile);
  UpdateTileRequire(output_order, row_tile, col_tile);
  if (rows % row_tile != 0 || cols % col_tile != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Shape not meet clean tile requirement!", dims);
  }
  return Status::OK();
}

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr) {
  int64_t order_value;
  Status status = info.GetAttr(order_attr, &order_value);
  ORT_ENFORCE(status.IsOK(), "Attribute ", order_attr, " is not set.");
  return gsl::narrow<cublasLtOrder_t>(order_value);
}

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr,
                                     int num_allowed_orders, const cublasLtOrder_t* orders_allowed, const char* error_msg) {
  cublasLtOrder_t order = GetCublasLtOrderAttr(info, order_attr);
  ORT_ENFORCE(std::any_of(orders_allowed, orders_allowed + num_allowed_orders,
                          [order](cublasLtOrder_t allowed_order) { return allowed_order == order; }),
              error_msg);
  return order;
}

QuantizeWithOrder::QuantizeWithOrder(const OpKernelInfo& info) : CudaKernel(info) {
  int cuda_runtime_version = 0;
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&cuda_runtime_version));
  ORT_ENFORCE(cuda_runtime_version >= 11040, "QOrderedMatmul need cuda runtime higher than 11.4");

  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_ROW,
              "Only CUBLASLT_ORDER_ROW is supported for order_input");
}

DequantizeWithOrder::DequantizeWithOrder(const OpKernelInfo& info) : CudaKernel(info) {
  int cuda_runtime_version = 0;
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&cuda_runtime_version));
  ORT_ENFORCE(cuda_runtime_version >= 11040, "QOrderedMatmul need cuda runtime higher than 11.4");

  int64_t to_type = 0;
  Status status = info.GetAttr("to", &to_type);
  ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
  ORT_ENFORCE(to_type == onnx::TensorProto_DataType_FLOAT16 || to_type == onnx::TensorProto_DataType_FLOAT,
              "Attribute to only support float(", onnx::TensorProto_DataType_FLOAT, ") or float16(", onnx::TensorProto_DataType_FLOAT16, ").");
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW,
              "Only CUBLASLT_ORDER_ROW are supported for order_output");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_COL32 || order_input_ == CUBLASLT_ORDER_ROW,
              "Only CUBLASLT_ORDER_COL32 or CUBLASLT_ORDER_ROW is supported for order_input");
}

Status QuantizeWithOrder::ComputeInternal(OpKernelContext* context) const {
  int64_t rows = 0, cols = 0, batch = 0, n = 0;
  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(
      input_tensor, (cublasLtOrder_t)order_input_, (cublasLtOrder_t)order_output_, rows, cols, batch, n));
  const float* scale = context->Input<Tensor>(1)->Data<float>();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream(context);
  const auto& device_prop = GetDeviceProp();

  // Note that order_input_ == CUBLASLT_ORDER_ROW
  if (order_output_ == CUBLASLT_ORDER_COL32) {
    if (input_tensor.IsDataType<MLFloat16>()) {
      ORT_RETURN_IF_ERROR(QOrderQuantizeRowToCol32(
          stream, device_prop, (const __half*)input_tensor.Data<MLFloat16>(), output_tensor->MutableData<int8_t>(),
          *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols)));
    } else {
      ORT_RETURN_IF_ERROR(QOrderQuantizeRowToCol32(
          stream, device_prop, input_tensor.Data<float>(), output_tensor->MutableData<int8_t>(),
          *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols)));
    }
  } else {
    auto q8_buffer = GetScratchBuffer<int8_t>(order_input_ == order_output_ ? 0LL : n, context->GetComputeStream());
    int8_t* dst = (order_input_ == order_output_ ? output_tensor->MutableData<int8_t>() : q8_buffer.get());
    if (input_tensor.IsDataType<MLFloat16>()) {
      ORT_RETURN_IF_ERROR(QOrderQuantize_Strict(stream, device_prop, (const __half*)input_tensor.Data<MLFloat16>(), dst, *scale, n));
    } else {
      ORT_RETURN_IF_ERROR(QOrderQuantize(stream, device_prop, input_tensor.Data<float>(), dst, *scale, n));
    }

    if (order_input_ != order_output_) {
      ORT_RETURN_IF_ERROR(Reorder(
          cublasLt, stream, device_prop, gsl::narrow<int>(batch), rows, cols, CUDA_R_8I,
          q8_buffer.get(), (cublasLtOrder_t)order_input_, output_tensor->MutableDataRaw(), (cublasLtOrder_t)order_output_));
    }
  }

  return Status::OK();
}

Status DequantizeWithOrder::ComputeInternal(OpKernelContext* context) const {
  int64_t rows = 0, cols = 0, batch = 0, n = 0;

  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(
      input_tensor, (cublasLtOrder_t)order_output_, (cublasLtOrder_t)order_output_, rows, cols, batch, n));
  const Tensor& scale_tensor = *context->Input<Tensor>(1);
  const float* scale = scale_tensor.Data<float>();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cudaStream_t stream = Stream(context);
  const auto& device_prop = GetDeviceProp();

  // Note that order_output_ == CUBLASLT_ORDER_ROW
  if (order_input_ == CUBLASLT_ORDER_COL32) {
    if (output_tensor->IsDataType<MLFloat16>()) {
      ORT_RETURN_IF_ERROR(QOrderDequantizeCol32ToRow(
          stream, device_prop, input_tensor.Data<int8_t>(), (__half*)output_tensor->MutableData<MLFloat16>(),
          *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols)));
    } else {
      ORT_RETURN_IF_ERROR(QOrderDequantizeCol32ToRow(
          stream, device_prop, input_tensor.Data<int8_t>(), output_tensor->MutableData<float>(),
          *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols)));
    }
  } else {
    if (output_tensor->IsDataType<MLFloat16>()) {
      ORT_RETURN_IF_ERROR(QOrderDequantize_Strict(
          stream, device_prop, input_tensor.Data<int8_t>(), (__half*)output_tensor->MutableData<MLFloat16>(), *scale, n));
    } else {
      ORT_RETURN_IF_ERROR(QOrderDequantize(
          stream, device_prop, input_tensor.Data<int8_t>(), output_tensor->MutableData<float>(), *scale, n));
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
