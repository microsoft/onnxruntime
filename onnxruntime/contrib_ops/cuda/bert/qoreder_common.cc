
#include "qorder_common.h"

#if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include "gsl/gsl"
#include "absl/cleanup/cleanup.h"

#include "core/common/common.h"
#include "core/common/type_list.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_types.h"

#include "core/providers/cuda/tensor/quantize_linear.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    QuantizeWithOrder,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("F", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>()),
    QuantizeWithOrder);

ONNX_OPERATOR_KERNEL_EX(
    DequantizeWithOrder,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("F", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>()),
    DequantizeWithOrder);

cublasLtOrder_t GetCublasLtOrderAttr(const OpKernelInfo& info, const char* order_attr) {
  int64_t order_value;
  Status status = info.GetAttr(order_attr, &order_value);
  ORT_ENFORCE(status.IsOK(), "Attribute ", order_attr, " is not set.");
  return gsl::narrow_cast<cublasLtOrder_t>(order_value);
}

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order) {
  switch (order) {
    case CUBLASLT_ORDER_ROW:
      return cols;
    case CUBLASLT_ORDER_COL:
      return rows;
    case CUBLASLT_ORDER_COL32:
      return 32 * rows;
    case CUBLASLT_ORDER_COL4_4R2_8C:
      return 32 * ((rows + 8 - 1) / 8) * 8;
    case CUBLASLT_ORDER_COL32_2R_4R4:
      return 32 * ((rows + 32 - 1) / 32) * 32;
    default:
      return 0;
  }
}

void UpdateTileRequire(cublasLtOrder_t order, int64_t& row_tile, int64_t& col_tile) {
  switch (order) {
    case CUBLASLT_ORDER_COL32:
      col_tile = max(col_tile, 32);
      break;
    case CUBLASLT_ORDER_COL4_4R2_8C:
      col_tile = max(col_tile, 32);
      row_tile = max(row_tile, 8);
      break;
    case CUBLASLT_ORDER_COL32_2R_4R4:
      col_tile = max(col_tile, 32);
      row_tile = max(row_tile, 32);
      break;
  }
}

static Status Reorder(cublasLtHandle_t cublasLt, cudaStream_t stream,
                      int32_t batchCount, int rows, int cols, cudaDataType_t data_type,
                      const void* input, cublasLtOrder_t order_input, void* output, cublasLtOrder_t order_input) {
  cublasLtMatrixTransformDesc_t transform_desc = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));
  absl::Cleanup clean_transform_desc = [&transform_desc](void) { cublasLtMatrixTransformDescDestroy(transform_desc); };

  cublasLtMatrixLayout_t InputLayout = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&InputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_input)));
  absl::Cleanup clean_input_layout = [&InputLayout]() { cublasLtMatrixLayoutDestroy(InputLayout); };
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_input_, sizeof(order_input)));

  cublasLtMatrixLayout_t OutputLayout = nullptr;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&OutputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_output)));
  absl::Cleanup clean_output_layout = [&OutputLayout]() { cublasLtMatrixLayoutDestroy(OutputLayout); };
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_output_, sizeof(order_output)));

  if (batchCount > 1) {
    int64_t batch_stride_input = Rows * Cols;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransform(cublasLt, transform_desc, &alpha, input, InputLayout,
                                                 &beta, nullptr, nullptr, output, OutputLayout, stream));

  return Status::OK();
};

static Status CheckTensorOrder(const Tensor& input_tensor, cublasLtOrder_t input_order, cublasLtOrder_t output_order,
                               int64_t& rows, int64_t& cols, int64_t& batchCount, int64_t& elementCount) {
  const auto dims = input_tensor.Shape().GetDims();
  cols = dims.back();
  rows = (dims.size() <= 1 ? 1LL : dims[dims.size() - 2]);
  batchCount = (dims.size() <= 2 ? 1LL : std::accumulate(dims.begin(), dims.begin() + (dims.size() - 2), 1LL, std::multiply<int64_t>()));
  elementCount = cols * rows * batchCount;
  int64_t row_tile = 1, col_tile = 1;
  UpdateTileRequire(input_order, row_tile, col_tile);
  UpdateTileRequire(output_order, row_tile, col_tile);
  if (rows % row_tile != 0 || cols % col_tile != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Shape not meet clean tile requirement!", dims);
  }
  return Status::OK();
}

QuantizeWithOrder::QuantizeWithOrder(const OpKernelInfo& info) : OpKernel(info) {
  order_input_ = GetOrderAttr(info, "order_input");
  order_output_ = GetOrderAttr(info, "order_output");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is support for order_input");
  ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_COL32 || order_output_ == CUBLASLT_ORDER_COL4_4R2_8C || order_output_ == CUBLASLT_ORDER_COL32_2R_4R4,
              "Only CUBLASLT_ORDER_COL32, CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4 is support for order_output");
}

DequantizeWithOrder::DequantizeWithOrder(const OpKernelInfo& info) : OpKernel(info) {
  order_input_ = GetOrderAttr(info, "order_input");
  order_output_ = GetOrderAttr(info, "order_output");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_COL32 || order_input_ == CUBLASLT_ORDER_COL4_4R2_8C || order_input_ == CUBLASLT_ORDER_COL32_2R_4R4,
              "Only CUBLASLT_ORDER_COL32, CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4 is support for order_input");
  ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is support for order_output");
}

Status QuantizeWithOrder::ComputeInternal(OpKernelContext* ctx) const {
  int64_t rows = 0, cols = 0, batch = 0, n = 0;

  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(input_tensor, order_input_, order_output_, rows, cols, batch, n));
  const void* scale = context->Input<Tensor>(1)->DataRaw();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cublasLtHandle_t cublasLt = cublasLtHandle();
  cudaStream_t stream = (cudaStream_t)GetComputeStream();

  // TODO: Currently use existing quantize kernel first, may merge into one kernel if performance needed
  auto q8_buffer = GetScratchBuffer<int8_t>(order_input_ == order_output_ ? 0 : n);
  int8_t* dst = order_input_ == order_output_ ? output_tensor->MutableData<int8_t>() : q8_buffer.Get();
  if (input_tensor.IsDataType<float>()) {
    ORT_RETURN_IF_ERROR(CudaQuantizeLinear(stream, input_tensor.template Data<float>(), dst, (const float*)scale, nullptr, n));
  }
  else {
    ORT_RETURN_IF_ERROR(CudaQuantizeLinear(stream, input_tensor.template Data<half>(), dst, (const half*)scale, nullptr, n));
  }

  if (order_input_ != order_output_) {
    ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, batchCount, rows, cols, CUDA_R_8I,
                                q8_buffer.Get(), order_input_, output_tensor->MutableDataRaw(), order_output_));
  }

  return Status::OK();
}

Status DequantizeWithOrder::ComputeInternal(OpKernelContext* ctx) const {
  int64_t rows = 0, cols = 0, batch = 0, n = 0;

  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(input_tensor, order_input_, order_output_, rows, cols, batch, n));
  const void* scale = context->Input<Tensor>(1)->DataRaw();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cublasLtHandle_t cublasLt = cublasLtHandle();
  cudaStream_t stream = (cudaStream_t)GetComputeStream();

  // TODO: Currently use existing quantize kernel first, may merge into one kernel if performance needed
  int64_t eff_n = input_tensor.IsDataType<float>() ? n : (n + 1) / 2;
  auto fp_buffer = GetScratchBuffer<float>(order_input_ == order_output_ ? 0 : eff_n);
  void* dst = order_input_ == order_output_ ? input_tensor->MutableDataRaw() : (void*)fp_buffer.Get();
  if (input_tensor.IsDataType<float>()) {
    ORT_RETURN_IF_ERROR(CudaDequantizeLinear(stream, input_tensor.template Data<int8_t>(), (float*)dst, (const float*)scale, nullptr, n));
  } else {
    ORT_RETURN_IF_ERROR(CudaDequantizeLinear(stream, input_tensor.template Data<int8_t>(), (half*)dst, (const half*)scale, nullptr, n));
  }

  if (order_input_ != order_output_) {
    ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, batch, rows, cols, input_tensor.IsDataType<float>() ? CUDA_R_32F : CUDA_R_16F,
                                fp_buffer.Get(), order_input_, output_tensor->MutableDataRaw(), order_output_));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
