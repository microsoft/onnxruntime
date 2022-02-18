
#include "qorder_common.h"

// #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include <numeric>
#include <functional>
#include <sstream>
#include "gsl/gsl"

#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include "qorder_common_impl.h"
#include "qorder_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

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

static void cublasLtMatMulInt8SetupAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo, int algoId, int swizzle,
                                        int customOption, int tile, int splitK_val, int reductionScheme, int stages) {
  cublasLtMatmulAlgoInit(cublasLt_handle, CUBLAS_COMPUTE_32I, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
}

static inline std::string AlgoKey(const cudaDeviceProp& /*device_prop*/,
                                  int batch_count, int m, int n, int k,
                                  cublasLtOrder_t order_weight, cublasLtOrder_t input_output_order) {
  std::stringstream ss;
  ss << batch_count << "-" << m << "_" << n << "_" << k << "-" << (int)order_weight << "-" << (int)input_output_order;
  return ss.str();
}

CublasLtMMAlgoMap& CublasLtMMAlgoMap::instance() {
  static CublasLtMMAlgoMap instance;
  return instance;
}

CublasLtMMAlgoMap::CublasLtMMAlgoMap() {
  // TODO: Load config file
  return;
}

void CublasLtMMAlgoMap::GetAlgo(cublasLtHandle_t cublasLt_handle, cublasLtMatmulAlgo_t& algo, const cudaDeviceProp& device_prop,
                                int batch_count, int m, int n, int k,
                                cublasLtOrder_t order_weight, cublasLtOrder_t input_output_order) const {
  std::string mark = AlgoKey(device_prop, batch_count, m, n, k, order_weight, input_output_order);
  auto algo_it = best_algos_.find(mark);
  if (algo_it != best_algos_.end() && algo_it->second.workspaceSize == 0) {
    const auto& algo_info = algo_it->second;
    cublasLtMatMulInt8SetupAlgo(cublasLt_handle, algo, algo_info.algoId, algo_info.swizzle, algo_info.customOption,
                                algo_info.tile, algo_info.splitK_val, algo_info.reductionScheme, algo_info.stages);
  } else {
    if (order_weight == CUBLASLT_ORDER_COL) {
      int algoId = 21;
      int stages = 0;
      cublasLtMatMulInt8SetupAlgo(cublasLt_handle, algo, algoId, 0, 0, 20, 0, 0, stages);
    } else {
      int algoId = (order_weight == CUBLASLT_ORDER_COL4_4R2_8C) ? 6 : 7 /* CUBLASLT_ORDER_COL32_2R_4R4 */;
      int stages = (order_weight == CUBLASLT_ORDER_COL4_4R2_8C) ? 13 : 15 /* CUBLASLT_ORDER_COL32_2R_4R4 */;
      cublasLtMatMulInt8SetupAlgo(cublasLt_handle, algo, algoId, 0, 0, 20, 0, 0, stages);
    }
  }
}

static Status CreateLtMatrixLayout(cublasLtMatrixLayout_t& layoutDesc,
                                   int const batchCount, int64_t const rowsAfterOp, int64_t const colsAfterOp,
                                   cudaDataType_t const matType, cublasLtOrder_t const matOrder, cublasOperation_t const matTrans) {
  if (matTrans == CUBLAS_OP_T) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&layoutDesc, matType, colsAfterOp, rowsAfterOp, CalcLeadingDimensionLt(colsAfterOp, rowsAfterOp, matOrder)));
  } else {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&layoutDesc, matType, rowsAfterOp, colsAfterOp, CalcLeadingDimensionLt(rowsAfterOp, colsAfterOp, matOrder)));
  }
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layoutDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &matOrder, sizeof(matOrder)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layoutDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
  if (batchCount > 1) {
    int64_t strideBatch = rowsAfterOp * colsAfterOp;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(layoutDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideBatch, sizeof(strideBatch)));
  }
  return Status::OK();
}

Status QOrdered_MatMul(cublasLtHandle_t cublasLt_handle, cudaStream_t stream, [[maybe_unused]] const cudaDeviceProp& device_prop,
                       int32_t batchCount, int64_t m, int64_t n, int64_t k,
                       const float* alpha, const int8_t* A, const int8_t* B, int32_t batchB,
                       const float* bias, const float* beta,
                       const int8_t* C, int32_t batchC,
                       int8_t* D,
                       cublasLtOrder_t order_weight,
                       cublasLtPointerMode_t pointer_mode) {
  const cublasOperation_t transpose_op = CUBLAS_OP_T;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  auto clean_matmul_desc = gsl::finally([&matmul_desc]() {if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc); });
  cublasLtMatrixLayout_t desc_A = nullptr;
  auto clean_desc_A = gsl::finally([&desc_A]() {if (desc_A) cublasLtMatrixLayoutDestroy(desc_A); });
  cublasLtMatrixLayout_t desc_B = nullptr;
  auto clean_desc_B = gsl::finally([&desc_B]() {if (desc_B) cublasLtMatrixLayoutDestroy(desc_B); });
  cublasLtMatrixLayout_t desc_C = nullptr;
  auto clean_desc_C = gsl::finally([&desc_C]() {if (desc_C) cublasLtMatrixLayoutDestroy(desc_C); });
  cublasLtMatrixLayout_t desc_D = nullptr;
  auto clean_desc_D = gsl::finally([&desc_D]() {if (desc_D) cublasLtMatrixLayoutDestroy(desc_D); });
  const float beta_zero = 0.0f;
  beta = (C == nullptr ? &beta_zero : beta);

  cublasLtMatmulAlgo_t algo;
  cublasLtOrder_t order_ACD = ((order_weight == CUBLASLT_ORDER_COL || order_weight == CUBLASLT_ORDER_ROW) ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_COL32);
  CublasLtMMAlgoMap::instance().GetAlgo(cublasLt_handle, algo, device_prop, batchCount, (int)m, (int)n, (int)k, order_weight, order_ACD);
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32F));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  if (order_weight == CUBLASLT_ORDER_COL) {
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_op, sizeof(transpose_op)));
    if (bias != nullptr) {
      cublasLtEpilogue_t epilogue_bias = CUBLASLT_EPILOGUE_BIAS;
      CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_bias, sizeof(epilogue_bias)));
      CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_A, batchCount, k, m, CUDA_R_8I, CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // for A'
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_B, batchB, k, n, CUDA_R_8I, CUBLASLT_ORDER_COL, CUBLAS_OP_N));      // For B
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_D, batchCount, n, m, CUDA_R_8I, CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // For D'
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_C, batchC, n, m, CUDA_R_8I, CUBLASLT_ORDER_COL, CUBLAS_OP_N));  // For C'
      CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc,
                                          alpha, B, desc_B, A, desc_A,
                                          beta, C == nullptr ? D : C, C == nullptr ? desc_D : desc_C,
                                          D, desc_D,
                                          &algo, nullptr, 0,  // algo, workspace, workspace_size
                                          stream));
  } else if (order_weight == CUBLASLT_ORDER_ROW) {
    ORT_ENFORCE(bias == nullptr);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_op, sizeof(transpose_op)));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_A, batchCount, m, k, CUDA_R_8I, CUBLASLT_ORDER_ROW, CUBLAS_OP_T));  // for A
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_B, batchB, k, n, CUDA_R_8I, CUBLASLT_ORDER_ROW, CUBLAS_OP_N));      // For B
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_D, batchCount, m, n, CUDA_R_8I, CUBLASLT_ORDER_ROW, CUBLAS_OP_N));  // For D
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_C, batchC, m, n, CUDA_R_8I, CUBLASLT_ORDER_ROW, CUBLAS_OP_N));  // For C
      CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc,
                                          alpha, A, desc_A, B, desc_B,
                                          beta, C == nullptr ? D : C, C == nullptr ? desc_D : desc_C,
                                          D, desc_D,
                                          nullptr, nullptr, 0,  // algo, workspace, workspace_size
                                          stream));
  } else {
    ORT_ENFORCE(bias == nullptr);
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_op, sizeof(transpose_op)));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_A, batchCount, m, k, CUDA_R_8I, CUBLASLT_ORDER_COL32, CUBLAS_OP_N));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_B, batchB, k, n, CUDA_R_8I, order_weight, CUBLAS_OP_T));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_B, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_D, batchCount, m, n, CUDA_R_8I, CUBLASLT_ORDER_COL32, CUBLAS_OP_N));
    if (C != nullptr) {
      ORT_RETURN_IF_ERROR(CreateLtMatrixLayout(desc_C, batchC, m, n, CUDA_R_8I, CUBLASLT_ORDER_COL32, CUBLAS_OP_N));
      CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(desc_C, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    }
    CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(cublasLt_handle, matmul_desc,
                                          alpha, A, desc_A, B, desc_B,
                                          beta, C == nullptr ? D : C, C == nullptr ? desc_D : desc_C,
                                          D, desc_D,
                                          &algo, nullptr, 0,  // algo, workspace, workspace_size
                                          stream));
  }

  return Status::OK();
}

Status Reorder(cublasLtHandle_t cublasLt, cudaStream_t stream, const cudaDeviceProp& device_prop,
               int32_t batchCount, int64_t rows, int64_t cols, cudaDataType_t data_type,
               const void* input, cublasLtOrder_t order_input, void* output, cublasLtOrder_t order_output) {
  if (data_type == CUDA_R_8I && order_input == CUBLASLT_ORDER_ROW && order_output == CUBLASLT_ORDER_COL32) {
    ReorderS8RowToCol32(stream, device_prop, (const int8_t*)input, (int8_t*)output,
                        (unsigned)batchCount, gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols));
    return Status::OK();
  }

  cublasLtMatrixTransformDesc_t transform_desc = nullptr;
  auto clean_transform_desc = gsl::finally([&transform_desc]() {if (transform_desc) cublasLtMatrixTransformDescDestroy(transform_desc); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32I));

  cublasLtMatrixLayout_t InputLayout = nullptr;
  auto clean_InputLayout = gsl::finally([&InputLayout]() {if (InputLayout) cublasLtMatrixLayoutDestroy(InputLayout); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&InputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_input)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_input, sizeof(order_input)));

  cublasLtMatrixLayout_t OutputLayout = nullptr;
  auto clean_OutputLayout = gsl::finally([&OutputLayout]() {if (OutputLayout) cublasLtMatrixLayoutDestroy(OutputLayout); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&OutputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_output)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_output, sizeof(order_output)));

  if (batchCount > 1) {
    int64_t batch_stride_input = rows * cols;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
  }

  int32_t alpha = 1;
  int32_t beta = 0;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransform(cublasLt, transform_desc, &alpha, input, InputLayout,
                                                 &beta, nullptr, nullptr, output, OutputLayout, stream));

  return Status::OK();
};

static Status CheckTensorOrder(const Tensor& input_tensor, cublasLtOrder_t input_order, cublasLtOrder_t output_order,
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

QuantizeWithOrder::QuantizeWithOrder(const OpKernelInfo& info) : CudaKernel(info) {
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_ROW,
              "Only CUBLASLT_ORDER_ROW is supported for order_input");
}

DequantizeWithOrder::DequantizeWithOrder(const OpKernelInfo& info) : CudaKernel(info) {
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

QOrderedMatMul::QOrderedMatMul(const OpKernelInfo& info) : CudaKernel(info) {
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
}

Status QOrderedMatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                               /*out*/ bool& is_packed,
                               /*out*/ PrePackedWeights* /* prepacked_weights */) {
  is_packed = false;
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

  return Status::OK();
}

Status QuantizeWithOrder::ComputeInternal(OpKernelContext* context) const {
  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

  int64_t rows = 0, cols = 0, batch = 0, n = 0;
  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(input_tensor, (cublasLtOrder_t)order_input_, (cublasLtOrder_t)order_output_, rows, cols, batch, n));
  const float* scale = context->Input<Tensor>(1)->Data<float>();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cublasLtHandle_t cublasLt = CublasLtHandle();
  cudaStream_t stream = Stream();
  const auto& device_prop = GetDeviceProp();

  // Note that order_input_ == CUBLASLT_ORDER_ROW
  if (order_output_ == CUBLASLT_ORDER_COL32) {
    if (input_tensor.IsDataType<MLFloat16>()) {
      QOrderQuantizeRowToCol32(stream, device_prop, (const __half*)input_tensor.Data<MLFloat16>(), output_tensor->MutableData<int8_t>(),
                               *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols));
    } else {
      QOrderQuantizeRowToCol32(stream, device_prop, input_tensor.Data<float>(), output_tensor->MutableData<int8_t>(),
                               *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols));
    }
  } else {
    auto q8_buffer = GetScratchBuffer<int8_t>(order_input_ == order_output_ ? 0LL : n);
    int8_t* dst = (order_input_ == order_output_ ? output_tensor->MutableData<int8_t>() : q8_buffer.get());
    if (input_tensor.IsDataType<MLFloat16>()) {
      QOrderQuantize_Strict(stream, device_prop, (const __half*)input_tensor.Data<MLFloat16>(), dst, *scale, n);
    } else {
      QOrderQuantize(stream, device_prop, input_tensor.Data<float>(), dst, *scale, n);
    }

    if (order_input_ != order_output_) {
      ORT_RETURN_IF_ERROR(Reorder(cublasLt, stream, device_prop, gsl::narrow<int>(batch), rows, cols, CUDA_R_8I,
                                  q8_buffer.get(), (cublasLtOrder_t)order_input_, output_tensor->MutableDataRaw(), (cublasLtOrder_t)order_output_));
    }
  }

  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();
  return Status::OK();
}

Status DequantizeWithOrder::ComputeInternal(OpKernelContext* context) const {
  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

  int64_t rows = 0, cols = 0, batch = 0, n = 0;

  const Tensor& input_tensor = *context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(CheckTensorOrder(input_tensor, (cublasLtOrder_t)order_output_, (cublasLtOrder_t)order_output_, rows, cols, batch, n));
  const Tensor& scale_tensor = *context->Input<Tensor>(1);
  const float* scale = scale_tensor.Data<float>();
  Tensor* output_tensor = context->Output(0, input_tensor.Shape());
  cudaStream_t stream = Stream();
  const auto& device_prop = GetDeviceProp();

  // Note that order_output_ == CUBLASLT_ORDER_ROW
  if (order_input_ == CUBLASLT_ORDER_COL32) {
    if (output_tensor->IsDataType<MLFloat16>()) {
      QOrderDequantizeCol32ToRow(stream, device_prop, input_tensor.Data<int8_t>(), (__half*)output_tensor->MutableData<MLFloat16>(),
                                 *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols));
    } else {
      QOrderDequantizeCol32ToRow(stream, device_prop, input_tensor.Data<int8_t>(), output_tensor->MutableData<float>(),
                                 *scale, gsl::narrow<unsigned>(batch), gsl::narrow<unsigned>(rows), gsl::narrow<unsigned>(cols));
    }
  } else {
    if (output_tensor->IsDataType<MLFloat16>()) {
      QOrderDequantize_Strict(stream, device_prop, input_tensor.Data<int8_t>(), (__half*)output_tensor->MutableData<MLFloat16>(), *scale, n);
    } else {
      QOrderDequantize(stream, device_prop, input_tensor.Data<int8_t>(), output_tensor->MutableData<float>(), *scale, n);
    }
  }

  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();
  return Status::OK();
}

Status QOrderedMatMul::ComputeInternal(OpKernelContext* context) const {
  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

  int64_t rowsA = 0, colsA = 0, batchA = 1, elementsA = 0;
  int64_t rowsB = 0, colsB = 0, batchB = 1, elementsB = 0;
  int64_t rowsC = 0, colsC = 0, batchC = 1, elementsC = 0;

  const Tensor& tensor_A = *context->Input<Tensor>(0);
  const Tensor& tensor_B = *context->Input<Tensor>(2);

  // Support General case only. No broadcasting, is handled now.
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
    pointer_mode = (cublasLtPointerMode_t)4; // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST; weired compilation error report
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

  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// #endif