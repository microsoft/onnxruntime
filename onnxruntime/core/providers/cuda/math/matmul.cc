// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"

#include <iostream>
#include <iomanip>

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_KERNEL_TYPED(BFloat16)
#endif

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  int64_t left_k = transa ? left_shape[left_num_dims - 2] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_num_dims - 2];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_num_dims - 2];
  int64_t m = transb ? right_shape[right_num_dims - 2] : right_shape[right_num_dims - 1];
  stride_A = n * left_k;
  stride_B = right_num_dims == 2 ? 0 : right_k * m;
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

// We can test with float only here
//void DumpMatrixImpl(const std::string& name, const float* src, int row, int col, int offset, int col_width) {
//  std::cout << "Dump matrix: " << name << std::endl;
//
//  if (col_width == -1) col_width = col;
//
//  for (int r = 0; r < row; r++) {
//    for (int c = 0; c < col; c++) {
//      int index = r * col_width + offset + c;
//      std::cout << std::setw(12) << std::setprecision(8) << src[index];
//    }
//    std::cout << std::endl;
//  }
//  std::cout << std::endl;
//}

template <typename T>
std::ostream& DumpArrayImpl(std::ostream& os, const std::string& name, const void* in, size_t len, size_t col_width) {
  std::unique_ptr<T[]> buf(new T[len]);
  cudaMemcpy(buf.get(), in, len * sizeof(T), cudaMemcpyDeviceToHost);
  const T* src = buf.get();

  std::cout << "Dump array: " << name << std::endl;

  if (col_width == -1) col_width = len;

  for (size_t i = 0; i < len;) {
    for (size_t w = 0; w < col_width && i < len; ++w, ++i) {
      os << std::setw(12) << std::setprecision(8) << src[i];
    }
    os << std::endl;
  }
  os << std::endl;
  return os;
}

#define DUMP_ARRAY(T, ...) DumpArrayImpl<T>(__VA_ARGS__)

#ifdef USE_CUSPARSE_LT

static Status MakeDescriptors(const cusparseLtHandle_t* handle, cudaDataType cuda_type, size_t data_type_size, cusparseComputeType precision,
                              int64_t m, int64_t k, int64_t n, bool transa, bool transb,
                              cusparseLtMatDescriptor_t& mat_A, cusparseLtMatDescriptor_t& mat_B,
                              cusparseLtMatDescriptor_t& mat_C, cusparseLtMatmulDescriptor_t& mat_mul) {
  // Switch K and M as we are feeding them swapped. This argument swapping will only be available in the
  // next release of the library.
  CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_A, k, m,
                                                           (transa) ? m : k,
                                                           static_cast<uint32_t>(data_type_size), cuda_type,
                                                           CUSPARSE_ORDER_COL));

  CUSPARSELT_RETURN_IF_ERROR(cusparseLtStructuredDescriptorInit(handle, &mat_B, n, k,
                                                                (transb) ? n : k,
                                                                static_cast<uint32_t>(data_type_size), cuda_type,
                                                                CUSPARSE_ORDER_COL, CUSPARSELT_SPARSITY_50_PERCENT));

  CUSPARSELT_RETURN_IF_ERROR(cusparseLtDenseDescriptorInit(handle, &mat_C, n, m,
                                                           n, static_cast<uint32_t>(data_type_size), cuda_type,
                                                           CUSPARSE_ORDER_COL));

  cusparseOperation_t transA = transa ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = transb ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulDescriptorInit(handle, &mat_mul,
                                                            transB,
                                                            transA,
                                                            &mat_B,
                                                            &mat_A,
                                                            &mat_C,
                                                            &mat_C,
                                                            precision));

  return Status::OK();
}

/// <summary>
/// This class contains helper methods to deal with 2:4 sparse data by means
/// of cuSparseLT library.
/// </summary>

template <typename T>
class Sparse2x4ComputeHelper {
 public:
  Sparse2x4ComputeHelper() = default;
  ~Sparse2x4ComputeHelper() = default;
  /// <summary>
  /// Creates necessary descriptors and copies right tensor data to GPU and compressed it
  /// Prepack() has already verified that this data is a valid 2:4 format
  /// </summary>
  /// <param name="helper"></param>
  /// <param name="kernel"></param>
  /// <param name="transa"></param>
  /// <param name="transb"></param>
  /// <param name="right">2:4 initializer data</param>
  /// <returns>status</returns>
  onnxruntime::Status Compute(const CudaKernel* kernel, const SparseInfo& sparse_info,
                              const MatMulComputeHelper& helper, float alpha, bool transa, bool transb,
                              const Tensor* left, Tensor* Y) const {
    constexpr size_t data_type_size = sizeof(T);
    constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
    constexpr auto cuda_precision = ToCudaTypeEnum<T>::at_least_precision;
    const cusparseLtHandle_t* handle = &*sparse_info.handle_lt_;

    const int64_t m = helper.M();
    const int64_t k = helper.K();
    const int64_t n = helper.N();

    const int64_t sparse_size = sparse_info.shape_.Size();
    ORT_ENFORCE(sparse_size == n * k, "Sparse initializer shape size does not match computed K*N");

    cusparseLtMatDescriptor_t mat_A_desc;
    cusparseLtMatDescriptor_t mat_B_desc;
    cusparseLtMatDescriptor_t mat_C_desc;
    cusparseLtMatmulDescriptor_t mat_mul_desc;
    ORT_RETURN_IF_ERROR(MakeDescriptors(handle, cuda_type, data_type_size, cuda_precision, m, n, k,
                                        transa, transb, mat_A_desc, mat_B_desc, mat_C_desc, mat_mul_desc));

    cusparseLtMatmulAlgSelection_t alg_selection;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulAlgSelectionInit(handle, &alg_selection, &mat_mul_desc, CUSPARSELT_MATMUL_ALG_DEFAULT));

    int alg_id = 0;  // set algorithm ID
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulAlgSetAttribute(handle, &alg_selection,
                                                               CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                                               &alg_id, sizeof(alg_id)));

    size_t workspace_size;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulGetWorkspace(handle, &alg_selection, &workspace_size));
    auto workspace_buffer = kernel->GetScratchBuffer<T>(workspace_size);

    auto plan_destroy = [](const cusparseLtMatmulPlan_t* p) { cusparseLtMatmulPlanDestroy(p); };
    cusparseLtMatmulPlan_t plan;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmulPlanInit(handle, &plan, &mat_mul_desc, &alg_selection, workspace_size));
    std::unique_ptr<cusparseLtMatmulPlan_t, decltype(plan_destroy)> plan_guard(&plan, plan_destroy);

    size_t compressed_size;  // bytes
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompressedSize(handle, &plan, &compressed_size));
    size_t num_compressed_elements = compressed_size / data_type_size;
    if ((num_compressed_elements * data_type_size) < compressed_size) {
      num_compressed_elements++;
    }
    auto compressed_buffer = kernel->GetScratchBuffer<T>(num_compressed_elements);

    const float beta = 0.0f;
    cudaStream_t* streams = nullptr;
    int64_t stride_A, stride_B, stride_C, batch_count;
    // Batches
    if (helper.OutputOffsets().size() == 1) {
      // No batches, we compress the whole buffer as a single matrix
      CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompress(handle, &plan, sparse_info.prepack_buffers_.front().get(),
                                                         compressed_buffer.get(), nullptr /* default stream */));

      // We swapping arguments in hopes that the next release of the library supports the feature
      auto* output = Y->MutableData<T>();
      CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmul(handle, &plan, &alpha, compressed_buffer.get(), left->Data<T>(),
                                                  &beta, output, output, workspace_buffer.get(), streams, 0));
      return Status::OK();
    } else if (CanUseStridedBatchedGemm(left->Shape(), sparse_info.shape_,
                                        transa, transb, stride_A, stride_B, stride_C, batch_count)) {
      // XXX: Consider parallelizing it
      const auto* a_data = left->Data<T>();
      const auto* b_data = reinterpret_cast<const T*>(sparse_info.prepack_buffers_.front().get());
      auto* y_data = Y->MutableData<T>();

      // compress once
      if (stride_B == 0) {
        CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompress(handle, &plan, b_data,
                                                           compressed_buffer.get(), nullptr /* default stream */));
      }

      for (int64_t batch = 0; batch < batch_count; batch++) {
        // Compress if needed and compute
        if (stride_B > 0) {
          CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompress(handle, &plan, b_data,
                                                             compressed_buffer.get(), nullptr /* default stream */));
        }
        CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmul(handle, &plan, &alpha, compressed_buffer.get(), a_data,
                                                    &beta, y_data, y_data, workspace_buffer.get(), streams, 0));
        a_data += stride_A;
        b_data += stride_B;
        y_data += stride_C;
      }
    } else {
      ORT_ENFORCE(helper.LeftOffsets().size() == helper.RightOffsets().size(), "Left and right have different number of offsets");
      ORT_ENFORCE(helper.RightOffsets().size() == helper.OutputOffsets().size(), "Right and Output have different number of offsets");
      std::vector<const T*> left_arrays(helper.LeftOffsets().size());
      std::vector<const T*> right_arrays(helper.RightOffsets().size());
      std::vector<T*> output_arrays(helper.OutputOffsets().size());

      MatMulComputeHelper::OffsetToArrays(left->template Data<T>(), helper.LeftOffsets(), gsl::make_span(left_arrays));
      MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const T*>(sparse_info.prepack_buffers_.front().get()), helper.RightOffsets(),
                                          gsl::make_span(right_arrays));
      MatMulComputeHelper::OffsetToArrays(Y->template MutableData<T>(), helper.OutputOffsets(), gsl::make_span(output_arrays));

      // XXX: Consider parallelizing it
      batch_count = helper.OutputOffsets().size();
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        const T* a_data = left_arrays[batch];
        const T* b_data = right_arrays[batch];
        T* y_data = output_arrays[batch];
        CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMACompress(handle, &plan, b_data,
                                                           compressed_buffer.get(), nullptr /* default stream */));
        // Swapped arguments not supported at this time.
        CUSPARSELT_RETURN_IF_ERROR(cusparseLtMatmul(handle, &plan, &alpha, compressed_buffer.get(), a_data,
                                                    &beta, y_data, y_data, workspace_buffer.get(), streams, 0));
      }
    }

    return Status::OK();
  }

  /// <summary>
  /// This method validates constant initializer to be a valid 2:4 sparse data
  /// It creates fake A and C descriptors, Matmul descriptor and calls cusparseLtSpMMAPruneCheck()
  /// to validate the initializer. If the initializer has more than 2 dimensions, it is flattened.
  /// If it has only one dimension, a is appended to its shape.
  ///  See https://github.com/NVIDIA/CUDALibrarySamples/issues/19
  /// </summary>
  static onnxruntime::Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& param,
                                     bool transA, bool transB, std::unique_ptr<SparseInfo>& sparse_info, bool& is_packed) {
    is_packed = false;

    if (!tensor.IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name.get() + " : wrong data type for the constant initializer");
    }

    const auto& right_shape = tensor.Shape();

    int64_t M = 2;
    const auto right_num_dims = right_shape.NumDimensions();
    int64_t K = 0;
    int64_t N = 0;
    if (right_num_dims >= 2) {
      // Flatten the initializer to 2 - D
      K = right_shape.SizeToDimension(right_shape[right_num_dims - 1]);
      N = right_shape[right_num_dims - 1];
    } else {
      K = right_shape[0];
      N = 1;
    }

    TensorShape left_shape({M, K});

    constexpr auto cuda_type = ToCudaTypeEnum<T>::type;
    constexpr auto cuda_precision = ToCudaTypeEnum<T>::at_least_precision;
    constexpr size_t data_type_size = sizeof(T);

    auto destroy_cusparse_fn = [](cusparseLtHandle_t* handle) { cusparseLtDestroy(handle); };
    cusparseLtHandle_t handle;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtInit(&handle));
    std::unique_ptr<cusparseLtHandle_t, decltype(destroy_cusparse_fn)> handle_guard(&handle, destroy_cusparse_fn);

    // A and C are fake for validation purpose
    // https://github.com/NVIDIA/CUDALibrarySamples/issues/19
    cusparseLtMatDescriptor_t mat_A_desc;
    cusparseLtMatDescriptor_t mat_B_desc;
    cusparseLtMatDescriptor_t mat_C_desc;
    cusparseLtMatmulDescriptor_t mat_mul_desc;

    ORT_RETURN_IF_ERROR(MakeDescriptors(&handle, cuda_type, data_type_size, cuda_precision, M, K, N,
                                        transA, transB, mat_A_desc, mat_B_desc, mat_C_desc, mat_mul_desc));

    // Initializer tensors are stored on host, copy them for validation
    const auto data_size = right_shape.Size();
    auto device_buffer = kernel->GetPersistentBuffer<uint8_t>(data_size * data_type_size);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(device_buffer.get(), tensor.DataRaw(), data_size * data_type_size, cudaMemcpyHostToDevice));

    int valid = 1;
    CUSPARSELT_RETURN_IF_ERROR(cusparseLtSpMMAPruneCheck(&handle, &mat_mul_desc,
                                                         device_buffer.get(),
                                                         &valid,
                                                         static_cast<cudaStream_t>(0)));

    if (valid == 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name.get() + " : 2:4 data format validation failed");
    }

    sparse_info = onnxruntime::make_unique<SparseInfo>(param, right_shape);
    sparse_info->prepack_buffers_.push_back(std::move(device_buffer));
    sparse_info->handle_lt_ = onnxruntime::make_optional<cusparseLtHandle_t>(handle);
    handle_guard.relese();

    is_packed = true;
    return Status::OK();
  }
};

#endif  // USE_CUSPARSE_LT

template <class T>
class CuSparseHelper {
 public:
  static Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& param,
                        bool transa, bool transb, std::unique_ptr<SparseInfo>& sparse_info, bool& is_packed) {
    is_packed = false;
    if (!tensor.IsDataType<T>()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name.get() + " : wrong data type for the constant initializer");
    }

    const auto cuda_type = ToCudaTypeEnum<T>::type;

    auto close_dense_fn = [](cusparseDnMatDescr_t* desc) { cusparseDestroyDnMat(*desc); };
    auto close_sparse_fn = [](cusparseSpMatDescr_t* desc) { cusparseDestroySpMat(*desc); };
    auto destroy_cusparse_fn = [](cusparseHandle_t* desc) { cusparseDestroy(*desc); };

    // We do not want to use per-thread handles but persists between the runs.
    cusparseHandle_t handle;
    CUSPARSE_RETURN_IF_ERROR(cusparseCreate(&handle));
    std::unique_ptr<cusparseHandle_t, decltype(destroy_cusparse_fn)> handle_guard(&handle, destroy_cusparse_fn);

    // XXX: Currently support only 2 and 3-Dim Matrices
    const auto& right_shape = tensor.Shape();
    const auto element_size = tensor.DataType()->Size();
    const auto num_elements = right_shape.Size();
    const auto right_num_dims = right_shape.NumDimensions();
    int64_t K = 0;
    int64_t N = 0;
    if (right_num_dims >= 2) {
      K = right_shape[right_num_dims - 2];
      N = right_shape[right_num_dims - 1];
    } else {
      K = right_shape[0];
      N = 1;
    }

    auto values_buffer = kernel->GetScratchBuffer<T>(num_elements);
    CUDA_RETURN_IF_ERROR(cudaMemcpy(values_buffer.get(), tensor.DataRaw(), element_size * num_elements, cudaMemcpyHostToDevice));

    cusparseDnMatDescr_t dense_desc;
    // Feed column order and swap dims
    CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&dense_desc,
                                                 N,  // Number of rows in B(T)
                                                 K,  // Number of columns in B(T)
                                                 transb ? K : N,
                                                 values_buffer.get(),
                                                 cuda_type,
                                                 CUSPARSE_ORDER_COL));

    std::unique_ptr<cusparseDnMatDescr_t, decltype(close_dense_fn)> dense_guard(&dense_desc, close_dense_fn);

    cusparseSpMatDescr_t sparse_desc;
    std::unique_ptr<cusparseSpMatDescr_t, decltype(close_sparse_fn)> sparse_guard(nullptr, close_sparse_fn);

    auto sp_info = onnxruntime::make_unique<SparseInfo>(param, right_shape);

    if (param.UseCsrFormat() || param.UseCooFormat()) {
      // This will have data transposed, swap dims
      if (param.UseCsrFormat()) {
        CUSPARSE_RETURN_IF_ERROR(cusparseCreateCsr(&sparse_desc,
                                                   N,
                                                   K,
                                                   0,  // nnz is zero now
                                                   nullptr,
                                                   nullptr,                                 // colInd is null according to the example
                                                   nullptr,                                 // values is null according to the example
                                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,  // indicies are int
                                                   CUSPARSE_INDEX_BASE_ZERO, cuda_type));
      } else {
        CUSPARSE_RETURN_IF_ERROR(cusparseCreateCoo(&sparse_desc,
                                                   N,
                                                   K,
                                                   0,  // nnz
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   CUSPARSE_INDEX_32I,  // indicies are int
                                                   CUSPARSE_INDEX_BASE_ZERO, cuda_type));
      }
      // Guard against errors
      sparse_guard.reset(&sparse_desc);

      size_t buffer_size = 0;
      CUSPARSE_RETURN_IF_ERROR(cusparseDenseToSparse_bufferSize(
          handle, dense_desc, sparse_desc,
          CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
          &buffer_size));

      auto work_buffer = kernel->GetScratchBuffer<int8_t>(buffer_size);
      CUSPARSE_RETURN_IF_ERROR(cusparseDenseToSparse_analysis(handle, dense_desc, sparse_desc,
                                                              CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                              work_buffer.get()));

      int64_t rows_tmp, cols_tmp, nnz;
      CUSPARSE_RETURN_IF_ERROR(cusparseSpMatGetSize(sparse_desc, &rows_tmp, &cols_tmp,
                                                    &nnz));

      std::cout << "rows_tmp: " << rows_tmp << " cols_tmp: "
                << cols_tmp << " nnz: " << nnz << std::endl;

      if (param.UseCsrFormat()) {
        auto csr_offsets = kernel->GetPersistentBuffer<uint8_t>((K + 1) * sizeof(int));
        auto csr_cols = kernel->GetPersistentBuffer<uint8_t>(nnz * sizeof(int));
        auto csr_values = kernel->GetPersistentBuffer<uint8_t>(nnz * element_size);
        CUSPARSE_RETURN_IF_ERROR(cusparseCsrSetPointers(sparse_desc, csr_offsets.get(), csr_cols.get(),
                                                        csr_values.get()));
        sp_info->prepack_buffers_.push_back(std::move(csr_values));
        sp_info->prepack_buffers_.push_back(std::move(csr_offsets));
        sp_info->prepack_buffers_.push_back(std::move(csr_cols));
      } else {
        auto coo_row_ind = kernel->GetPersistentBuffer<uint8_t>(nnz * sizeof(int));
        auto coo_col_ind = kernel->GetPersistentBuffer<uint8_t>(nnz * sizeof(int));
        auto coo_values = kernel->GetPersistentBuffer<uint8_t>(nnz * element_size);
        CUSPARSE_RETURN_IF_ERROR(cusparseCooSetPointers(sparse_desc, coo_row_ind.get(),
                                                        coo_col_ind.get(), coo_values.get()));
        sp_info->prepack_buffers_.push_back(std::move(coo_row_ind));
        sp_info->prepack_buffers_.push_back(std::move(coo_col_ind));
        sp_info->prepack_buffers_.push_back(std::move(coo_values));
      }

      CUSPARSE_RETURN_IF_ERROR(cusparseDenseToSparse_convert(handle, dense_desc, sparse_desc,
                                                             CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                             work_buffer.get()));

      // XXX: Print all the buffers
      const auto& bufs = sp_info->prepack_buffers_;
      if (param.UseCsrFormat()) {
        DUMP_ARRAY(int, std::cout, "csr_offsets", bufs[0].get(), K + 1, 10);
        DUMP_ARRAY(int, std::cout, "csr_cols", bufs[1].get(), nnz, 10);
        DUMP_ARRAY(T, std::cout, "csr_values", bufs[2].get(), nnz, 10);
      } else {
        DUMP_ARRAY(int, std::cout, "coo_row_ind", bufs[0].get(), nnz, 10);
        DUMP_ARRAY(int, std::cout, "coo_col_ind", bufs[1].get(), nnz, 10);
        DUMP_ARRAY(T, std::cout, "coo_values", bufs[2].get(), nnz, 10);
      }
    } else if (param.UseEllFormat()) {
      const auto& dev_props = kernel->GetDeviceProp();
      if (dev_props.major < 7) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name.get() + 
          " : Need device major version 7 or later to support BlockedEll format");
      }
    }

    sp_info->sparse_desc_ = onnxruntime::make_optional<cusparseSpMatDescr_t>(sparse_desc);
    sparse_guard.release();
    sp_info->handle_ = onnxruntime::make_optional<cusparseHandle_t>(handle);
    handle_guard.release();
    sparse_info = std::move(sp_info);

    is_packed = true;
    return Status::OK();
  }
};  // namespace cuda

template <typename T>
Status MatMul<T>::PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) {
  is_packed = false;
  // We only pack Matrix B just like CPU version
  // only if it is 2:4 pruned and only if A100 available
  // However, we will feed this to cuSparseLT as the first argument.
  // cuSparseLt only handles 2-D matrices
  if (param.input_idx == 1) {
#ifdef USE_CURSPARSELT
    if (IsAmpereAvaiable() && param.Is2x4Format()) {
      ORT_RETURN_IF_ERROR(Sparse2x4ComputeHelper<T>::PrePack(this, tensor, param, trans_A_, trans_B_, sparse_info_, is_packed));
    }
#endif
    if (param.UseCsrFormat() || param.UseCooFormat() || param.UseEllFormat()) {
      CuSparseHelper<T>::PrePack(this, tensor, param, trans_A_, trans_B_, sparse_info_, is_packed);
    }
  }
  return Status::OK();
}

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = trans_A_;
  bool transb = trans_B_;
  if (left_X->Shape().NumDimensions() == 1) {
    transa = false;
  }
  if (right_X->Shape().NumDimensions() == 1) {
    transb = false;
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape(), transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

#ifdef USE_CUSPARSELT
  if (sparse_info_) {
    Sparse2x4ComputeHelper<T> sparse_helper;
    return sparse_helper.Compute(this, *sparse_info_, helper, alpha_, transa, transb, left_X, Y);
  }
#endif

  const CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = transa ? static_cast<int>(helper.M()) : static_cast<int>(helper.K());
  const int ldb = transb ? static_cast<int>(helper.K()) : static_cast<int>(helper.N());
  const int ldc = static_cast<int>(helper.N());
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();
  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        Base::CublasHandle(),
        transB,
        transA,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(right_X->template Data<T>()),
        ldb,
        reinterpret_cast<const CudaT*>(left_X->template Data<T>()),
        lda,
        &zero,
        reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
        ldc,
        device_prop));
    return Status::OK();
  } else if (CanUseStridedBatchedGemm(left_X->Shape(), right_X->Shape(),
                                      transa, transb, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(Base::CublasHandle(),
                                                          transB,
                                                          transA,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &alpha,
                                                          reinterpret_cast<const CudaT*>(right_X->template Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(left_X->template Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));

    return Status::OK();
  }

  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->template Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->template Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->template MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(),
      transB,
      transA,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      right_arrays.GpuPtr(),
      ldb,
      left_arrays.GpuPtr(),
      lda,
      &zero,
      output_arrays.GpuPtr(),
      ldc,
      static_cast<int>(helper.OutputOffsets().size()),
      device_prop));

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
