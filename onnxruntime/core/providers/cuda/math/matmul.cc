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

template <typename T>
struct DumpType {
  static std::ostream& dump(std::ostream& os, T v) {
    return os << std::setw(12) << std::setprecision(8) << v;
  }
};

template <>
struct DumpType<MLFloat16> {
  static std::ostream& dump(std::ostream& os, const MLFloat16& v) {
    return DumpType<float>::dump(os, v.ToFloat());
  }
};

template <>
struct DumpType<BFloat16> {
  static std::ostream& dump(std::ostream& os, const BFloat16& v) {
    return DumpType<float>::dump(os, v.ToFloat());
  }
};

template <typename T>
struct DumpArray {
  void operator()(std::ostream& os, const std::string& name, const void* in, size_t len, size_t col_width) const {
    std::unique_ptr<T[]> buf(new T[len]);
    cudaMemcpy(buf.get(), in, len * sizeof(T), cudaMemcpyDeviceToHost);
    const T* src = buf.get();

    os << "Dump array: " << name << std::endl;

    if (col_width == -1) col_width = len;

    for (size_t i = 0; i < len;) {
      for (size_t w = 0; w < col_width && i < len; ++w, ++i) {
        DumpType<T>::dump(os, src[i]);
      }
      os << std::endl;
    }
    os << std::endl;
  }
};

#if 1
#define DUMP_TYPE(T, ...) DumpType<T>()(__VA_ARGS__)
#define DUMP_ARRAY(T, ...) DumpArray<T>()(__VA_ARGS__)
#define DUMP_DISP(var, t, ...) utils::MLTypeCallDispatcher<__VA_ARGS__> var(t)
#define DUMP_INVOKE(var, fn, ...) var.Invoke<fn>(__VA_ARGS__)
#else
#define DUMP_DISP(var, t, ...)
#define DUMP_INVOKE(var, fn, ...)
#define DUMP_TYPE(T, ...)
#define DUMP_ARRAY(T, ...)
#endif

template <typename T>
struct IsZero {
  bool operator()(T v) const noexcept {
    return v == static_cast<T>(0);
  }
};

static const MLFloat16 zero_ml16(0.f);

template <>
struct IsZero<MLFloat16> {
  bool operator()(MLFloat16 v) const noexcept {
    return zero_ml16.val == v.val;
  }
};

static const BFloat16 zero_b16(0.f);

template <>
struct IsZero<BFloat16> {
  bool operator()(BFloat16 v) const noexcept {
    return zero_b16.val == v.val;
  }
};

/// <summary>
/// Finds the first non-zero entry and computes its col index.
/// Advances restart past the found entry. restart is nullptr if
/// reached the end of the block row.
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
struct FindIfZero {
  // returns nullptr if not found
  void operator()(int64_t N,
                  const uint8_t* block_row_begin,
                  const uint8_t*& restart, const uint8_t* block_row_end,
                  int64_t& block_col) const {
    const T* block_row_begin_T = reinterpret_cast<const T*>(block_row_begin);
    const T* start = reinterpret_cast<const T*>(restart);
    const T* block_row_end_T = reinterpret_cast<const T*>(block_row_end);
    auto hit = std::find_if(start, block_row_end_T, IsZero<T>());
    if (hit != block_row_end_T) {
      block_col = (hit - block_row_begin_T) % N;
      restart = reinterpret_cast<const uint8_t*>(hit + 1);
    } else {
      restart = nullptr;
    }
  }
};

/// <summary>
/// Scans the matrix and converts it into Blocked Ell format.
/// </summary>
/// <param name="ell_block_size">Chosen block sizes</param>
/// <param name="K">rows in the matrix to be sparsified</param>
/// <param name="N">columns in the incoming matrix</param>
/// <param name="transpose">whether transpose flag was specified</param>
/// <param name="element_type">element type</param>
/// <param name="element_size">element size in the matrix</param>
/// <param name="input_data">matrix buffer</param>
/// <param name="ell_col_ind">output parameter, ell_col_indices represented as an array</param>
/// <param name="ell_values">non-zero blocks values</param>
/// <returns>status</returns>
static Status ConvertToBlockedEll(int64_t ell_block_size, int64_t K, int64_t N, bool transpose, int32_t element_type, size_t element_size,
                                  const void* input_data, std::unique_ptr<int[]>& ell_col_ind, std::unique_ptr<uint8_t[]>& ell_values,
                                  int64_t& ell_rows, int64_t& ell_cols) {
  const int64_t block_elements = ell_block_size * ell_block_size;
  const int64_t block_bytes = block_elements * element_size;

  const int64_t src_row_element_bytes = K * element_size;
  const int64_t src_block_rows = K / ell_block_size;
  const int64_t src_block_cols = N / ell_block_size;
  const int64_t ell_block_row_bytes = ell_block_size * element_size;

  const int64_t src_block_row_bytes = src_block_cols * block_bytes;
  const auto dst_block_rows = (transpose) ? src_block_rows : src_block_cols;

  // Key is the row and it contains the set of sorted col indices
  std::map<int64_t, std::set<int64_t>> rows_to_cols;

  // We scan for non-zero blocks
  utils::MLTypeCallDispatcher<float, double, MLFloat16, BFloat16> t_disp(element_type);
  for (int64_t block_row = 0; block_row < src_block_rows; ++block_row) {
    const auto* block_row_begin = reinterpret_cast<const uint8_t*>(input_data) + block_row * src_block_row_bytes;
    const auto* block_row_end = block_row_begin + src_block_row_bytes;
    const auto* start = block_row_begin;
    int64_t block_col = -1;
    while (true) {
      t_disp.Invoke<FindIfZero>(N, block_row_begin, start, block_row_end, block_col);
      if (start != nullptr) {
        assert(block_col != -1);
        // We transpose for !transpose because we want to
        // swap arguments in Gemm formula
        if (!transpose) {
          rows_to_cols[block_col].insert(block_row);
        } else {
          rows_to_cols[block_row].insert(block_col);
        }
        block_col = -1;
      } else {
        break;
      }
    }
  }

  // Calculate the amount of indecies per row by finding the max
  size_t max_cols = 0;
  for (const auto& e : rows_to_cols) {
    max_cols = std::max(max_cols, e.second.size());
  }

  if (max_cols == 0) {
    // matrix is all zeros
    // outputs are also nullptr
    return Status::OK();
  }

  // Now we have all non-empty blocks.
  // We need to make sure, that we do not have any missing rows in the col index.
  // For each row, we must make sure that we have equal amount col indecies in each row
  // by inserting indecies to some zero blocks that we discarded earlier.
  // See https://github.com/NVIDIA/CUDALibrarySamples/issues/24
  for (int64_t block_row = 0; block_row < dst_block_rows; ++block_row) {
    auto& cols = rows_to_cols[block_row];
    // fill in some zero indecies for padding
    for (int64_t i = 0; cols.size() < max_cols; ++i) {
      assert(i < K);
      cols.insert(i);  // duplicates will not be inserted
    }
  }

  // Indecies array of rows X max_cols
  const int64_t nnz_blocks = dst_block_rows * static_cast<int64_t>(max_cols);
  std::unique_ptr<int[]> col_ind(new int[nnz_blocks]);
  // Value blocks are square block_elements * element_size * nnz_blocks
  const int64_t values_bytes = block_bytes * nnz_blocks;
  std::unique_ptr<uint8_t[]> values(new uint8_t[values_bytes]);

  const int64_t dst_block_row_bytes = block_bytes * static_cast<int64_t>(max_cols);

  // Lets build the col In and copy value blocks in a transposed manner
  const uint8_t* input = reinterpret_cast<const uint8_t*>(input_data);
  int* col_ind_out = col_ind.get();
  uint8_t* values_out = values.get();
  int64_t blocks_copied = 0;
  for (const auto& e : rows_to_cols) {
    ORT_ENFORCE(e.second.size() == max_cols, "Failed to check for equal columns");
    // Copy the block, we do the opposite of the transpose flag
    // as we intend to swap the args.
    if (!transpose) {
      const auto src_block_col_idx = e.first;
      for (auto src_block_row_idx : e.second) {
        const auto* const block_row_start = input + src_block_row_idx * src_block_row_bytes +
                                            src_block_col_idx * ell_block_row_bytes;
        auto* const block_output = values_out + block_bytes * blocks_copied;
        // Copy row by row transposed
        for (int64_t row = 0; row < ell_block_size; ++row) {
          const auto* row_start = block_row_start + row * src_row_element_bytes;
          // Shift each row one element to the right
          auto* row_output = block_output + row * element_size;
          // Element by element with ell_block_size distance
          for (int64_t element = 0; element < ell_block_size; ++element) {
            // Spread output ell_block_row_bytes apart
            auto* element_output = row_output + element * ell_block_row_bytes;
            memcpy(element_output, row_start, element_size);
            row_start += element_size;
          }
        }
        // Becomes col index
        *col_ind_out++ = gsl::narrow_cast<int>(src_block_row_idx);
        ++blocks_copied;
      }
    } else {
      // Copy entire block row by row
      const auto src_block_row_idx = e.first;
      for (auto src_block_col_idx : e.second) {
        const auto* const block_row_start = input + src_block_row_idx * src_block_row_bytes +
                                            src_block_col_idx * ell_block_row_bytes;
        auto* row_output = values_out + block_bytes * blocks_copied;
        for (int64_t row = 0; row < ell_block_size; ++row) {
          const auto* row_start = block_row_start + row * src_row_element_bytes;
          // We copy entire block rows
          memcpy(row_output, row_start, ell_block_row_bytes);
          row_output += ell_block_row_bytes;
        }
        *col_ind_out++ = gsl::narrow_cast<int>(src_block_col_idx);
        ++blocks_copied;
      }
    }
  }

  // Let's dump the converted matrix
  // first indices
  const int* col_ind_in = col_ind.get();
  ORT_UNUSED_PARAMETER(col_ind_in);
  DUMP_ARRAY(int, std::cout, "col_ind_block", col_ind_in, nnz_blocks, max_cols);

  // Now the values
  const auto* values_in = values.get();
  ORT_UNUSED_PARAMETER(values_in);
  for (int64_t nnzs = 0; nnzs < nnz_blocks; ++nnzs) {
    DUMP_INVOKE(t_disp, DumpArray, std::cout, "col_ind_block", values_in, ell_block_size, ell_block_size);
    values_in += block_bytes;
  }

  ell_col_ind = std::move(col_ind);
  ell_values = std::move(values);
  ell_rows = static_cast<int64_t>(rows_to_cols.size());
  ell_cols = static_cast<int64_t>(max_cols);

  return Status::OK();
}

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
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : 2:4 data format validation failed");
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

namespace guards {
auto close_dense_fn = [](cusparseDnMatDescr_t* desc) { cusparseDestroyDnMat(*desc); };
auto close_sparse_fn = [](cusparseSpMatDescr_t* desc) { cusparseDestroySpMat(*desc); };
auto destroy_cusparse_fn = [](cusparseHandle_t* desc) { cusparseDestroy(*desc); };
}  // namespace guards

class CuSparseHelper {
 public:
  static Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
                        bool transb, int32_t expected_kernel_type, cudaDataType cuda_type,
                        std::unique_ptr<SparseInfo>& sparse_info, bool& is_packed) {
    is_packed = false;
    if (tensor.GetElementType() != expected_kernel_type) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, prepack_param.name + " : wrong data type for the constant initializer");
    }

    // We do not want to use per-thread handles but persists between the runs.
    cusparseHandle_t handle;
    CUSPARSE_RETURN_IF_ERROR(cusparseCreate(&handle));
    std::unique_ptr<cusparseHandle_t, decltype(guards::destroy_cusparse_fn)> handle_guard(&handle, guards::destroy_cusparse_fn);

    // XXX: Currently support only 2-D Matrices for experimental purposes
    const auto& right_shape = tensor.Shape();
    const auto element_size = tensor.DataType()->Size();
    const auto num_elements = right_shape.Size();
    const auto right_num_dims = right_shape.NumDimensions();
    if (right_num_dims > 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2");
    }

    if (right_num_dims == 1) {
      transb = false;
    }

    // MxKxN
    int64_t K = 0;
    int64_t N = 0;
    if (right_num_dims == 2) {
      K = right_shape[right_num_dims - 2];
      N = right_shape[right_num_dims - 1];
    } else {
      K = right_shape[0];
      N = 1;
    }

    cusparseSpMatDescr_t sparse_desc;
    std::unique_ptr<cusparseSpMatDescr_t, decltype(guards::close_sparse_fn)> sparse_guard(nullptr, guards::close_sparse_fn);

    auto sp_info = onnxruntime::make_unique<SparseInfo>(prepack_param, right_shape);
    const OpKernel::PrepackParam& param = sp_info->param_;
    sp_info->K_ = K;
    sp_info->N_ = N;

    // if Ell format is specified but the hardware is not we then default
    // to one of the belows
    const auto& dev_props = kernel->GetDeviceProp();
    if (param.UseEllFormat() && dev_props.major >= 7) {
      // Some tunables which we may adjust for both testing and depending on the matrix size.
      // Must be power of two
      constexpr int64_t ell_block_size = 8;
      if ((K % ell_block_size) != 0 || (N % ell_block_size) != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : Matrix dims: ", K, " ", N, " must divide evenly by a chosen Ell Block size: ", ell_block_size);
      }

      std::unique_ptr<int[]> ell_col_ind;
      std::unique_ptr<uint8_t[]> ell_values;
      int64_t ell_rows;
      int64_t ell_cols;
      ORT_RETURN_IF_ERROR(ConvertToBlockedEll(ell_block_size, K, N, transb, tensor.GetElementType(), element_size,
                                              tensor.DataRaw(), ell_col_ind, ell_values, ell_rows, ell_cols));
      const int64_t ell_ind_elements = ell_rows * ell_cols;
      auto ell_ind_buffer = kernel->GetPersistentBuffer<uint8_t>(ell_ind_elements * sizeof(int));
      CUDA_RETURN_IF_ERROR(cudaMemcpy(ell_ind_buffer.get(), ell_col_ind.get(), ell_ind_elements * sizeof(int), cudaMemcpyHostToDevice));

      const int64_t ell_values_elements = ell_ind_elements * ell_block_size * ell_block_size;
      auto ell_values_buffer = kernel->GetPersistentBuffer<uint8_t>(ell_values_elements * element_size);
      CUDA_RETURN_IF_ERROR(cudaMemcpy(ell_values_buffer.get(), ell_values.get(), ell_values_elements * element_size,
                                      cudaMemcpyHostToDevice));

      CUSPARSE_RETURN_IF_ERROR(cusparseCreateBlockedEll(&sparse_desc,
                                                        transb ? K : N,
                                                        transb ? N : K,
                                                        ell_block_size,
                                                        ell_cols,
                                                        ell_ind_buffer.get(),
                                                        ell_values_buffer.get(),
                                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                        cuda_type));

      sparse_guard.reset(&sparse_desc);
      sp_info->prepack_buffers_.push_back(std::move(ell_ind_buffer));
      sp_info->prepack_buffers_.push_back(std::move(ell_values_buffer));

    } else if (param.UseEllFormat()) {
      // XXX: Right now we just choose some format
      // Hardware is not available, default to Csr format
      sp_info->param_.sparse_flags = param.sparse_flags & static_cast<int>(~OrtSparseFlags::USE_ELL_FORMAT);
      sp_info->param_.sparse_flags |= OrtSparseFlags::USE_CSR_FORMAT;
    }

    if (param.UseCsrFormat() || param.UseCooFormat()) {
      auto values_buffer = kernel->GetScratchBuffer<uint8_t>(element_size * num_elements);
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

      std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> dense_guard(&dense_desc, guards::close_dense_fn);

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
      ORT_UNUSED_PARAMETER(bufs);
      DUMP_DISP(t_disp, expected_kernel_type, float, double, MLFloat16, BFloat16);
      if (param.UseCsrFormat()) {
        DUMP_ARRAY(int, std::cout, "csr_offsets", bufs[0].get(), K + 1, 10);
        DUMP_ARRAY(int, std::cout, "csr_cols", bufs[1].get(), nnz, 10);
        DUMP_INVOKE(t_disp, DumpArray, std::cout, "csr_values", bufs[2].get(), nnz, 10);
      } else {
        DUMP_ARRAY(int, std::cout, "coo_row_ind", bufs[0].get(), nnz, 10);
        DUMP_ARRAY(int, std::cout, "coo_col_ind", bufs[1].get(), nnz, 10);
        DUMP_INVOKE(t_disp, DumpArray, std::cout, "coo_values", bufs[2].get(), nnz, 10);
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

  static Status Compute(const CudaKernel* kernel, OpKernelContext* ctx, const SparseInfo& sparse_info,
                        float alpha, bool transa, bool transb, const Tensor* left, cudaDataType cuda_type) {
    const auto& left_shape = left->Shape();
    const auto left_num_dims = left_shape.NumDimensions();
    if (left_num_dims > 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently do not support dims higher than 2");
    }

    const auto& right_shape = sparse_info.shape_;
    if (right_shape.NumDimensions() == 1) {
      transb = false;
    }

    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(left->Shape(), right_shape, transa, transb));

    const auto M = helper.M();
    const auto N = helper.N();
    const auto K = helper.K();

    ORT_RETURN_IF_NOT(K == sparse_info.K_, "K does not match sparse weight computed K");
    ORT_RETURN_IF_NOT(N == sparse_info.N_, "N does not match sparse weight computed N");

    const auto& output_shape = helper.OutputShape();
    // Make sure we still request output allocation if the len is zero
    Tensor* Y = ctx->Output(0, output_shape);
    // Bail out early if the output is going to be empty
    if (output_shape.Size() == 0)
      return Status::OK();

    cusparseDnMatDescr_t dense_desc;
    CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&dense_desc,
                                                 (transa) ? M : K,
                                                 (transa) ? K : M,
                                                 (transa) ? M : K,
                                                 const_cast<void*>(left->DataRaw()),  // They say we can safely cast constness away :)
                                                 cuda_type,
                                                 CUSPARSE_ORDER_COL));  // We have RowMajor but feeding like Column
    std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> dense_guard(&dense_desc, guards::close_dense_fn);

    // Create output matrix with transposed dimensions
    cusparseDnMatDescr_t output_desc;
    CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&output_desc,
                                                 N,
                                                 M,
                                                 N,
                                                 Y->MutableDataRaw(),
                                                 cuda_type,
                                                 CUSPARSE_ORDER_COL));
    std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> output_guard(&output_desc, guards::close_dense_fn);

    cusparseOperation_t op_A = transa ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t op_B = transb ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    constexpr float beta = 0.f;
    size_t buffer_size = 0;

    CUSPARSE_RETURN_IF_ERROR(cusparseSpMM_bufferSize(*sparse_info.handle_,
                                                     op_A,
                                                     op_B,
                                                     &alpha,
                                                     *sparse_info.sparse_desc_,
                                                     dense_desc,
                                                     &beta,
                                                     output_desc,
                                                     cuda_type,
                                                     CUSPARSE_SPMM_ALG_DEFAULT,
                                                     &buffer_size));

    auto work_buffer = kernel->GetScratchBuffer<uint8_t>(buffer_size);
    CUSPARSE_RETURN_IF_ERROR(cusparseSpMM(*sparse_info.handle_,
                                          op_A,
                                          op_B,
                                          &alpha,
                                          *sparse_info.sparse_desc_,
                                          dense_desc,
                                          &beta,
                                          output_desc,
                                          cuda_type,
                                          CUSPARSE_SPMM_ALG_DEFAULT,
                                          work_buffer.get()));

    // Debug dump
    DUMP_DISP(t_disp, left->GetElementType(), float, double, MLFloat16, BFloat16);
    DUMP_INVOKE(t_disp, DumpArray, std::cout, "cusparseSpMM output", Y->DataRaw(), Y->Shape().Size(), helper.K());

    return Status::OK();
  }
};

template <typename T>
Status MatMul<T>::PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) {
  is_packed = false;
  if (param.input_idx == 1) {
#ifdef USE_CURSPARSELT
    if (IsAmpereAvaiable() && param.Is2x4Format()) {
      ORT_RETURN_IF_ERROR(Sparse2x4ComputeHelper<T>::PrePack(this, tensor, param, trans_A_, trans_B_, sparse_info_, is_packed));
    }
#endif
    if (param.UseCsrFormat() || param.UseCooFormat() || param.UseEllFormat()) {
      CuSparseHelper::PrePack(this, tensor, param, trans_B_, utils::ToTensorProtoElementType<T>(), ToCudaTypeEnum<T>::type,
                              sparse_info_, is_packed);
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

#ifdef USE_CUSPARSELT
  if (sparse_info_ && sparse_info_->param_.Is2x4Format()) {
    Sparse2x4ComputeHelper<T> sparse_helper;
    // XXX: Use the helper inside and allocate output inside as well
    return sparse_helper.Compute(this, *sparse_info_, helper, alpha_, transa, transb, left_X, Y);
  }
#endif

  if (sparse_info_) {
    if (sparse_info_->param_.UseCooFormat() ||
        sparse_info_->param_.UseCsrFormat() ||
        sparse_info_->param_.UseEllFormat()) {
      return CuSparseHelper::Compute(this, ctx, *sparse_info_, alpha_, trans_A_, trans_B_, left_X, ToCudaTypeEnum<T>::type);
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, sparse_info_->param_.name + 
            " : Unsupported sparse format specified");
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

    DUMP_ARRAY(T, std::cout, "Offsets 1 output", Y->DataRaw(), Y->Shape().Size(), helper.K());
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
    DUMP_ARRAY(T, std::cout, "StridedBatched output", Y->DataRaw(), Y->Shape().Size(), helper.K());
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

  DUMP_ARRAY(T, std::cout, "Batched output", Y->DataRaw(), Y->Shape().Size(), helper.K());
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
