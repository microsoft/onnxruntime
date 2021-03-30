// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dump_utils.h"
#include "cusparse_support.h"

#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/math/matmul.h"

namespace onnxruntime {
namespace cuda {

namespace cusparse_helper {
namespace guards {
auto close_dense_fn = [](cusparseDnMatDescr_t* desc) { cusparseDestroyDnMat(*desc); };
auto close_sparse_fn = [](cusparseSpMatDescr_t* desc) { cusparseDestroySpMat(*desc); };
auto destroy_cusparse_fn = [](cusparseHandle_t* desc) { cusparseDestroy(*desc); };
}  // namespace guards

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
/// <returns>Status</returns>
Status ConvertToBlockedEll(const CudaKernel* kernel,
                           int64_t ell_block_size, int64_t K, int64_t N, bool transpose, int32_t element_type, size_t element_size,
                           const void* input_data_initialier, IAllocatorUniquePtr<uint8_t>& ell_indicies_buffer, IAllocatorUniquePtr<uint8_t>& ell_values_buffer,
                           int64_t& ell_cols) {
  //Copy to CPU for processing
  const int64_t total_input_bytes = K * N * element_size;
  std::unique_ptr<uint8_t[]> cpu_local(new uint8_t[total_input_bytes]);
  CUDA_RETURN_IF_ERROR(cudaMemcpy(cpu_local.get(), input_data_initialier, total_input_bytes, cudaMemcpyDeviceToHost));

  const void* input_data = cpu_local.get();
  const int64_t block_elements = ell_block_size * ell_block_size;
  const int64_t block_bytes = block_elements * element_size;

  const int64_t src_row_element_bytes = N * element_size;  // bytes in a row of elements
  const int64_t src_block_rows = K / ell_block_size;
  const int64_t src_block_cols = N / ell_block_size;
  const int64_t ell_block_row_bytes = ell_block_size * element_size;

  const int64_t src_block_row_bytes = src_block_cols * block_bytes;
  const auto dst_block_rows = (transpose) ? src_block_rows : src_block_cols;

  // Key is the row and it contains the set of sorted col indicies
  std::map<int64_t, std::set<int64_t>> rows_to_cols;

  // We scan for non-zero blocks
  utils::MLTypeCallDispatcher<float, double, MLFloat16, BFloat16> t_disp(element_type);
  for (int64_t block_row = 0; block_row < src_block_rows; ++block_row) {
    const auto* block_row_begin = reinterpret_cast<const uint8_t*>(input_data) + block_row * src_block_row_bytes;
    const auto* block_row_end = block_row_begin + src_block_row_bytes;
    const auto* start = block_row_begin;
    int64_t block_col = -1;
    while (true) {
      t_disp.Invoke<FindNotZero>(N, ell_block_size, block_row_begin, start, block_row_end, block_col);
      if (start != nullptr) {
        assert(block_col != -1);
        assert(block_col < src_block_cols);
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

  // Calculate the amount of indicies per row by finding the max
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
  // For each row, we must make sure that we have equal amount col indicies in each row
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

  // Indicies array of rows X max_cols
  const int64_t nnz_blocks = dst_block_rows * static_cast<int64_t>(max_cols);
  std::unique_ptr<int[]> col_ind(new int[nnz_blocks]);
  // Value blocks are square block_elements * element_size * nnz_blocks
  const int64_t values_bytes = block_bytes * nnz_blocks;
  std::unique_ptr<uint8_t[]> values(new uint8_t[values_bytes]);

  // Lets build the col In and copy value blocks in a transposed manner
  const uint8_t* input = reinterpret_cast<const uint8_t*>(input_data);
  int* col_ind_out = col_ind.get();
  uint8_t* values_out = values.get();
  int64_t blocks_copied = 0;
  for (const auto& e : rows_to_cols) {
    ORT_ENFORCE(e.second.size() == max_cols, "Failed to check for equal columns");
    // Copy the block, we do the opposite of the transpose flag
    // as we intend to swap the args.
    // XXX: This currently copies block bytes together. Transposed.
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
      const auto* const src_block_row_start = input + src_block_row_idx * src_block_row_bytes;
      int64_t src_row_bytes_offset = 0;
      auto* row_output = values_out + blocks_copied * block_bytes;
      for (int64_t row = 0; row < ell_block_size; ++row) {
        for (auto src_block_col_idx : e.second) {
          const auto* const block_col_start = src_block_row_start +
                                              src_block_col_idx * ell_block_row_bytes;

          const auto* row_start = src_row_bytes_offset + block_col_start;

          // We copy element row by row
          memcpy(row_output, row_start, ell_block_row_bytes);
          row_output += ell_block_row_bytes;
        }
        src_row_bytes_offset += src_row_element_bytes;
      }

      // The above loops copy all blocks of the block row
      for (auto src_block_col_idx : e.second) {
        *col_ind_out++ = gsl::narrow_cast<int>(src_block_col_idx);
      }
      blocks_copied += max_cols;
    }
  }

  auto ell_indicies_buffer_local = kernel->GetPersistentBuffer<uint8_t>(nnz_blocks * sizeof(int));
  CUDA_RETURN_IF_ERROR(cudaMemcpy(ell_indicies_buffer_local.get(), col_ind.get(), nnz_blocks * sizeof(int), cudaMemcpyHostToDevice));
  auto ell_values_buffer_local = kernel->GetPersistentBuffer<uint8_t>(values_bytes);
  CUDA_RETURN_IF_ERROR(cudaMemcpy(ell_values_buffer_local.get(), values.get(), values_bytes, cudaMemcpyHostToDevice));

  DUMP_ARRAY(int, std::cout, "ell_indicies_buffer_local", ell_indicies_buffer_local.get(), nnz_blocks, max_cols);
  // Now the values
  const auto* values_in = ell_values_buffer_local.get();
  for (int64_t nnzs = 0; nnzs < nnz_blocks; ++nnzs) {
    // Dump block elements
    DUMP_INVOKE(t_disp, DumpArray, std::cout, "ell_values_blocks", values_in, block_elements, ell_block_size);
    values_in += block_bytes;
  }

  ell_indicies_buffer = std::move(ell_indicies_buffer_local);
  ell_values_buffer = std::move(ell_values_buffer_local);
  ell_cols = max_cols * ell_block_size;

  return Status::OK();
}

/// <summary>
/// The function executes the logic of BlockedEll format support
/// This function assumes that cudaDataType for a, b are uniform.
/// see https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm
/// </summary>
/// <param name="device_major">has to be 7 or 8</param>
/// <param name="a_b_type">cuda type for A and B</param>
/// <param name="op_A">transpose mode for B, but bc we are swapping args, it is A</param>
/// <returns></returns>
bool CanUseBlockedEllFormat(const cudaDeviceProp& dev_props, cudaDataType a_b_type,
                            cudaDataType c_type, cudaDataType compute_type) {
  if (
      (
          a_b_type == CUDA_R_16F &&
          (c_type == CUDA_R_16F || c_type == CUDA_R_32F) &&
          compute_type == c_type) ||
      (a_b_type == CUDA_R_16F && c_type == CUDA_R_16F && compute_type == CUDA_R_32F)) {
    return dev_props.major >= 7;
  }

  // Omit this for now
  //if (a_b_type == CUDA_R_8I &&
  //    c_type == CUDA_R_8I && compute_type == CUDA_R_32I &&
  //    !transa) {
  //  return (dev_props.major >= 7 && dev_props.minor >= 5);
  //}

  if (
      (a_b_type == CUDA_R_32F || a_b_type == CUDA_R_32F) &&
      c_type == a_b_type &&
      compute_type == a_b_type) {
    return dev_props.major >= 8;
  }

  return false;
}

Status PrePack(const CudaKernel* kernel, const Tensor& tensor, const OpKernel::PrepackParam& prepack_param,
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

  // Feed column order and swap dims
  const int64_t num_rows = transb ? K : N;
  const int64_t num_cols = transb ? N : K;
  const int64_t lead_dim = transb ? K : N;

  // Blocked ELL has partial support for Volta. For Float32/Float64 it still requires Ampere arch (8)
  // https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm
  // XXX:
  // const auto& dev_props = kernel->GetDeviceProp();
  // if (param.UseEllFormat() && CanUseBlockedEllFormat(dev_props, cuda_type, cuda_type, cuda_type)) {
  if (param.UseEllFormat()) {
    // Some tunables which we may adjust for both testing and depending on the matrix size.
    // Must be power of two
    const int64_t ell_block_size = param.ell_block_size;
    if ((K % ell_block_size) != 0 || (N % ell_block_size) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, param.name + " : Matrix dims: ", K, " ", N, " must divide evenly by a chosen Ell Block size: ", ell_block_size);
    }

    IAllocatorUniquePtr<uint8_t> ell_ind_buffer;
    IAllocatorUniquePtr<uint8_t> ell_values_buffer;
    int64_t ell_cols;
    ORT_RETURN_IF_ERROR(ConvertToBlockedEll(kernel, ell_block_size, K, N, transb, tensor.GetElementType(), element_size,
                                            tensor.DataRaw(), ell_ind_buffer, ell_values_buffer, ell_cols));

    /// XXX:
    ///  return Status::OK();

    CUSPARSE_RETURN_IF_ERROR(cusparseCreateBlockedEll(&sparse_desc,
                                                      num_rows,
                                                      num_cols,
                                                      ell_block_size,
                                                      ell_cols,
                                                      ell_ind_buffer.get(),
                                                      ell_values_buffer.get(),
                                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                      cuda_type));

    sparse_guard.reset(&sparse_desc);
    sp_info->prepack_buffers_.push_back(std::move(ell_ind_buffer));
    sp_info->prepack_buffers_.push_back(std::move(ell_values_buffer));
  } 
  
  //else if (param.UseEllFormat()) {
  //  // XXX: Right now we just choose some format
  //  // How do we log here?
  //  sp_info->param_.sparse_flags = param.sparse_flags & static_cast<int>(~OrtSparseFlags::USE_ELL_FORMAT);
  //  sp_info->param_.sparse_flags |= OrtSparseFlags::USE_CSR_FORMAT;
  //}

  if (param.UseCsrFormat() || param.UseCooFormat()) {
    cusparseDnMatDescr_t dense_desc;
    CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&dense_desc,
                                                 num_rows,  // Number of rows in B(T)
                                                 num_cols,  // Number of columns in B(T)
                                                 lead_dim,
                                                 // values_buffer.get(),
                                                 const_cast<void*>(tensor.DataRaw()),
                                                 cuda_type,
                                                 CUSPARSE_ORDER_COL));

    std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> dense_guard(&dense_desc, guards::close_dense_fn);

    onnxruntime::IAllocatorUniquePtr<uint8_t> csr_offsets;
    if (param.UseCsrFormat()) {
      csr_offsets = kernel->GetPersistentBuffer<uint8_t>((num_rows + 1) * sizeof(int));
      CUSPARSE_RETURN_IF_ERROR(cusparseCreateCsr(&sparse_desc,
                                                 num_rows,
                                                 num_cols,
                                                 0,  // nnz is zero now
                                                 csr_offsets.get(),
                                                 nullptr,                                 // colInd is null according to the example
                                                 nullptr,                                 // values is null according to the example
                                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,  // indicies are int
                                                 CUSPARSE_INDEX_BASE_ZERO, cuda_type));
    } else {
      CUSPARSE_RETURN_IF_ERROR(cusparseCreateCoo(&sparse_desc,
                                                 num_rows,
                                                 num_cols,
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

    if (param.UseCsrFormat()) {
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

    CUDA_CALL(cudaDeviceSynchronize());
    // XXX: Print all the buffers
    const auto& bufs = sp_info->prepack_buffers_;
    ORT_UNUSED_PARAMETER(bufs);
    DUMP_DISP(t_disp, expected_kernel_type, float, double, MLFloat16, BFloat16);
    if (param.UseCsrFormat()) {
      DUMP_ARRAY(int, std::cout, "csr_offsets", bufs[0].get(), num_rows + 1, 10);
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
  sparse_info = std::move(sp_info);

  is_packed = true;
  return Status::OK();
}

Status Compute(const CudaKernel* kernel, OpKernelContext* ctx, const SparseInfo& sparse_info,
               float alpha, bool transa, bool transb, cudaDataType cuda_type) {
  const Tensor* left = ctx->Input<Tensor>(0);
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

  // Dims swapped and we indicate ColumnMajor in all A, B and C
  // We should end up RowMajor C since we are feeding dense X sparse instead of
  // sparse X dense.
  const int64_t num_rows_A = transa ? M : K;
  const int64_t num_cols_A = transa ? K : M;
  const int64_t lead_dim_A = transa ? M : K;

  const int64_t num_rows_C = N;
  const int64_t num_cols_C = M;
  const int64_t lead_dim_C = N;

  cusparseDnMatDescr_t dense_desc_A;
  CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&dense_desc_A,
                                               num_rows_A,
                                               num_cols_A,
                                               lead_dim_A,
                                               const_cast<void*>(left->DataRaw()),  // They say we can safely cast constness away :)
                                               cuda_type,
                                               CUSPARSE_ORDER_COL));  // We have RowMajor but feeding like Column
  std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> dense_guard_A(&dense_desc_A, guards::close_dense_fn);

  cusparseDnMatDescr_t output_desc_C;
  CUSPARSE_RETURN_IF_ERROR(cusparseCreateDnMat(&output_desc_C,
                                               num_rows_C,
                                               num_cols_C,
                                               lead_dim_C,
                                               Y->MutableDataRaw(),
                                               cuda_type,
                                               CUSPARSE_ORDER_COL));
  std::unique_ptr<cusparseDnMatDescr_t, decltype(guards::close_dense_fn)> output_guard_C(&output_desc_C, guards::close_dense_fn);

  cusparseOperation_t op_A = transa ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op_B = transb ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  constexpr float beta = 0.f;
  size_t buffer_size = 0;

  cusparseHandle_t handle = kernel->CusparseHandle();

  // For CSR
  // CUSPARSE_SPMM_CSR_ALG3 is the newer algo but it does not support transposing A (in our case B) and falls back to
  // CUSPARSE_SPMM_CSR_ALG2 which provides better performance with RowMajor. We are RowMajor but
  // because of the argument swapping we say ColMajor.
  // CUSPARSE_SPMM_CSR_ALG3 also does not support float16/bfloat16 data types
  cusparseSpMMAlg_t spmm_algo = CUSPARSE_SPMM_ALG_DEFAULT;
  if (sparse_info.param_.UseCsrFormat()) {
    if (op_B == CUSPARSE_OPERATION_TRANSPOSE) {
      spmm_algo = CUSPARSE_SPMM_CSR_ALG2;
    } else if (cuda_type != CUDA_C_16F && cuda_type != CUDA_C_16BF) {
      spmm_algo = CUSPARSE_SPMM_CSR_ALG3;
    }
  } else if (sparse_info.param_.UseCooFormat()) {
    spmm_algo = CUSPARSE_SPMM_COO_ALG3;
  }

  // For Blocked ELL the default is translated to CUSPARSE_SPMM_BLOCKED_ELL_ALG1 algorithm

  CUSPARSE_RETURN_IF_ERROR(cusparseSpMM_bufferSize(handle,
                                                   op_B,
                                                   op_A,
                                                   &alpha,
                                                   *sparse_info.sparse_desc_,
                                                   dense_desc_A,
                                                   &beta,
                                                   output_desc_C,
                                                   cuda_type,
                                                   spmm_algo,
                                                   &buffer_size));

  IAllocatorUniquePtr<uint8_t> work_buffer;
  if (buffer_size > 0) {
    work_buffer = kernel->GetScratchBuffer<uint8_t>(buffer_size);
  }

  CUSPARSE_RETURN_IF_ERROR(cusparseSpMM(handle,
                                        op_B,
                                        op_A,
                                        &alpha,
                                        *sparse_info.sparse_desc_,
                                        dense_desc_A,
                                        &beta,
                                        output_desc_C,
                                        cuda_type,
                                        spmm_algo,
                                        work_buffer.get()));

  CUDA_CALL(cudaDeviceSynchronize());
  // Debug dump
  DUMP_DISP(t_disp, left->GetElementType(), float, double, MLFloat16, BFloat16);
  DUMP_INVOKE(t_disp, DumpArray, std::cout, "cusparseSpMM output", Y->DataRaw(), Y->Shape().Size(), helper.K());

  return Status::OK();
}
}  // namespace cusparse_helper
}  // namespace cuda
}  // namespace onnxruntime
