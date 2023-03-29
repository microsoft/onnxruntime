// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/tensor/transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Transpose);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Transpose);

// special case acceleration using cublas matrix transpose
static std::tuple<int, int> TryTransposeWithCublas(const gsl::span<const size_t>& perm, const TensorShape& input_shape) {
  int M = 0;
  int N = 0;

  if (perm.size() == 4 && input_shape[0] == 1 && perm[0] == 0) {
    // NCHW <-> NHWC when N == 1
    if ((perm[1] == 2 && perm[2] == 3 && perm[3] == 1) ||
        (perm[1] == 3 && perm[2] == 1 && perm[3] == 2)) {
      if (perm[1] == 2) {
        M = gsl::narrow<int>(input_shape[1]);
        N = gsl::narrow<int>(input_shape[2] * input_shape[3]);
      } else {
        M = gsl::narrow<int>(input_shape[1] * input_shape[2]);
        N = gsl::narrow<int>(input_shape[3]);
      }
    }
  } else if (perm.size() == 2 && perm[1] == 0 && perm[0] == 1) {
    // 2D matrix transpose
    M = gsl::narrow<int>(input_shape[0]);
    N = gsl::narrow<int>(input_shape[1]);
  }

  return std::make_tuple(M, N);
}

template <typename T>
Status TransposeWithCublas(cudaStream_t stream, cublasHandle_t cublas_handle, const Tensor& input, Tensor& output, int M, int N) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  const CudaT* input_data = reinterpret_cast<const CudaT*>(input.Data<T>());
  CudaT* output_data = reinterpret_cast<CudaT*>(output.MutableData<T>());

  CUBLAS_RETURN_IF_ERROR(
      cublasTransposeHelper(stream,
                            cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T, M, N,
                            &one,
                            input_data,
                            N,
                            &zero,
                            input_data,
                            N,
                            output_data,
                            M));
  return Status::OK();
}

Status Transpose::DoTranspose(const Transpose& transpose_kernel,
                              onnxruntime::Stream* ort_stream,
                              const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output) {
  cudaStream_t cuda_stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;
  return Transpose::DoTranspose(transpose_kernel.GetDeviceProp(),
                                cuda_stream,
                                CudaKernel::GetCublasHandle(static_cast<CudaStream*>(ort_stream)),
                                permutations,
                                input, output);
}

Status Transpose::DoTranspose(const cudaDeviceProp& prop,
                              cudaStream_t stream,
                              const cublasHandle_t cublas_handle,
                              const gsl::span<const size_t>& permutations, const Tensor& input, Tensor& output,
                              const TensorShape* input_shape_override,
                              const TensorShape* output_shape_override) {
  // special case when there is a dim value of 0 in the shape.
  if (output.Shape().Size() == 0)
    return Status::OK();

  const auto input_dims = input_shape_override ? input_shape_override->GetDims() : input.Shape().GetDims();
  const auto output_dims = output_shape_override ? output_shape_override->GetDims() : output.Shape().GetDims();
  auto rank = static_cast<int32_t>(input_dims.size());

  // flatten the adjacent dimensions which are contiguous
  // for example: permutations[0, 2, 3, 1] -> [0, 2, 1], permutations[0, 3, 1, 2] -> [0, 2, 1]
  auto new_rank = rank;
  InlinedVector<size_t> new_permutations(permutations.begin(), permutations.end());
  TensorShapeVector new_input_dims = ToShapeVector(input_dims);
  TensorShapeVector new_output_dims = ToShapeVector(output_dims);

  // Remove all dims with value 1.
  std::vector<bool> dims_to_remove(new_rank, false);
  int input_pos = 0;
  int output_pos = 0;
  int perm_pos = 0;
  for (int i = 0; i < new_rank; ++i) {
    if (new_input_dims[i] != 1) {
      new_input_dims[input_pos++] = new_input_dims[i];
    } else {
      dims_to_remove[i] = true;
    }
    if (new_output_dims[i] != 1) {
      new_output_dims[output_pos++] = new_output_dims[i];
    }
  }
  for (int i = 0; i < new_rank; ++i) {
    if (!dims_to_remove[new_permutations[i]]) {
      new_permutations[perm_pos++] = new_permutations[i];
    }
  }
  for (int i = new_rank - 1; i >= 0; --i) {
    if (dims_to_remove[i]) {
      for (int j = 0; j < perm_pos; ++j) {
        if (new_permutations[j] > static_cast<size_t>(i)) {
          new_permutations[j] -= 1;
        }
      }
    }
  }
  ORT_ENFORCE(input_pos == output_pos && input_pos == perm_pos);
  new_rank = input_pos;
  new_input_dims.resize(new_rank);
  new_output_dims.resize(new_rank);
  new_permutations.resize(new_rank);

  for (auto i = new_rank - 1; i > 0; i--) {
    auto curr = new_permutations[i];
    auto prev = new_permutations[static_cast<ptrdiff_t>(i) - 1];
    if (prev + 1 == curr) {
      // all dims bigger than curr need to be reduced by 1 due to the merging.
      for (auto j = 0; j < new_rank; j++) {
        if (new_permutations[j] > curr) {
          new_permutations[j] -= 1;
        }
      }
      for (auto j = i + 1; j < new_rank; j++) {
        new_permutations[static_cast<ptrdiff_t>(j) - 1] = new_permutations[j];
      }

      // update input dims
      new_input_dims[prev] *= new_input_dims[curr];
      new_input_dims[curr] = 1;
      for (auto j = static_cast<int32_t>(curr + 1); j < new_rank; j++) {
        new_input_dims[static_cast<ptrdiff_t>(j) - 1] = new_input_dims[j];
      }
      new_input_dims[new_rank - 1] = 1;

      // update output dims
      new_output_dims[i - 1] *= new_output_dims[i];
      new_output_dims[i] = 1;
      for (auto j = i + 1; j < new_rank; j++) {
        new_output_dims[j - 1] = new_output_dims[j];
      }
      new_output_dims[new_rank - 1] = 1;

      new_rank--;
    }
  }
  new_permutations.resize(new_rank);
  new_input_dims.resize(new_rank);
  new_output_dims.resize(new_rank);

  if (new_rank <= 1) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output.MutableDataRaw(), input.DataRaw(),
                                         input.Shape().Size() * input.DataType()->Size(), cudaMemcpyDeviceToDevice,
                                         stream));
    return Status::OK();
  }

  auto element_type = input.GetElementType();
  size_t element_size = input.DataType()->Size();
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
      element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    auto mn = TryTransposeWithCublas(new_permutations, new_input_dims);
    int M = std::get<0>(mn);
    int N = std::get<1>(mn);
    if (M != 0 && N != 0) {
      if (element_type == utils::GetONNXTensorElementDataType<float>()) {
        return TransposeWithCublas<float>(stream, cublas_handle, input, output, M, N);
      } else if (element_type == utils::GetONNXTensorElementDataType<double>()) {
        return TransposeWithCublas<double>(stream, cublas_handle, input, output, M, N);
      } else {
        return TransposeWithCublas<MLFloat16>(stream, cublas_handle, input, output, M, N);
      }
    }
  }

  // Transpose021 has a specialized Transpose3DImpl kernel
  dim3 grid_size, block_size;
  if (CanDoTranspose3D(prop, static_cast<size_t>(new_rank), new_input_dims, new_permutations, grid_size, block_size)) {
    TensorPitches new_input_strides(new_input_dims);
    return Transpose3DImpl(stream, element_size, ToConstSpan(new_input_dims), ToConstSpan(new_input_strides),
                           input.DataRaw(), output.MutableDataRaw(), output.Shape().Size(), grid_size, block_size);
  }

  // 3D-Transpose can treated as a special case of 4D-Transpose with first dimension being 1.
  if (new_rank == 3) {
    new_permutations[0]++;
    new_permutations[1]++;
    new_permutations[2]++;
    new_permutations.insert(new_permutations.begin(), 0);
    new_input_dims.insert(new_input_dims.begin(), 1);
    new_output_dims.insert(new_output_dims.begin(), 1);
    new_rank = 4;
  }

  TensorPitches new_input_strides(new_input_dims);
  TensorPitches new_output_strides(new_output_dims);
  TArray<int64_t> input_shape(new_input_dims);
  TArray<int64_t> tmp_input_strides(new_input_strides);

  if (CanDoTranspose4DParallelizeMultipleElementsPerThreadInInnermostDim(
          prop, element_size, new_rank, new_input_dims, new_permutations,
          grid_size, block_size)) {
    TArray<int64_t> tmp_output_strides(new_rank);
    for (auto i = 0; i < new_rank; i++) {
      tmp_output_strides[static_cast<int32_t>(new_permutations[i])] = new_output_strides[i];
    }
    return Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim(
        stream, element_size, input_shape, tmp_input_strides, input.DataRaw(),
        tmp_output_strides, output.MutableDataRaw(), gsl::narrow<int>(output.Shape().Size()),
        grid_size, block_size);
  }
  // We used to check if Transpose4DParallelizeOneElementPerThread can be used before falling back to generic case,
  // But tests on lots of cases showing that Transpose4DParallelizeOneElementPerThread is not faster than generic case,
  // and even much slower than generic case for some cases.

  // General cases
  TArray<int64_t> input_strides(new_rank);
  for (auto i = 0; i < new_rank; i++) {
    input_strides[i] = new_input_strides[new_permutations[i]];
  }

  TArray<fast_divmod> output_strides(new_rank);
  for (auto i = 0; i < new_rank; i++) {
    output_strides[i] = fast_divmod(gsl::narrow_cast<int>(new_output_strides[i]));
  }

  auto status = TransposeImpl(stream, element_size, new_rank, input_strides, input.DataRaw(),
                              output_strides, output.MutableDataRaw(), gsl::narrow<int>(output.Shape().Size()));

  return status;
}

Status Transpose::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X_ptr = ctx->Input<Tensor>(0);
  if (X_ptr == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *X_ptr;
  const TensorShape& input_shape = X.Shape();
  int32_t rank = gsl::narrow_cast<int32_t>(input_shape.NumDimensions());

  TensorShapeVector output_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* p_perm = nullptr;
  const auto& status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  return DoTranspose(this->GetDeviceProp(), this->Stream(ctx), this->GetCublasHandle(ctx), *p_perm, X, *Y);
}

}  // namespace cuda
}  // namespace onnxruntime
