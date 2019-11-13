// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose.h"
#include "transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(Transpose,
                        kOnnxDomain,
                        1,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
                        Transpose);

// special case acceleration using cublas matrix transpose
static std::tuple<int, int> TryTransposeWithCublas(const std::vector<size_t>& perm, const TensorShape& input_shape) {
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
Status TransposeWithCublas(cublasHandle_t cublas_handle, const Tensor& input, Tensor& output, int M, int N) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  const CudaT* input_data = reinterpret_cast<const CudaT*>(input.Data<T>());
  CudaT* output_data = reinterpret_cast<CudaT*>(output.MutableData<T>());
  CUBLAS_RETURN_IF_ERROR(
      cublasTransposeHelper(cublas_handle,
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

Status Transpose::DoTranspose(const Transpose& kernel,
                              const std::vector<size_t>& permutations, const Tensor& input, Tensor& output) {
  // special case when there is a dim value of 0 in the shape.
  if (output.Shape().Size() == 0)
    return Status::OK();

  auto element_type = input.GetElementType();
  if (element_type == utils::GetONNXTensorElementDataType<float>() ||
      element_type == utils::GetONNXTensorElementDataType<double>()) {
    auto mn = TryTransposeWithCublas(permutations, input.Shape());
    int M = std::get<0>(mn);
    int N = std::get<1>(mn);
    if (M != 0 && N != 0) {
      if (element_type == utils::GetONNXTensorElementDataType<float>()) {
        return TransposeWithCublas<float>(kernel.CublasHandle(), input, output, M, N);
      } else {
        return TransposeWithCublas<double>(kernel.CublasHandle(), input, output, M, N);
      }
    }
  }

  const std::vector<int64_t>& input_dims = input.Shape().GetDims();
  const std::vector<int64_t>& output_dims = output.Shape().GetDims();

  auto rank = input_dims.size();
  CudaAsyncBuffer<int64_t> input_strides(&kernel, rank);
  CudaAsyncBuffer<size_t> perm(&kernel, permutations);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(&kernel, rank);
  ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dims));
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(perm.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  size_t element_size = input.DataType()->Size();
  auto status = TransposeImpl(element_size, rank, input_strides.GpuPtr(), perm.GpuPtr(), input.DataRaw(),
                              fdm_output_strides.GpuPtr(), output.MutableDataRaw(), output.Shape().Size());

  return status;
}

Status Transpose::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X_ptr = ctx->Input<Tensor>(0);
  if (X_ptr == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  const Tensor& X = *X_ptr;
  const TensorShape& input_shape = X.Shape();
  const std::vector<int64_t>& input_dims = input_shape.GetDims();
  size_t rank = input_dims.size();

  std::vector<int64_t> output_dims(rank);
  std::vector<size_t> default_perm(rank);
  const std::vector<size_t>* p_perm = nullptr;
  const auto& status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  return DoTranspose(*this, *p_perm, X, *Y);
}

}  // namespace cuda
}  // namespace onnxruntime
