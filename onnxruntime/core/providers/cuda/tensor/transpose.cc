// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose.h"
#include "transpose_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Transpose,                                                  \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Transpose<T>);

// special case acceleration using cublas matrix transpose
template <typename T>
Status TryCublasTranspose(cublasHandle_t handle,
                          const std::vector<size_t>& perm,
                          const TensorShape& input_shape,
                          const typename ToCudaType<T>::MappedType* input_data,
                          typename ToCudaType<T>::MappedType* output_data,
                          bool* is_transposed);

// calculate dimensions to use in cublas matrix transpose
std::tuple<int, int> CalculateCublasTransposeDimensions(const std::vector<size_t>& perm, const TensorShape& input_shape) {
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
Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const {
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

  // special case when there is a dim value of 0 in the shape.
  if (output_shape.Size() == 0)
    return Status::OK();

  bool is_transposed = false;
  ORT_RETURN_IF_ERROR(TryCublasTranspose<T>(CublasHandle(),
                                            *p_perm,
                                            input_shape,
                                            reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.template Data<T>()),
                                            reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->template MutableData<T>()),
                                            &is_transposed));
  if (is_transposed) {
    return Status::OK();
  }

  CudaAsyncBuffer<int64_t> input_strides(this, rank);
  CudaAsyncBuffer<size_t> perm(this, *p_perm);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, rank);
  ORT_ENFORCE(TensorPitches::Calculate(input_strides.CpuSpan(), input_dims));
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));

  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(perm.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  TransposeImpl(
      rank,
      input_strides.GpuPtr(),
      perm.GpuPtr(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.template Data<T>()),
      fdm_output_strides.GpuPtr(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->template MutableData<T>()),
      output_shape.Size());

  return Status::OK();
}

#define CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(T)                                               \
  template <>                                                                              \
  Status TryCublasTranspose<T>(cublasHandle_t /* handle */,                                \
                               const std::vector<size_t>& /* perm */,                      \
                               const TensorShape& /* input_shape */,                       \
                               const typename ToCudaType<T>::MappedType* /* input_data */, \
                               typename ToCudaType<T>::MappedType* /* output_data */,      \
                               bool* is_transposed) {                                      \
    *is_transposed = false;                                                                \
    return Status::OK();                                                                   \
  }

#define CUBLAS_TRANSPOSE_IMPL_SUPPORTED(T)                                      \
  template <>                                                                   \
  Status TryCublasTranspose<T>(cublasHandle_t handle,                           \
                               const std::vector<size_t>& perm,                 \
                               const TensorShape& input_shape,                  \
                               const ToCudaType<T>::MappedType* input_data,     \
                               ToCudaType<T>::MappedType* output_data,          \
                               bool* is_transposed) {                           \
    auto mn = CalculateCublasTransposeDimensions(perm, input_shape);            \
    int M = std::get<0>(mn);                                                    \
    int N = std::get<1>(mn);                                                    \
    if (M == 0 || N == 0) {                                                     \
      *is_transposed = false;                                                   \
    } else {                                                                    \
      typename ToCudaType<T>::MappedType one = ToCudaType<T>::FromFloat(1.0f);  \
      typename ToCudaType<T>::MappedType zero = ToCudaType<T>::FromFloat(0.0f); \
      CUBLAS_RETURN_IF_ERROR(                                                   \
          cublasTransposeHelper(                                                \
              handle,                                                           \
              CUBLAS_OP_T,                                                      \
              CUBLAS_OP_T,                                                      \
              M,                                                                \
              N,                                                                \
              &one,                                                             \
              input_data,                                                       \
              N,                                                                \
              &zero,                                                            \
              input_data,                                                       \
              N,                                                                \
              output_data,                                                      \
              M));                                                              \
      *is_transposed = true;                                                    \
    }                                                                           \
    return Status::OK();                                                        \
  }

CUBLAS_TRANSPOSE_IMPL_SUPPORTED(float)
CUBLAS_TRANSPOSE_IMPL_SUPPORTED(double)
CUBLAS_TRANSPOSE_IMPL_SUPPORTED(MLFloat16)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(int8_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(int16_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(int32_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(int64_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(uint8_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(uint16_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(uint32_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(uint64_t)
CUBLAS_TRANSPOSE_IMPL_UNSUPPORTED(bool)

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)
SPECIALIZED_COMPUTE(int8_t)
SPECIALIZED_COMPUTE(int16_t)
SPECIALIZED_COMPUTE(int32_t)
SPECIALIZED_COMPUTE(int64_t)
SPECIALIZED_COMPUTE(uint8_t)
SPECIALIZED_COMPUTE(uint16_t)
SPECIALIZED_COMPUTE(uint32_t)
SPECIALIZED_COMPUTE(uint64_t)
SPECIALIZED_COMPUTE(bool)

}  // namespace cuda
}  // namespace onnxruntime
