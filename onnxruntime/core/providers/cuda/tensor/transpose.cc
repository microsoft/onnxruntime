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
std::tuple<int, int> TryTransposeWithCublas(const std::vector<size_t>& perm, const TensorShape& input_shape) {
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
  int32_t rank = gsl::narrow_cast<int32_t>(input_dims.size());

  std::vector<int64_t> output_dims(rank);
  std::vector<size_t> default_perm(rank);
  const std::vector<size_t>* p_perm = nullptr;
  const auto& status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor* Y = ctx->Output(0, output_shape);

  auto mn = TryTransposeWithCublas(*p_perm, input_shape);
  int M = std::get<0>(mn);
  int N = std::get<1>(mn);
  if (M != 0 && N != 0) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);
    const CudaT* input_data = reinterpret_cast<const CudaT*>(X.template Data<T>());
    CudaT* output_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    CUBLAS_RETURN_IF_ERROR(
        cublasTransposeHelper(
            CublasHandle(),
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            M,
            N,
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

  TensorPitches original_input_strides(input_dims);
  TensorPitches original_output_strides(output_dims);

  ORT_ENFORCE(rank <= MAX_ARRAY_SIZE);
  TArray<int64_t> input_strides(rank);
  for (auto i = 0; i < rank; i++) {
    input_strides.data_[i] = original_input_strides[(*p_perm)[i]];
  }
  TArray<fast_divmod> output_strides(rank);
  for (auto i = 0; i < rank; i++) {
    output_strides.data_[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }

  TransposeImpl(
      rank,
      output_shape.Size(),
      input_strides,
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.template Data<T>()),
      output_strides,
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y->template MutableData<T>()));

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Transpose<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
