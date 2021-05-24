// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class Inverse final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit Inverse(const OpKernelInfo& info) : CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  using Base = CudaKernel;
  using CublasHandle = cublasHandle_t;

  template <typename T>
  struct ComputeImpl;
};

ONNX_OPERATOR_KERNEL_EX(
    Inverse,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16>()),
    Inverse);

namespace inverse_internal {

template <typename T>
Status ComputeMatrixOffsets(cudaStream_t stream, T* workspace_data, size_t num_batches, size_t rows, IAllocatorUniquePtr<T*>& matrix_ptrs) {
  std::vector<T*> cuda_ptrs;
  const size_t matrix_size = rows * rows;
  for (size_t i = 0; i < num_batches; ++i) {
    cuda_ptrs.push_back(workspace_data);
    workspace_data += matrix_size;
  }

  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(matrix_ptrs.get(), cuda_ptrs.data(), sizeof(T*) * num_batches,
                                       cudaMemcpyHostToDevice, stream));
  return Status::OK();
}

Status CheckForSingularity(cudaStream_t stream, const IAllocatorUniquePtr<int>& info, const std::unique_ptr<int[]>& info_cpu, size_t num_batches) {
  // Let's check if any of the info values is non-zero
  // cudaMemcpyAsync from device memory to pageable host memory will return only once the copy has completed.
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(info_cpu.get(), info.get(), sizeof(int) * num_batches,
                                       cudaMemcpyDeviceToHost, stream));
  for (size_t i = 0; i < num_batches; ++i) {
    if (info_cpu[i] != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Matrix is singular at batch:", i);
    }
  }
  return Status::OK();
}

}  // namespace inverse_internal

template <typename T>
struct Inverse::ComputeImpl {
  Status operator()(cudaStream_t stream, Inverse::CublasHandle cublas_h, const Inverse* inst, const Tensor& input, Tensor& output,
                    const IAllocatorUniquePtr<int>& info, const IAllocatorUniquePtr<int>& pivots,
                    size_t num_batches, size_t rows) const {
    using namespace onnxruntime::cuda;
    using namespace inverse_internal;
    using CudaT = typename ToCudaType<T>::MappedType;
    const size_t input_count = static_cast<size_t>(input.Shape().Size());
    auto info_cpu = std::make_unique<int[]>(num_batches);
    const auto dim = static_cast<int>(rows);
    const auto n_batches = static_cast<int>(num_batches);

    // Make a copy of the input which will serve as a workspace as well.
    if (std::is_same<T, float>::value || std::is_same<T, MLFloat16>::value) {
      IAllocatorUniquePtr<float> input_workspace = inst->GetScratchBuffer<float>(input_count);
      if (std::is_same<T, MLFloat16>::value) {
        // Convert from MLFloat16(half) to float
        Impl_Cast<CudaT, float>(stream, reinterpret_cast<const CudaT*>(input.Data<MLFloat16>()), input_workspace.get(), input_count);
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_workspace.get(), input.Data<float>(), sizeof(float) * input_count,
                                             cudaMemcpyDeviceToDevice, stream));
      }
      IAllocatorUniquePtr<float*> matrix_ptrs = inst->GetScratchBuffer<float*>(n_batches);
      ORT_RETURN_IF_ERROR(ComputeMatrixOffsets<float>(stream, input_workspace.get(), num_batches, rows, matrix_ptrs));
      // Do LU factorization
      CUBLAS_RETURN_IF_ERROR(cublasSgetrfBatched(cublas_h, dim, matrix_ptrs.get(), dim, pivots.get(), info.get(), n_batches));
      ORT_RETURN_IF_ERROR(CheckForSingularity(stream, info, info_cpu, num_batches));

      // Need to compute ptrs for output buffers
      // Output for MLFloat
      IAllocatorUniquePtr<float*> output_ptrs = inst->GetScratchBuffer<float*>(n_batches);
      if (std::is_same<T, MLFloat16>::value) {
        IAllocatorUniquePtr<float> ml_float_output = inst->GetScratchBuffer<float>(input_count);
        ORT_RETURN_IF_ERROR(ComputeMatrixOffsets<float>(stream, ml_float_output.get(), num_batches, rows, output_ptrs));
        // Do the inverse
        CUBLAS_RETURN_IF_ERROR(cublasSgetriBatched(cublas_h, dim, matrix_ptrs.get(), dim, pivots.get(), output_ptrs.get(), dim, info.get(), n_batches));
        ORT_RETURN_IF_ERROR(CheckForSingularity(stream, info, info_cpu, num_batches));
        // Copy the result to output with casting
        Impl_Cast<float, CudaT>(stream, ml_float_output.get(), reinterpret_cast<CudaT*>(output.MutableData<MLFloat16>()), input_count);
        // We are done here
      } else {
        ORT_RETURN_IF_ERROR(ComputeMatrixOffsets<float>(stream, output.MutableData<float>(), num_batches, rows, output_ptrs));
        // Do the inverse
        CUBLAS_RETURN_IF_ERROR(cublasSgetriBatched(cublas_h, dim, matrix_ptrs.get(), dim, pivots.get(), output_ptrs.get(), dim, info.get(), n_batches));
        ORT_RETURN_IF_ERROR(CheckForSingularity(stream, info, info_cpu, num_batches));
        // We are done here
      }
    } else if (std::is_same<T, double>::value) {
      IAllocatorUniquePtr<double> input_workspace = inst->GetScratchBuffer<double>(static_cast<int>(input_count));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input_workspace.get(), input.Data<double>(), sizeof(double) * input_count,
                                           cudaMemcpyDeviceToDevice, stream));

      IAllocatorUniquePtr<double*> matrix_ptrs = inst->GetScratchBuffer<double*>(n_batches);
      ORT_RETURN_IF_ERROR(ComputeMatrixOffsets<double>(stream, input_workspace.get(), num_batches, rows, matrix_ptrs));
      // Do LU factorization
      CUBLAS_RETURN_IF_ERROR(cublasDgetrfBatched(cublas_h, dim, matrix_ptrs.get(), dim, pivots.get(), info.get(), n_batches));
      ORT_RETURN_IF_ERROR(CheckForSingularity(stream, info, info_cpu, num_batches));

      // Need to compute ptrs for output buffers
      IAllocatorUniquePtr<double*> output_ptrs = inst->GetScratchBuffer<double*>(n_batches);
      ORT_RETURN_IF_ERROR(ComputeMatrixOffsets<double>(stream, output.MutableData<double>(), num_batches, rows, output_ptrs));
      CUBLAS_RETURN_IF_ERROR(cublasDgetriBatched(cublas_h, dim, matrix_ptrs.get(), dim, pivots.get(), output_ptrs.get(), dim, info.get(), n_batches));
      ORT_RETURN_IF_ERROR(CheckForSingularity(stream, info, info_cpu, num_batches));
      // We are done here
    } else {
      ORT_THROW("Type is not supported");
    }
    return Status::OK();
  }
};

Status Inverse::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input = ctx->Input<Tensor>(0);
  const auto& input_shape = input->Shape();
  const auto num_dim = input_shape.NumDimensions();
  auto* output = ctx->Output(0, input_shape);

  size_t num_batches = 1;
  const size_t rows = static_cast<size_t>(input_shape.GetDims()[num_dim - 2]);
  const size_t cols = static_cast<size_t>(input_shape.GetDims()[num_dim - 1]);
  ORT_ENFORCE(rows == cols, "Expecting square matrices");
  if (num_dim > 2) {
    num_batches = static_cast<size_t>(input_shape.SizeToDimension(num_dim - 2));
  }

  IAllocatorUniquePtr<int> info = GetScratchBuffer<int>(num_batches);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(info.get(), 0, num_batches, Stream()));
  IAllocatorUniquePtr<int> pivots = GetScratchBuffer<int>(rows * num_batches);

  utils::MLTypeCallDispatcher<float, double, MLFloat16> t_disp(input->GetElementType());
  return t_disp.InvokeRet<Status, ComputeImpl>(
      Stream(), Base::CublasHandle(), this, *input, *output, info, pivots, num_batches, rows);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
