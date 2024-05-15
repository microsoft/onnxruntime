#include "contrib_ops/cuda/my_triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
std::string GetTypeAsString();

template <>
std::string GetTypeAsString<float>() {
  return "fp32";
}

template <>
std::string GetTypeAsString<MLFloat16>() {
  return "fp16";
}

template <typename T>
std::string GetSoftmaxTritonFunctionName(int64_t block_size) {
  std::string ret = "my_triton_kernel_";
  ret += GetTypeAsString<T>();
  ret += "_" + std::to_string(block_size);

  return ret;
}

}  // end of namespace

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MyTritonKernel,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MyTritonKernel<T>);

REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);

template <typename T>
MyTritonKernel<T>::MyTritonKernel(const OpKernelInfo& info) : CudaKernel{info} {
  input_step_size = info.GetAttrOrDefault<int64_t>("input_step_size", int64_t{10});
  output_step_size = info.GetAttrOrDefault<int64_t>("output_step_size", int64_t{10});
  mask_size = info.GetAttrOrDefault<int64_t>("mask_size", int64_t{10});
  batch_size = info.GetAttrOrDefault<int64_t>("batch_size", int64_t{10});
  block_size = info.GetAttrOrDefault<int64_t>("block_size", int64_t{1024});
}

template <typename T>
Status MyTritonKernel<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto Y_data = reinterpret_cast<CudaT*>(Y);
  auto X_data = reinterpret_cast<const CudaT*>(X);

  // const int64_t N = X_shape.SizeToDimension(0);
  // const int64_t D = X_shape.SizeFromDimension(0);

  // int64_t block_size = D - 1;
  // block_size |= block_size >> 1;
  // block_size |= block_size >> 2;
  // block_size |= block_size >> 4;
  // block_size |= block_size >> 8;
  // block_size |= block_size >> 16;
  // block_size |= block_size >> 32;
  // block_size += 1;
  // block_size = block_size >= 1024 ? block_size : 1024;
  // block_size = block_size <= 16384 ? block_size : 16384;

  std::string function_name = GetSoftmaxTritonFunctionName<T>(block_size);

  // construct args for launch kernel
  struct {
    void* out;
    const void* in;
    int in_stride;
    int out_stride;
    int n_cols;
  } args = {
    (void*)Y_data,
    (const void*)X_data,
    static_cast<int32_t>(input_step_size),
    static_cast<int32_t>(output_step_size),
    static_cast<int32_t>(mask_size),
    // static_cast<int32_t>(D),
    // static_cast<int32_t>(D),
    // static_cast<int32_t>(D)
  };

  cudaStream_t stream = Stream(ctx);

  // grid size is (n_rows, 1, 1), meaning the kernel should be called once per row
  // return onnxruntime::cuda::LaunchTritonKernel(stream, function_name, N, 1, 1, &args, sizeof(args));
  return onnxruntime::cuda::LaunchTritonKernel(stream, function_name, batch_size, 1, 1, &args, sizeof(args));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
