#include "contrib_ops/cuda/my_triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
std::string GetTypeAsString();

#define TYPE_TO_STRING(T,S)           \
  template <>                         \
  std::string GetTypeAsString<T>() {  \
    return S;                         \
  }

TYPE_TO_STRING(MLFloat16, "fp16");
TYPE_TO_STRING(float, "fp32");

template <typename T>
std::string GetMyTritonFunctionName(int64_t block_size) {
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
  input_size = info.GetAttrOrDefault<int64_t>("input_size", int64_t{128});
  block_size = info.GetAttrOrDefault<int64_t>("block_size", int64_t{64});
}

template <typename T>
Status MyTritonKernel<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  std::string function_name = GetMyTritonFunctionName<T>(block_size);
  int64_t grid_size = (X_shape[0] + block_size - 1) / block_size;
  cudaStream_t stream = Stream(ctx);

  struct {
    void* output_ptr;
    const void* input_ptr;
    int32_t input_size_;
  } args = {
    (void*)Y,
    (const void*)X,
    static_cast<int32_t>(input_size),
  };

  return onnxruntime::cuda::LaunchTritonKernel(stream, function_name, grid_size, 1, 1, &args, sizeof(args));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
