#include "contrib_ops/cuda/my_triton_softmax.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

std::string GetSoftmaxTritonFunctionName(int32_t element_type, int64_t block_size) {
  std::string ret = "my_triton_softmax_";

  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    ret += "fp32";
  }
  else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    ret += "fp16";
  }
  else {
    ret += "unknowntype_" + std::to_string(element_type);
  }
  ret += "_" + std::to_string(block_size);

  return ret;
}

}  // end of namespace

ONNX_OPERATOR_KERNEL_EX(
  MyTritonSoftmax,
  kOnnxDomain,
  1,
  kCudaExecutionProvider,
  (*KernelDefBuilder::Create()).TypeConstraint(
    "T",
    BuildKernelDefConstraints<float, MLFloat16>()),
  MyTritonSoftmax);

Status MyTritonSoftmax::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  const int64_t n_rows = X_shape.GetDims()[0];
  const int64_t n_cols = X_shape.GetDims()[1];

  int64_t next_power_of_2 = n_cols - 1;
  next_power_of_2 |= next_power_of_2 >> 1;
  next_power_of_2 |= next_power_of_2 >> 2;
  next_power_of_2 |= next_power_of_2 >> 4;
  next_power_of_2 |= next_power_of_2 >> 8;
  next_power_of_2 |= next_power_of_2 >> 16;
  next_power_of_2 |= next_power_of_2 >> 32;
  next_power_of_2 += 1;

  const int64_t block_size = next_power_of_2;

  std::string function_name = GetSoftmaxTritonFunctionName(X->GetElementType(), block_size);

  // construct args for launch kernel
  struct {
    void* out;
    const void* in;
    int in_stride;
    int out_stride;
    int n_cols;
  } args = {(void*)Y, (const void*)Y, static_cast<int32_t>(n_cols), static_cast<int32_t>(n_cols), static_cast<int32_t>(n_cols)};

  // grid size is (n_rows, 1, 1), meaning the kernel should be called once per row
  return onnxruntime::cuda::LaunchTritonKernel(Stream(ctx), function_name, n_rows, 1, 1, &args, sizeof(args));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
