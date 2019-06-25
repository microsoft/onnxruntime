// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand.h"
#include "expand_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Expand,                                                    \
      kOnnxDomain,                                               \
      8,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(1),               \
      Expand<T>);

void PrintData2(int32_t* buffer, const void* data, int64_t count, std::string name) {
  cudaMemcpy(buffer, data, count, cudaMemcpyDeviceToHost);
  std::cout << name << ":" << std::endl;
  for (int64_t i = 0; i < count; ++i) {
    std::cout << buffer[i] << std::endl;
  }
}

static Status ComputeOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t out_dim = std::max(lhs_dim, rhs_dim);
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}

template <typename T>
Status Expand<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto& input0 = *ctx->Input<Tensor>(0);
  const auto& input1 = *ctx->Input<Tensor>(1);
  int device_id = GetDeviceId();

  // new shape to be expanded to
  const auto* p_shape = input1.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input1.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(onnxruntime::cuda::ComputeOutputShape(Node().Name(), input0.Shape(), output_dims, output_shape));

  // pad input_dims with 1 to make ranks match
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto input_dims = input0.Shape().GetDims();
  for (int i = 0; i < output_shape.GetDims().size() - input_dims.size(); i++) {
    input_dims.insert(input_dims.begin(), 1);
  }

  CudaAsyncBuffer<int64_t> input_dims_gpu(this, device_id, input_dims);
  CudaAsyncBuffer<int64_t> output_dims_gpu(this, device_id, output_shape.GetDims());
  ORT_RETURN_IF_ERROR(input_dims_gpu.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_dims_gpu.CopyToGpu());

  ExpandImpl(
      output_tensor.Shape().NumDimensions(),
      output_tensor.Shape().Size(),
      input0.Shape().Size(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input0.template Data<T>()),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.template MutableData<T>()),
      input_dims_gpu.GpuPtr(),
      output_dims_gpu.GpuPtr());
  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Expand<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
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
};  // namespace onnxruntime
