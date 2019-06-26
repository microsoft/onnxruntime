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

template <typename T>
Status Expand<T>::ComputeInternal(OpKernelContext* ctx) const {
  const auto& input0 = *ctx->Input<Tensor>(0);
  const auto& input1 = *ctx->Input<Tensor>(1);
  int device_id = GetDeviceId();

  // new shape to be expanded to
  const auto* p_shape = input1.template Data<int64_t>();
  std::vector<int64_t> output_dims{p_shape, p_shape + input1.Shape().Size()};
  TensorShape output_shape(output_dims);

  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), input0.Shape(), output_dims, output_shape));

  // pad input_dims with 1 to make ranks match
  auto rank = output_shape.NumDimensions();
  auto& output_tensor = *ctx->Output(0, output_shape);
  auto input_shape = input0.Shape().GetDims();
  for (int i = 0; i < rank - input_shape.size(); i++) {
    input_shape.insert(input_shape.begin(), 1);
  }

  // create fast_divmod using dimension values
  CudaAsyncBuffer<fast_divmod> fdm_input_dims_gpu(this, device_id, rank);
  CudaAsyncBuffer<fast_divmod> fdm_output_dims_gpu(this, device_id, rank);
  {
    auto in_span = fdm_input_dims_gpu.CpuSpan();
    auto out_span = fdm_output_dims_gpu.CpuSpan();
    for (auto i = 0; i < rank; i++) {
      in_span[i] = fast_divmod(static_cast<int>(input_shape[i]));
      out_span[i] = fast_divmod(static_cast<int>(output_shape[i]));
    }
  }

  ORT_RETURN_IF_ERROR(fdm_input_dims_gpu.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_dims_gpu.CopyToGpu());

  ExpandImpl(
      output_tensor.Shape().NumDimensions(),
      output_tensor.Shape().Size(),
      input0.Shape().Size(),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input0.template Data<T>()),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.template MutableData<T>()),
      fdm_input_dims_gpu.GpuPtr(),
      fdm_output_dims_gpu.GpuPtr());

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
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
};  // namespace onnxruntime
