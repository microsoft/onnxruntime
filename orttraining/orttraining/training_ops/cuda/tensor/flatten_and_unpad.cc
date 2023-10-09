// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/flatten_and_unpad.h"
#include "orttraining/training_ops/cuda/tensor/flatten_and_unpad_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    FlattenAndUnpad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t, MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T_INT", DataTypeImpl::GetTensorType<int64_t>())
        .OutputMemoryType(OrtMemTypeCPUOutput, 1),
    FlattenAndUnpad);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct FlattenAndUnpadFunctor {
  void operator()(cudaStream_t stream,
                  const int64_t output_element_count,
                  const fast_divmod output_element_stride_fdm,
                  const int64_t index_value_upper_bound,
                  const Tensor& input_tensor,
                  const Tensor& indices_tensor,
                  Tensor& output_tensor) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_tensor.Data<T>());

    FlattenAndUnpadImpl<CudaT>(stream, output_element_count, output_element_stride_fdm, index_value_upper_bound,
                               input_data, indices_tensor.Data<int64_t>(),
                               reinterpret_cast<CudaT*>(output_tensor.MutableData<T>()));
  }
};

}  // namespace

Status FlattenAndUnpad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* indices_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(indices_tensor->Shape().NumDimensions() == 1,
              "indices_tensor tensor must be 1-D.", indices_tensor->Shape().NumDimensions());

  std::vector<int64_t> output_shape_vec;
  output_shape_vec.push_back(indices_tensor->Shape()[0]);
  const auto& input_shape = input_tensor->Shape();
  int64_t element_stride = 1;
  for (size_t i = 2; i < input_shape.NumDimensions(); ++i) {
    output_shape_vec.push_back(input_shape[i]);
    element_stride *= input_shape[i];
  }

  fast_divmod output_element_stride_fdm(static_cast<int>(element_stride));
  auto output_shape = TensorShape(output_shape_vec);
  Tensor* output_tensor = context->Output(0, output_shape);

  std::vector<int64_t> unflatten_dims_vec;
  unflatten_dims_vec.push_back(input_shape[0]);
  unflatten_dims_vec.push_back(input_shape[1]);
  const int64_t index_value_upper_bound = input_shape[0] * input_shape[1];

  utils::MLTypeCallDispatcher<int32_t, int64_t, float, MLFloat16, double, BFloat16>
      t_disp(input_tensor->GetElementType());
  t_disp.Invoke<FlattenAndUnpadFunctor>(Stream(context),
                                        output_shape.Size(),
                                        output_element_stride_fdm,
                                        index_value_upper_bound,
                                        *input_tensor,
                                        *indices_tensor,
                                        *output_tensor);

  size_t rank = unflatten_dims_vec.size();
  Tensor* unflatten_dims_tensor = context->Output(1, {static_cast<int>(rank)});
  TensorShape(unflatten_dims_vec).CopyDims(unflatten_dims_tensor->MutableData<int64_t>(), rank);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
