// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/pad_and_unflatten.h"
#include "orttraining/training_ops/cuda/tensor/pad_and_unflatten_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    PadAndUnflatten,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T_INT", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T_INDEX", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    PadAndUnflatten);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct PadAndUnflattenFunctor {
  void operator()(cudaStream_t stream,
                  const int64_t input_element_count,
                  const fast_divmod output_element_stride_fdm,
                  const int64_t index_value_upper_bound,
                  const Tensor& input_tensor,
                  const Tensor& indices_tensor,
                  Tensor& output_tensor) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_tensor.Data<T>());

    CUDA_CALL_THROW(cudaMemset(output_tensor.MutableDataRaw(), 0, output_tensor.Shape().Size() * sizeof(CudaT)));
    PadAndUnflattenImpl<CudaT>(stream, input_element_count, output_element_stride_fdm, index_value_upper_bound,
                               input_data, indices_tensor.Data<int64_t>(),
                               reinterpret_cast<CudaT*>(output_tensor.MutableData<T>()));
  }
};

}  // namespace

Status PadAndUnflatten::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* indices_tensor = context->Input<Tensor>(1);
  const Tensor* unflatten_dims_tensor = context->Input<Tensor>(2);  // Parse the 1-D shape tensor.
  ORT_ENFORCE(unflatten_dims_tensor->Shape().NumDimensions() == 1,
              "unflatten_dims_tensor tensor must be 1-D.", unflatten_dims_tensor->Shape().NumDimensions());
  ORT_ENFORCE(unflatten_dims_tensor->Shape().Size() == 2,
              "unflatten_dims_tensor tensor must contain 2 values.", unflatten_dims_tensor->Shape().Size());

  const int64_t* dims_ptr = unflatten_dims_tensor->Data<int64_t>();
  const auto& input_shape = input_tensor->Shape();
  ORT_ENFORCE(input_shape[0] == indices_tensor->Shape()[0],
              "The first dimension of input and indices must be the same.");

  std::vector<int64_t> output_shape_vec;
  output_shape_vec.push_back(dims_ptr[0]);
  output_shape_vec.push_back(dims_ptr[1]);

  const int64_t flatten_dim_factor = dims_ptr[0] * dims_ptr[1];

  int64_t element_stride = 1;
  for (size_t i = 1; i < input_shape.NumDimensions(); ++i) {
    output_shape_vec.push_back(input_shape[i]);
    element_stride *= input_shape[i];
  }

  fast_divmod output_element_stride_fdm(static_cast<int>(element_stride));
  auto output_shape = TensorShape(output_shape_vec);
  Tensor* output_tensor = context->Output(0, output_shape);

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<PadAndUnflattenFunctor>(Stream(context),
                                        input_shape.Size(),
                                        output_element_stride_fdm,
                                        flatten_dim_factor,
                                        *input_tensor,
                                        *indices_tensor,
                                        *output_tensor);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
