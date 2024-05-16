// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    S2SModelSplitQuickGelu, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    S2SModelSplitQuickGelu);

// template <typename T>
// void S2SModelSplitQuickGelu::KernelLaunchDispatcher<T>::operator()(cudaStream_t stream, int64_t input_size, int64_t axis,
//                                                                    int64_t alpha, const Tensor& X, const Tensor& S,
//                                                                    Tensor& Y) const {
//   using CudaT = typename ToCudaType<T>::MappedType;
//   LaunchS2SModelSplitQuickGeluKernel<CudaT>(stream, input_size, axis, alpha, reinterpret_cast<const CudaT*>(X.template Data<T>()),
//                                             reinterpret_cast<const CudaT*>(S.template Data<T>()),
//                                             reinterpret_cast<CudaT*>(Y.template MutableData<T>()));
// }

Status S2SModelSplitQuickGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor);
  const auto& input_shape = input_tensor->Shape();
  const auto& input_dims = input_shape.GetDims();
  // Replace it with output count of split?
  const int num_outputs = 2;
  // int64_t axis = HandleNegativeAxis(axis_, input_shape.NumDimensions());

  int num_dims = static_cast<int64_t>(input_dims.size());
  std::vector<int64_t> output_dims(num_dims, 0);
  std::copy(input_dims.begin(), input_dims.end(), output_dims.begin());
  output_dims[num_dims-1] = output_dims[num_dims-1]/2;
  TensorShape output_shape(output_dims);
  auto* output_tensor = context->Output(0, output_shape);

  // auto input_data = input_tensor->DataRaw();

  ORT_RETURN_IF_ERROR(LaunchS2SModelSplitQuickGeluKernel(Stream(context), num_outputs, input_tensor,
                                                         output_tensor));


  return Status::OK();



  // int before_dims = 0;
  // int block_size_including_axis_dim = 0;
  // int block_size_inside_axis_dim = 0;
  // std::vector<int64_t> split_sizes(num_outputs);

  // const Tensor* split_tensor = context->Input<Tensor>(1);
  // if (split_tensor) {
  //   ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "A split tensor must be a vector tensor.");
  //   auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
  //   const int64_t* data = split_tensor->Data<int64_t>();
  //   split_sizes.assign(data, data + nDims);
  // } else {
  //   split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  // }

  // ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
  //                                       num_outputs,
  //                                       axis,
  //                                       before_dims,
  //                                       block_size_including_axis_dim,
  //                                       block_size_inside_axis_dim,
  //                                       split_sizes));

  // auto input_data = input_tensor->DataRaw();

  // auto input_dims = input_shape.GetDims();
  // // Correct this
  // auto output_dimensions{input_shape.AsShapeVector()};
  // CudaAsyncBuffer<void*> output_ptr(this, num_outputs);
  // gsl::span<void*> output_ptr_span = output_ptr.CpuSpan();
  // TensorShapeVector axis_dimension_input_output_mapping(input_dims[axis]);
  // int index = 0;
  // for (int i = 0; i < num_outputs; ++i) {
  //   // update size of dimension for axis we're splitting on
  //   auto split_size = gsl::narrow<int>(split_sizes[i]);
  //   output_dimensions[axis] = split_size;

  //   // Tensor* output = context->Output(TensorShape{output_dimensions});

  //   Tensor* output = context->Output(i, TensorShape{output_dimensions});
  //   auto output_data = output->MutableDataRaw();
  //   output_ptr_span[i] = output_data;
  //   for (int j = 0; j < split_size; ++j) {
  //     axis_dimension_input_output_mapping.at(index++) = i;
  //   }
  // }

  // size_t element_size = input_tensor->DataType()->Size();
  // TArray<void*, 32> output_ptr_array(num_outputs);
  // for (int i = 0; i < num_outputs; ++i) output_ptr_array[i] = output_ptr_span[i];
  // ORT_RETURN_IF_ERROR(LaunchS2SModelSplitQuickGeluKernel(Stream(context), element_size, block_size_including_axis_dim,
  //                                                        block_size_inside_axis_dim, split_sizes[0], num_outputs,
  //                                                        input_data, output_ptr_array,
  //                                                        static_cast<size_t>(input_shape.Size())));

}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
