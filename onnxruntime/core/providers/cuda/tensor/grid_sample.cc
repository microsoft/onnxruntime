// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grid_sample.h"
#include "core/providers/cpu/tensor/grid_sample.h"
#include "grid_sample_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T)                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(GridSample, kOnnxDomain, 16, 19, T, kCudaExecutionProvider,     \
                                    (*KernelDefBuilder::Create())                                \
                                        .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
                                        .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
                                    GridSample<T>);

#define REGISTER_KERNEL_TYPED_20(T)                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(GridSample, kOnnxDomain, 20, T, kCudaExecutionProvider,      \
                                (*KernelDefBuilder::Create())                                \
                                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
                                    .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
                                GridSample<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED_20(float)

template <typename T>
GridSample<T>::GridSample(const OpKernelInfo& info) : CudaKernel(info) {
  typename onnxruntime::GridSample<T>::GridSampleInterpolationMode mode{onnxruntime::GridSample<T>::Linear};
  typename onnxruntime::GridSample<T>::GridSamplePaddingMode padding_mode{onnxruntime::GridSample<T>::Zeros};
  std::tie(mode, padding_mode, align_corners_) = onnxruntime::GridSample<T>::ParseAttributes(info);
  mode_i_ = static_cast<int>(mode);
  padding_mode_i_ = static_cast<int>(padding_mode);
}

template <typename T>
Status GridSample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& dims_input = X->Shape().GetDims();
  const Tensor* Grid = context->Input<Tensor>(1);
  const auto& dims_grid = Grid->Shape().GetDims();

  if (dims_input.size() != 4 || dims_grid.size() != 4) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Only 4-D tensor is supported");
  }
  ORT_ENFORCE(dims_grid[0] == dims_input[0], "Grid batch size ", dims_grid[0], " does not match input batch size ", dims_input[0]);
  ORT_ENFORCE(dims_grid[3] == 2, "Last dimension of grid: ", dims_grid[3], ", expect 2");

  TensorShapeVector dims_output(4);
  dims_output[0] = dims_input[0];
  dims_output[1] = dims_input[1];
  dims_output[2] = dims_grid[1];
  dims_output[3] = dims_grid[2];
  Tensor* Y = context->Output(0, dims_output);
  // Return early if the output tensor is going to be of size 0
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  GridSampleImpl<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(X->Data<T>()),
      reinterpret_cast<const CudaT*>(Grid->Data<T>()),
      mode_i_,
      padding_mode_i_,
      align_corners_,
      dims_input.data(),
      dims_grid[1],
      dims_grid[2],
      Y_data);
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
