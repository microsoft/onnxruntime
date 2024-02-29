// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grid_sample.h"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, VERSION, LAYOUT, DOMAIN)          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      GridSample,                                                  \
      DOMAIN,                                                      \
      VERSION,                                                     \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()), \
      onnxruntime::contrib::cuda::GridSample<T, LAYOUT>);

REGISTER_KERNEL_TYPED(float, 1, LAYOUT_NCHW, kMSDomain)
REGISTER_KERNEL_TYPED(float, 16, LAYOUT_NHWC, kMSInternalNHWCDomain)

template <typename T, bool IsNHWC>
GridSample<T, IsNHWC>::GridSample(const OpKernelInfo& info) : CudaKernel(info) {
  std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
  std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
  align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
  ORT_ENFORCE(mode_str == "bilinear" || mode_str == "nearest" || mode_str == "bicubic",
              "mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
  ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection",
              "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
  if (mode_str == "bicubic") {
    mode_i_ = 2;
  } else if (mode_str == "nearest") {
    mode_i_ = 1;
  } else {
    mode_i_ = 0;
  }
  if (padding_mode_str == "reflection") {
    padding_mode_i_ = 2;
  } else if (padding_mode_str == "border") {
    padding_mode_i_ = 1;
  } else {
    padding_mode_i_ = 0;
  }
}

template <typename T, bool IsNHWC>
Status GridSample<T, IsNHWC>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& dims_input = X->Shape().GetDims();
  const Tensor* Grid = context->Input<Tensor>(1);
  const auto& dims_grid = Grid->Shape().GetDims();

  if (dims_input.size() != 4 || dims_grid.size() != 4) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Only 4-D tensor is supported");
  }
  ORT_ENFORCE(dims_grid[0] == dims_input[0], "Grid batch size ", dims_grid[0], " does not match input batch size ", dims_input[0]);
  ORT_ENFORCE(dims_grid[3] == 2, "Last dimension of grid: ", dims_grid[3], ", expect 2");

  using Ch = Channels<IsNHWC>;

  TensorShapeVector dims_output(4);
  dims_output[Ch::N] = dims_input[Ch::N];
  dims_output[Ch::C] = dims_input[Ch::C];
  dims_output[Ch::H] = dims_grid[1 /* Grid::H */];
  dims_output[Ch::W] = dims_grid[2 /* Grid::W */];
  Tensor* Y = context->Output(0, dims_output);
  // Return early if the output tensor is going to be of size 0
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  GridSampleImpl<CudaT, IsNHWC>(
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
}  // namespace contrib

namespace cuda {
REGISTER_KERNEL_TYPED(float, 16, LAYOUT_NCHW, kOnnxDomain)
}  // namespace cuda
}  // namespace onnxruntime
