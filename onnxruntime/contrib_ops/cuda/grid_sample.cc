// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "grid_sample.h"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GridSample,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GridSample<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
GridSample<T>::GridSample(const OpKernelInfo& info) : CudaKernel(info) {
  std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
  std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
  align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
  ORT_ENFORCE(mode_str == "bilinear" || mode_str == "nearest" || mode_str == "bicubic", "mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
  ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection", "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
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

template <typename T>
Status GridSample<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& dims_input = X->Shape().GetDims();

  if (dims_input.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input is expected to have four dimensions (five dimensions is not implemented yet. ) corresponding to [N,C,H,W], got ", dims_input.size());
  }
  const Tensor* Grid = context->Input<Tensor>(1);
  const auto& dims_grid = Grid->Shape().GetDims();

  if (dims_grid.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 is expected to have four dimensions (five dimensions is not implemented yet. ) corresponding to [N,H,W,2], got ", dims_input.size());
  }
  if (dims_grid.data()[3] != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 is expected to have four dimensions (five dimensions is not implemented yet. ) corresponding to [N,H,W,2], got ", dims_input.size());
  }

  std::vector<int64_t> dims_output(4);
  dims_output[0] = dims_grid[0];
  dims_output[1] = dims_input[1];
  dims_output[2] = dims_grid[1];
  dims_output[3] = dims_grid[2];
  Tensor* Y = context->Output(0, dims_output);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  GridSampleImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->Data<T>()),
      reinterpret_cast<const CudaT*>(Grid->Data<T>()),
      mode_i_,
      padding_mode_i_,
      align_corners_,
      dims_input.data(),
      dims_grid[1],
      dims_grid[2],
      Y_data
      );
  return Status::OK();
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
