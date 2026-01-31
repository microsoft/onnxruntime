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

#define REGISTER_KERNEL_VERSIONED_TYPED(T, FROM_VERSION, TO_VERSION, LAYOUT, DOMAIN) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                           \
      GridSample,                                                                    \
      DOMAIN,                                                                        \
      FROM_VERSION,                                                                  \
      TO_VERSION,                                                                    \
      T,                                                                             \
      kCudaExecutionProvider,                                                        \
      (*KernelDefBuilder::Create())                                                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())                    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),                   \
      onnxruntime::contrib::cuda::GridSample<T, LAYOUT>);

REGISTER_KERNEL_TYPED(float, 1, LAYOUT_NCHW, kMSDomain)

#ifdef ENABLE_CUDA_NHWC_OPS
// Op was introduced in opset 16
REGISTER_KERNEL_VERSIONED_TYPED(float, 16, 19, LAYOUT_NHWC, kMSInternalNHWCDomain)

// Op was modified to support multiple spatial dimensions in opset 20
REGISTER_KERNEL_VERSIONED_TYPED(float, 20, 21, LAYOUT_NHWC, kMSInternalNHWCDomain)

// Op spec introduced BFloat16 support in opset 22
REGISTER_KERNEL_TYPED(float, 22, LAYOUT_NHWC, kMSInternalNHWCDomain)
#endif

template <typename T, bool IsNHWC>
GridSample<T, IsNHWC>::GridSample(const OpKernelInfo& info) : CudaKernel(info) {
  opset_start_version_ = info.node().SinceVersion();

  std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
  std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
  align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));

  if (opset_start_version_ >= 20) {
    std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "linear");
    if (mode_str == "cubic") {
      mode_i_ = 2;
    } else if (mode_str == "nearest") {
      mode_i_ = 1;
    } else if (mode_str == "linear") {
      mode_i_ = 0;
    } else {
      ORT_THROW("mode \"", mode_str, "\" not supported, expect linear, nearest or cubic");
    }
  } else {
    std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
    if (mode_str == "bicubic") {
      mode_i_ = 2;
    } else if (mode_str == "nearest") {
      mode_i_ = 1;
    } else if (mode_str == "bilinear") {
      mode_i_ = 0;
    } else {
      ORT_THROW("mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
    }
  }

  ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection",
              "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
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

  if (dims_input.size() != dims_grid.size()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Input and grid must have the same number of dimensions");
  }

  if (opset_start_version_ < 20 && dims_input.size() != 4) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Opset 16-19 versions of this op only supports 4-D input tensors");
  }

  if (dims_input[0] != dims_grid[0]) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Grid batch size does not match input batch size ");
  }

  if ((dims_input.size() == 4 && dims_grid[3] != 2) || (dims_input.size() == 5 && dims_grid[4] != 3)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Last dimension of grid input must match the number of "
                  "spatial dimensions in the input (2 for 2D, 3 for 3D).");
  }

  if (dims_input.size() != 4 && dims_input.size() != 5) {
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Only 4-D and 5-D input tensors are supported");
  }

  if (dims_input.size() == 5 && mode_i_ == 2) {
    // This is common for CPU and CUDA to not support Cubic mode for 5D input
    // So it won't break CUDA users who were previously dropping down to CPU version of the op.
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Cubic mode is only supported in 4-D cases.");
  }

  using Ch = Channels<IsNHWC>;

  TensorShapeVector dims_output(dims_input.size());
  if (dims_input.size() == 4) {
    dims_output[Ch::N] = dims_input[Ch::N];
    dims_output[Ch::C] = dims_input[Ch::C];
    dims_output[Ch::H] = dims_grid[1 /* Grid::H */];
    dims_output[Ch::W] = dims_grid[2 /* Grid::W */];
  } else {
    // 5D input - deal with both NCHW and NHWC layouts
    dims_output[0] = dims_input[0];
    dims_output[1] = !IsNHWC ? dims_input[1] : dims_grid[1];
    dims_output[2] = !IsNHWC ? dims_grid[1] : dims_grid[2];
    dims_output[3] = !IsNHWC ? dims_grid[2] : dims_grid[3];
    dims_output[4] = !IsNHWC ? dims_grid[3] : dims_input[4];
  }
  Tensor* Y = context->Output(0, dims_output);

  // Return early if the output tensor is going to be of size 0
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  if (dims_input.size() == 4) {
    // sample 2d
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
  } else {
    // sample 3d
    GridSampleImpl3D<CudaT, IsNHWC>(
        Stream(context),
        reinterpret_cast<const CudaT*>(X->Data<T>()),
        reinterpret_cast<const CudaT*>(Grid->Data<T>()),
        mode_i_,
        padding_mode_i_,
        align_corners_,
        dims_input.data(),
        dims_grid[1],
        dims_grid[2],
        dims_grid[3],
        Y_data);
  }

  return Status::OK();
}
}  // namespace cuda
}  // namespace contrib

namespace cuda {
// Op was introduced in opset 16
REGISTER_KERNEL_VERSIONED_TYPED(float, 16, 19, LAYOUT_NCHW, kOnnxDomain)

// Op was modified to support multiple spatial dimensions in opset 20
REGISTER_KERNEL_VERSIONED_TYPED(float, 20, 21, LAYOUT_NCHW, kOnnxDomain)

// Op spec introduced BFloat16 support in opset 22
REGISTER_KERNEL_TYPED(float, 22, LAYOUT_NCHW, kOnnxDomain)
}  // namespace cuda
}  // namespace onnxruntime
