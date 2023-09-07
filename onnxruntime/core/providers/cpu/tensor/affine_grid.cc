//// Copyright (c) Microsoft Corporation. All rights reserved.
//// Licensed under the MIT License.
//
// #include "core/providers/cpu/tensor/affine_grid.h"
//
// #include "core/common/common.h"
// #include "core/providers/op_kernel_type_control.h"
// #include "core/util/math_cpuonly.h"
//
// namespace onnxruntime {
//
// namespace op_kernel_type_control {
// ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
//    kCpuExecutionProvider, kOnnxDomain, AffineGrid, Output, 0,
//    float, double);
//}
//
// using EnabledEyeLikeDataTypes = ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
//    kCpuExecutionProvider, kOnnxDomain, AffineGrid, Output, 0);
//
// ONNX_CPU_OPERATOR_KERNEL(
//    AffineGrid,
//    9,
//    KernelDefBuilder()
//        .TypeConstraint(
//            "T1",
//            BuildKernelDefConstraintsFromTypeList<EnabledEyeLikeDataTypes>())
//        .TypeConstraint(
//            "T2",
//            BuildKernelDefConstraintsFromTypeList<EnabledEyeLikeDataTypes>()),
//    AffineGrid);
//
//
// Status AffineGrid::Compute(OpKernelContext* context) const {
//  const auto& theta = context->RequiredInput<Tensor>(0);
//
//  const auto& theta_shape = theta.Shape();
//
//  if (theta_shape.NumDimensions() != 3) {
//    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "AffineGrid : Input theta tensor dimension is not 3");
//  }
//
//  const auto& size = context->RequiredInput<Tensor>(1);
//  const auto& size_shape = size.Shape();
//  int64_t N, C, D, H, W;
//  if (size_shape.NumDimensions() == 4 && get_check_2d_grid_sample_consistency(theta_shape, size_shape, N, C, H, W)) {
//
//  } else if (size_shape.NumDimensions() == 5 && get_check_3d_grid_sample_consistency(theta_shape, size_shape, N, C, D, H, W)) {
//
//  } else {
//    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "AffineGrid : size theta tensor dimension is not 3");
//  }
//
//
//  // set output tensor shape same as input tensor
//  auto& T2 = context->RequiredOutput(0, input_shape);
//
//  const auto output_tensor_dtype =
//      has_dtype_ ? static_cast<ONNX_NAMESPACE::TensorProto::DataType>(dtype_) : T1.GetElementType();
//
//  utils::MLTypeCallDispatcherFromTypeList<EnabledEyeLikeDataTypes> dispatcher{output_tensor_dtype};
//  dispatcher.Invoke<ComputeDispatchTarget>(k_, T2);
//
//  return Status::OK();
//}
//
//}  // namespace onnxruntime
