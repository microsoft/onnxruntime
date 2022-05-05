// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace utils {

#if !defined(DISABLE_OPTIONAL_TYPE)
common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index);
#endif

// Dummy class for use by an EP that has statically registered NHWC versions of ONNX operators.
// The EP registers a dummy kernel with the ONNX NCHW operator for matching in the first call to
// IExecutionProvider::GetCapability. Following that, GraphPartitioner will convert the layout, resulting in the
// node changing from the ONNX NCHW operator to an kMSInternalNHWCDomain NHWC operator, for which the 'real' kernel
// registration will exist.
//
// Example usage:
//
// // NCHW ONNX operator stub
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep, kOnnxDomain, 11, Conv);
//
// // NHWC 'real' kernels
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep, kMSInternalNHWCDomain, 11, Conv);
//
// std::unique_ptr<KernelRegistry> RegisterKernels() {
//  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();
//  static const BuildKernelCreateInfoFn function_table[] = {
//      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
//      // orig NCHW ONNX op with dummy kernel
//      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep, kOnnxDomain, 11, Conv)>,
//      // 'real' NHWC kernels
//      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep, kMSInternalNHWCDomain, 11, Conv)>,
//      ...
//  };
//  ...
//}

class InvalidNchwKernel : public OpKernel {
 public:
  InvalidNchwKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* /*context*/) const override {
    ORT_THROW(
        "Layout transformation in GraphPartitioner should have replaced this node with one in the "
        "kMSInternalNHWCDomain domain.");
  }
};

}  // namespace utils
}  // namespace onnxruntime
