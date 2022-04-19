// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <xnnpack.h>

#include "core/common/status.h"
#include "core/framework/customregistry.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/xnnpack/build_kernel_info.h"
#include "core/xnnpack/schema/xnnpack_opset.h"

namespace onnxruntime {
namespace xnnpack {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, XnnPackConvolution2d);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, XnnPackDepthwiseConvolution2d);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, XnnPackMaxPooling2d);

static Status InitXnnPack() {
  static xnn_status st = xnn_initialize(nullptr);
  if (st == xnn_status_success) return Status::OK();
  return Status(common::ONNXRUNTIME, common::FAIL, "Init XNNPack failed");
}

Status GetXNNPackRegistry(CustomRegistry** xnnpack_registry) {
  ORT_RETURN_IF_ERROR(InitXnnPack());

  *xnnpack_registry = nullptr;
  std::unique_ptr<CustomRegistry> ret = std::make_unique<CustomRegistry>();
  // Copy the schemas to a vector
  auto schemas = onnxruntime::xnnpack::GetSchemas();
  // Then move it.
  ORT_RETURN_IF_ERROR(ret->RegisterOpSet(schemas, kMSDomain, 0, 1));
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, XnnPackConvolution2d)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1,
                                                            XnnPackDepthwiseConvolution2d)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, XnnPackMaxPooling2d)>,
  };
  for (auto& fn : function_table) {
    auto f = fn();
    ORT_RETURN_IF_ERROR(ret->RegisterCustomKernel(f));
  }
  *xnnpack_registry = ret.release();
  return Status::OK();
}
}  // namespace xnnpack
}  // namespace onnxruntime