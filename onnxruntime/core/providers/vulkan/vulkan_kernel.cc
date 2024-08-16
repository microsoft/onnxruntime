// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_kernel.h"

#include "core/common/logging/logging.h"
#include "core/providers/vulkan/math/matmul.h"

namespace onnxruntime {
namespace vulkan {

namespace {
using IsSupportedFn = std::function<bool(const GraphViewer&,
                                         const onnxruntime::Node&,
                                         const logging::Logger&)>;

using CreateFn = std::function<std::unique_ptr<VulkanKernel>(const VulkanExecutionProvider&,
                                                             const GraphViewer*,
                                                             const onnxruntime::Node&)>;

struct KernelRegistration {
  IsSupportedFn is_supported_fn;
  CreateFn create_fn;
};

#define REGISTER_KERNEL_SIMPLE(op)                       \
  {                                                      \
    #op, { op##Kernel::IsSupported, op##Kernel::Create } \
  }

#define REGISTER_KERNEL(op, impl_class, create_fn)          \
  {                                                         \
    #op, { impl_class::IsSupported, impl_class::create_fn } \
  }

std::unordered_map<std::string, KernelRegistration> kernel_registrations = {
    // REGISTER_KERNEL(Add, BinaryElementwiseKernel, CreateAdd),
    REGISTER_KERNEL_SIMPLE(MatMul),
    // REGISTER_KERNEL(Mul, BinaryElementwiseKernel, CreateMul),
};

}  // namespace

bool VulkanKernel::IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                               const logging::Logger& logger) {
  // start with fp32 only.
  if (node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << node.OpType() << ": unsupported input data type.";
    return false;
  }

  bool supported = false;
  if (auto it = kernel_registrations.find(node.OpType()); it != kernel_registrations.end()) {
    supported = it->second.is_supported_fn(graph_viewer, node, logger);
  }

  return supported;
}

Status VulkanKernel::Create(const VulkanExecutionProvider& vulkan_ep,
                            const GraphViewer* graph_viewer,
                            const onnxruntime::Node& node,
                            ValueIndexes& /*value_indexes*/,
                            std::unique_ptr<VulkanKernel>& kernel) {
  const auto& op = node.OpType();

  auto it = kernel_registrations.find(op);
  // sanity check. We should have checked IsSupported before calling Create.
  assert(it != kernel_registrations.end());

  kernel = it->second.create_fn(vulkan_ep, graph_viewer, node);

  if (!kernel) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create Vulkan kernel for ", node.OpType());
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
