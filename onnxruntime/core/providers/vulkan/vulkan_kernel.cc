// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_kernel.h"

#include "core/providers/vulkan/activation/activations.h"

namespace onnxruntime {
namespace vulkan {

namespace {
using IsSupportedFn = std::function<bool(const onnxruntime::Node&, const logging::Logger& logger)>;
using CreateFn = std::function<std::unique_ptr<VulkanKernel>(const VulkanExecutionProvider&,
                                                             const onnxruntime::Node&,
                                                             std::unique_ptr<ncnn::Layer>)>;

struct KernelRegistration {
  IsSupportedFn is_supported_fn;
  CreateFn create_fn;
};

std::unordered_map<std::string, KernelRegistration> kernel_registrations = {
    {"Sigmoid", {vulkan::SigmoidKernel::IsSupported, vulkan::SigmoidKernel::Create}},
};

}  // namespace

bool VulkanKernel::IsSupported(const onnxruntime::Node& node, const logging::Logger& logger) {
  // start with fp32 only.
  if (node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }

  bool supported = false;
  if (auto it = kernel_registrations.find(node.OpType()); it != kernel_registrations.end()) {
    supported = it->second.is_supported_fn(node, logger);
  }

  return supported;
}

common::Status VulkanKernel::Create(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node,
                                    ValueIndexes& value_indexes, std::unique_ptr<VulkanKernel>& kernel) {
  const auto& op = node.OpType();

  auto it = kernel_registrations.find(op);
  // temporary sanity check. We should have checked IsSupported before calling Create.
  assert(it != kernel_registrations.end());

  auto layer_idx = GetNcnnLayerIndex(op);
  if (layer_idx == -1) {
    // should never happen outside of during development
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to find layer for ", op, " in the NCNN kernels.");
  }

  std::unique_ptr<ncnn::Layer> layer(ncnn::create_layer_vulkan(layer_idx));

  if (!layer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create NCNN layer for ", op);
  }

  kernel = it->second.create_fn(vulkan_ep, node, std::move(layer));
  if (!kernel) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create Vulkan kernel for ", node.OpType());
  }

  return kernel->Init(value_indexes);
}

Status VulkanKernel::Init(ValueIndexes& value_indexes) {
  ncnn_layer_->vkdev = &vulkan_ep_.Device();

  // Alternative approach is to create ncnn::ParamDict and call load_params.
  // Try manually setting first for simplicity.
  // If for some reason an ONNX input value maps to a param it would not be available in SetupLayer (which we expect
  // to be called from the ctor) we'd need to create a pipeline on a per-Run basis.
  // Can consider this if required. Not supported for now.
  // ncnn::ParamDict params;
  // ncnn_layer_->load_param(params);

  // we manually set shape hints instead of using load_model as we handle initializers ourselves given they're coming
  // from the ONNX model and not the NCNN model.
  auto [input_shapes, output_shapes] = GetLayerShapeHints(node_);
  ncnn_layer_->bottom_shapes = input_shapes;
  ncnn_layer_->top_shapes = output_shapes;

  RETURN_IF_NCNN_ERROR(ncnn_layer_->create_pipeline(vulkan_ep_.NcnnOptions()));

  for (const NodeArg*& def : node_.InputDefs()) {
    auto entry = value_indexes.find(def->Name());
    assert(entry != value_indexes.end());  // should be impossible for an input value to not be in the map
    ncnn_layer_->bottoms.push_back(entry->second);
  }

  for (const NodeArg*& def : node_.OutputDefs()) {
    ncnn_layer_->tops.push_back(value_indexes.Add(*def));
  }

  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
