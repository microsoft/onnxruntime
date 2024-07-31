// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_kernel.h"

#include "core/providers/vulkan/activation/activations.h"
#include "core/providers/vulkan/math/binary_elementwise.h"

namespace onnxruntime {
namespace vulkan {

namespace {
using IsSupportedFn = std::function<bool(const GraphViewer& graph_viewer,
                                         const onnxruntime::Node&,
                                         const logging::Logger& logger)>;
using CreateFn = std::function<std::unique_ptr<VulkanKernel>(const VulkanExecutionProvider&,
                                                             const onnxruntime::Node&)>;

struct KernelRegistration {
  IsSupportedFn is_supported_fn;
  CreateFn create_fn;
};

std::unordered_map<std::string, KernelRegistration> kernel_registrations = {
    {"Add", {vulkan::BinaryElementwiseKernel::IsSupported, vulkan::BinaryElementwiseKernel::CreateAdd}},
    {"Mul", {vulkan::BinaryElementwiseKernel::IsSupported, vulkan::BinaryElementwiseKernel::CreateMul}},
    {"Sigmoid", {vulkan::SigmoidKernel::IsSupported, vulkan::SigmoidKernel::Create}},
};

Status CreateNcnnLayer(const std::string_view layer_name, std::unique_ptr<ncnn::Layer>& layer) {
  int index = -1;
  index = ncnn::layer_to_index(layer_name.data());
  if (index == -1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to find NCNN layer for ", layer_name);
  }

  layer.reset(ncnn::create_layer_vulkan(index));

  if (!layer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create NCNN layer for ", layer_name);
  }

  return Status::OK();
}
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
                            const GraphViewer& graph_viewer,
                            const onnxruntime::Node& node,
                            ValueIndexes& value_indexes,
                            std::unique_ptr<VulkanKernel>& kernel) {
  const auto& op = node.OpType();

  auto it = kernel_registrations.find(op);
  // temporary sanity check. We should have checked IsSupported before calling Create.
  assert(it != kernel_registrations.end());

  kernel = it->second.create_fn(vulkan_ep, node);

  if (!kernel) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create Vulkan kernel for ", node.OpType());
  }

  return kernel->SetupNcnnLayer(graph_viewer, value_indexes);
}

Status VulkanKernel::SetupNcnnLayer(const GraphViewer& graph_viewer, ValueIndexes& value_indexes) {
  ORT_RETURN_IF_ERROR(SetupParamDict(graph_viewer, params_));
  ORT_RETURN_IF_ERROR(CreateNcnnLayer(GetNcnnLayerName(), ncnn_layer_));

  ncnn_layer_->vkdev = &vulkan_ep_.Device();

  RETURN_IF_NCNN_ERROR(ncnn_layer_->load_param(params_));

  ORT_RETURN_IF_ERROR(SetupConstantInitializers(graph_viewer, *ncnn_layer_));

  // we manually set shape hints instead of using load_model as we handle initializers ourselves given they're coming
  // from the ONNX model and not the NCNN model.
  auto [input_shapes, output_shapes] = GetLayerShapeHints(node_);
  ncnn_layer_->bottom_shapes = input_shapes;
  ncnn_layer_->top_shapes = output_shapes;

  RETURN_IF_NCNN_ERROR(ncnn_layer_->create_pipeline(vulkan_ep_.NcnnOptions()));

  // set input/output values indexes in the Layer's bottoms and tops.
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
