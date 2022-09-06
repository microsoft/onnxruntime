// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/graph/function_utils.h"
#include "xnnpack_execution_provider.h"
#include "detail/utils.h"
#include "detail/node_support_checker.h"

#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/shared/node_unit/node_unit.h"

#include <xnnpack.h>

namespace onnxruntime {

namespace xnnpack {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, type, Op)>

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 12, Softmax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 13, Softmax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 10, uint8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 10, int8_t, QLinearConv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 1, QLinearAveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider,
                                      kDynamicDomainByCreate, 1, QLinearSoftmax);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      KERNEL_CREATE_INFO(11, Conv),
      KERNEL_CREATE_INFO_VERSIONED(11, 11, MaxPool),
      KERNEL_CREATE_INFO(12, MaxPool),
      KERNEL_CREATE_INFO(11, AveragePool),
      // layout insensitive, use ONNX-domain directly
      BuildKernelCreateInfo<
          ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 13, Softmax)>,
      BuildKernelCreateInfo<
          ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 12, Softmax)>,

      //  quantization op
      KERNEL_CREATE_INFO_TYPED(10, uint8_t, QLinearConv),
      KERNEL_CREATE_INFO_TYPED(10, int8_t, QLinearConv),
      KERNEL_CREATE_INFO(1, QLinearAveragePool),
      BuildKernelCreateInfo<
          ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kDynamicDomainByCreate, 1, QLinearSoftmax)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

}  // namespace xnnpack

using namespace xnnpack;

XnnpackExecutionProvider::XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& /*info*/)
    : IExecutionProvider{kXnnpackExecutionProvider, true} {
}

// implement RegisterAllocator to test/validate sharing the CPU EP's allocator
void XnnpackExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice cpu_device{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};

  // if EP is used in multiple inference sessions we may already have an allocator. if so use that.
  auto cpu_alloc = GetAllocator(cpu_device.Id(), OrtMemTypeDefault);
  if (!cpu_alloc) {
    // use shared allocator if available
    cpu_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, cpu_device);

    if (!cpu_alloc) {
      // create our allocator
      AllocatorCreationInfo allocator_info(
          [](int) {
            return std::make_unique<CPUAllocator>(OrtMemoryInfo(kXnnpackExecutionProvider,
                                                                OrtAllocatorType::OrtDeviceAllocator));
          });

      cpu_alloc = CreateAllocator(allocator_info);
      // enable sharing of our allocator
      allocator_manager.InsertAllocator(cpu_alloc);
    }

    InsertAllocator(cpu_alloc);
  }

  // TODO: Create `struct xnn_allocator` that wraps cpu_allocator, and provide in the call to xnn_initialize so that
  //       xnnpack is using the ORT allocator.
  xnn_status st = xnn_initialize(nullptr);
  if (st != xnn_status_success) {
    ORT_THROW("XNNPACK initialization failed with status ", st);
  }
}

// For ops are not lay-out sensitive and does not defined in
// onnx-domain, it will be created dynamicly
static bool RequestDynamicSchema(const NodeUnit& node_unit) {
  static const InlinedHashSet<std::string_view> dynamic_schema_set = {"QLinearSoftmax"};
  std::string key = node_unit.UnitType() == NodeUnit::Type::QDQGroup
                                                ? "QLinear" + node_unit.OpType() : node_unit.OpType();
  return dynamic_schema_set.contains(key);
}

// Add Compute Capability for the second call. All target nodes have the tag of "XnnpackExecutionProvider"
// after the first call. So we are going to do QDQ fusion in the second call
// All nodes was collected in one sub_graph
static void AddComputeCapabilityForNodeUnit(const NodeUnit& node_unit,
                                            const std::function<void(std::unique_ptr<IndexedSubGraph>)>& adder,
                                            std::unordered_map<const Node*, const NodeUnit*>& supported_map) {
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  auto process_node = [&sub_graph, &supported_map, &node_unit](const Node& node) {
    sub_graph->nodes.push_back(node.Index());
    supported_map.emplace(&node, &node_unit);
  };

  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    for (const auto* node_i : node_unit.GetAllNodesInGroup()) {
      process_node(*node_i);
    }
    sub_graph->SetMetaDef(FuseQDQGroup(node_unit));
  } else {
    process_node(node_unit.GetNode());
  }

  sub_graph->schema_source = RequestDynamicSchema(node_unit)
                                 ? IndexedSubGraph::SourceOfSchema::REUSE_OR_CREATE
                                 : IndexedSubGraph::SourceOfSchema::EXISTING;
  adder(std::move(sub_graph));
}

// The first call to add compute capability in GetCapability, we just tell this all nodes in nodeunit
// are supported by Xnnpack EP as long as it's target node is supported.
// One node in one sub_graph separately
static void AddComputeCapabilityForEachNodeInNodeUnit(
    const NodeUnit& node_unit,
    std::function<void(std::unique_ptr<IndexedSubGraph>)> adder,
    std::unordered_map<const Node*, const NodeUnit*>& supported_map) {
  auto process_node = [&adder, &node_unit, &supported_map](const Node& node) {
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node.Index());
    adder(std::move(sub_graph));
    supported_map.emplace(&node, &node_unit);
  };
  for (const auto* node_i : node_unit.GetAllNodesInGroup()) {
    process_node(*node_i);
  }
}

std::vector<std::unique_ptr<ComputeCapability>> XnnpackExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  std::shared_ptr<KernelRegistry> registry = GetKernelRegistry();
  std::unordered_map<const Node*, const NodeUnit*> supported_node_unit_map;
  NodeSupportChecker checker{graph, supported_node_unit_map};
  std::unordered_map<const NodeUnit*, ComputeCapability*> node_to_compute_capability;

  // Get all the NodeUnits in the GraphViewer so we can check if something is in a QDQ node group
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph);

  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit being checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_holder.size());
  for (NodeIndex idx : graph.GetNodesInTopologicalOrder()) {
    const Node* n = graph.GetNode(idx);
    if (n == nullptr) {
      continue;
    }
    // if node is part of a QDQ group,
    // we will mark it compatible in the first call as long as we support the target node.
    const NodeUnit& node_unit = *node_unit_map[n];

    bool request_node = false;
    // any node in NodeUnit will trigger IsNodeSupported, so we just check once.
    if (node_unit_supported_result.count(&node_unit)) {
      continue;
    } else if (node_unit.GetNode().GetExecutionProviderType() == "") {
      // unassigned node.
      // check if this is an ONNX operator that we have an NHWC xnnpack kernel for.
      if (checker.IsNodeSupported(node_unit)) {
        request_node = true;
      } else {
        // see if it's an activation we can fuse with a node we support. note that we can only do this after
        // the layout transform as we need to fuse with the NWHC op that we have the real kernel for.
        const NodeUnit* fuse_with = checker.IsNodeSupportedWithFusion(node_unit);
        if (fuse_with) {
          // add new MetaDef to existing ComputeCapability.
          // we know an entry must exist in node_to_compute_capability as we update supported_node_unit_map
          // when creating the ComputeCapability, and the logic in IsNodeSupportedWithFusion
          // checks the fuse_with node is in supported_node_unit_map.
          auto iter = node_to_compute_capability.find(fuse_with);
          ORT_ENFORCE(iter != node_to_compute_capability.cend(),
                      "node_to_compute_capability is not in sync with supported_node_unit_map.");

          // update the MetaDef to cover the nodes being fused.
          // the fused node will have OpType:'Conv' and Domain:kMSInternalNHWCDomain.
          // GraphPartitioner will match the statically registered xnnpack NHWC Conv kernel instead of
          // calling IExecutionProvider::Compile
          ComputeCapability& capability = *iter->second;
          capability.sub_graph->SetMetaDef(FuseActivation(*fuse_with, node_unit, graph));
          capability.sub_graph->nodes.push_back(node_unit.Index());
          capability.sub_graph->schema_source = IndexedSubGraph::SourceOfSchema::EXISTING;
        }
      }
    } else if (node_unit.GetNode().GetExecutionProviderType() == Type()) {
      // second call to GetCapability after layout changes.
      // as we requested the node in the first call, it should be supported in the second call.
      request_node = true;
    } else {
      // node belongs to another EP
      continue;
    }
    node_unit_supported_result[&node_unit] = request_node;
    if (request_node) {
      // Create ComputeCapability from IndexedSubGraph and add
      auto add_capability = [&](std::unique_ptr<IndexedSubGraph> sub_graph) {
        capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
        node_to_compute_capability.insert({&node_unit, capabilities.back().get()});
      };

      // first pass: add ComputeCapability for all individual nodes in NodeUnit
      if (node_unit.GetNode().GetExecutionProviderType().empty()) {
        AddComputeCapabilityForEachNodeInNodeUnit(node_unit, add_capability, supported_node_unit_map);
      } else {  // == Type()
        // second pass: add single ComputeCapability for all nodes in NodeUnit so any QDQ node groups get fused
        // Activation fusion happens later
        AddComputeCapabilityForNodeUnit(node_unit, add_capability, supported_node_unit_map);
      }
    }
  }

  // FUTURE: finding nodes to compile can be inserted here and added to the ComputeCapability instances returned.
  // GraphPartitioner can handle a mix of static and compiled kernels.

  return capabilities;
}

std::shared_ptr<KernelRegistry> XnnpackExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = xnnpack::RegisterKernels();
  return registry;
}

XnnpackExecutionProvider::~XnnpackExecutionProvider() {
  xnn_deinitialize();
}

}  // namespace onnxruntime
