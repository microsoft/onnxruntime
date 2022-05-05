// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_execution_provider.h"
#include "detail/utils.h"
#include "detail/op_support_checker.h"

#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/session_options.h"
#include "core/providers/shared/node_unit/node_unit.h"

#include <xnnpack.h>

namespace onnxruntime {

namespace xnnpack {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// define the class name for the NCHW stub and the 'real' NHWC kernel. same for the BuildKernelCreateInfo entries.
//
// NOTE: If KernelRegistry::HasImplementationOf supported overriding Node.Domain() we wouldn't need to have a dummy
//       registration in the ONNX domain as we could call HasImplementationOf to match the kMSInternalNHWCDomain kernel
//       in GetCapability. The NHWC schema is copied from the ONNX schema, so all the values relevant to kernel
//       matching are the same.
#define VERSIONED_KERNEL_CLASS_NAME(Start, End, Op)                                                        \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, Start, End, Op); \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, End, Op)

#define KERNEL_CLASS_NAME(Start, Op)                                                        \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, Start, Op); \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, Op)

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op)                                                      \
  BuildKernelCreateInfo<                                                                                  \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, Start, End, Op)>, \
      BuildKernelCreateInfo<                                                                              \
          ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op)                                                      \
  BuildKernelCreateInfo<                                                                   \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, Start, Op)>, \
      BuildKernelCreateInfo<                                                               \
          ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, Op)>

VERSIONED_KERNEL_CLASS_NAME(1, 10, Conv);
KERNEL_CLASS_NAME(11, Conv);

VERSIONED_KERNEL_CLASS_NAME(1, 7, MaxPool);
VERSIONED_KERNEL_CLASS_NAME(8, 9, MaxPool);
VERSIONED_KERNEL_CLASS_NAME(10, 10, MaxPool);
VERSIONED_KERNEL_CLASS_NAME(11, 11, MaxPool);
KERNEL_CLASS_NAME(12, MaxPool);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      KERNEL_CREATE_INFO_VERSIONED(1, 10, Conv),
      KERNEL_CREATE_INFO(11, Conv),

      KERNEL_CREATE_INFO_VERSIONED(1, 7, MaxPool),
      KERNEL_CREATE_INFO_VERSIONED(8, 9, MaxPool),
      KERNEL_CREATE_INFO_VERSIONED(10, 10, MaxPool),
      KERNEL_CREATE_INFO_VERSIONED(11, 11, MaxPool),
      KERNEL_CREATE_INFO(12, MaxPool),
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

XnnpackExecutionProvider::XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& info)
    : IExecutionProvider{kXnnpackExecutionProvider, true},
      session_options_{info.session_options} {
  // TODO: Could/should we provide our default CPU allocator to this call via an adapter?
  xnn_status st = xnn_initialize(nullptr);
  if (st != xnn_status_success) {
    ORT_THROW("XNNPACK initialization failed with status ", st);
  }

  // TODO: Allocation planner calls GetAllocator for the individual EP. It would be better if it goes through
  // the session state to get the allocator so it's per-device (or for the allocation planner to try the EP first
  // and fall back to using session state next by passing in a functor it can use to call SessionState::GetAllocator).
  // That way we only need one allocator per-device unless the EP needs/wants to override any previously registered
  // allocator. Which also means one arena per-device if the per-EP allocators are using arenas.

  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(kXnnpackExecutionProvider,
                                                            OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>> XnnpackExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  std::shared_ptr<KernelRegistry> registry = GetKernelRegistry();
  std::unordered_set<const Node*> supported_nodes;
  NodeSupportChecker checker{graph, supported_nodes};

  // L2 optimizations include fusing Conv or MaxPool with Clip or Relu.
  // check that the session options are available, and if so whether L2 optimizations are enabled.
  // If they're not available we can't assume the fusion will occur, so we can't take the activation node.
  bool l2_optimizations_enabled = session_options_ &&
                                  session_options_->graph_optimization_level >= TransformerLevel::Level2;

  std::unordered_map<const Node*, ComputeCapability*> node_to_compute_capability;

  // Get all the NodeUnits in the GraphViewer so we can check if something is in a QDQ node group
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph);

  for (NodeIndex idx : graph.GetNodesInTopologicalOrder()) {
    const Node* n = graph.GetNode(idx);
    if (n == nullptr) {
      continue;
    }

    // node is part of a QDQ group. when we implement support for quantized ops we could potentially handle the
    // QDQ group. until then we have to leave it as-is otherwise we'll make performance worse.
    // e.g. If we took the MaxPool from DQ -> MaxPool -> Q we are forcing it to run in fp32 instead of allowing a QDQ
    // aware EP to convert the group into a single quantized MaxPool node.
    if (node_unit_map[n]->UnitType() == NodeUnit::Type::QDQGroup) {
      continue;
    }

    const Node& node = *n;
    bool request_node = false;

    if (node.GetExecutionProviderType() == "") {
      // unassigned node. check if we have a kernel registration for it.
      if (KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        // we have a kernel registration for the operator, and any type constraints have been satisfied.
        // check other aspects of the node such as the attributes to make sure it is supported.
        request_node = checker.IsNodeSupported(node);
      } else {
        // see if it's an activation we can fuse with a node we support. note that we can only do this after
        // the layout transform as we need to fuse with the NWHC op that we have the real kernel for.
        if (l2_optimizations_enabled) {
          const Node* fuse_with = checker.IsNodeSupportedWithFusion(node);
          if (fuse_with) {
            // add new MetaDef to existing ComputeCapability.
            // we know an entry must exist in node_to_compute_capability as we update supported_nodes
            // when creating the ComputeCapability, and the logic in IsNodeSupportedWithFusion
            // checks the fuse_with node is in supported_nodes.
            auto iter = node_to_compute_capability.find(fuse_with);
            ORT_ENFORCE(iter != node_to_compute_capability.cend(),
                        "node_to_compute_capability is not is sync with supported_nodes. ");

            // update the MetaDef to cover the nodes being fused.
            // the fused node will have OpType:'Conv' and Domain:kMSInternalNHWCDomain.
            // GraphPartitioner will match the statically registered xnnpack NHWC Conv kernel instead of
            // calling IExecutionProvider::Compile
            ComputeCapability& capability = *iter->second;
            capability.sub_graph->SetMetaDef(FuseActivation(*fuse_with, node, graph));
            capability.sub_graph->nodes.push_back(node.Index());
            capability.sub_graph->SetUseExistingSchema(true);
          }
        }
      }
    } else if (node.GetExecutionProviderType() == Type()) {
      // second call to GetCapability after layout changes.
      // as we requested the node in the first call, it should be supported in the second call.
      request_node = true;
    } else {
      // node belongs to another EP
      continue;
    }

    if (request_node) {
      // create a ComputeCapability for the individual node.
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));

      node_to_compute_capability.insert({&node, capabilities.back().get()});
      supported_nodes.insert(&node);
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
