// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/model.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"

#include <queue>

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "internal_testing_ep_static_kernels.h"  // for BuildKernelCreateInfo declaration

namespace onnxruntime {
namespace internal_testing_ep {

constexpr const char* INTERNAL_TESTING_EP = "InternalTestingEP";

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops,
                                                                   const std::unordered_set<std::string>& stop_ops,
                                                                   DataLayout preferred_layout)
    : IExecutionProvider{utils::kInternalTestingExecutionProvider, true},
      ep_name_{INTERNAL_TESTING_EP},
      ops_{ops},
      stop_ops_{stop_ops},
      preferred_layout_{preferred_layout} {
}

AllocatorPtr InternalTestingExecutionProvider::GetAllocator(int device_id, OrtMemType mem_type) const {
  // replicate setup that some EPs have with a local allocator
  if (mem_type == OrtMemTypeDefault) {
    return local_allocator_;
  } else {
    return IExecutionProvider::GetAllocator(device_id, mem_type);
  }
}

// implement RegisterAllocator to test/validate sharing the CPU EP's allocator
void InternalTestingExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
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
            return std::make_unique<CPUAllocator>(OrtMemoryInfo(INTERNAL_TESTING_EP,
                                                                OrtAllocatorType::OrtDeviceAllocator));
          });

      cpu_alloc = CreateAllocator(allocator_info);

      // enable sharing of our allocator
      allocator_manager.InsertAllocator(cpu_alloc);
    }

    local_allocator_ = cpu_alloc;
    InsertAllocator(cpu_alloc);
  }
}

InternalTestingExecutionProvider::~InternalTestingExecutionProvider() {}

DataLayout InternalTestingExecutionProvider::GetPreferredLayout() const {
  return preferred_layout_;
}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*registries*/) const {
  // find nodes that have ops in our supported list
  std::unordered_set<const Node*> supported_static_nodes;
  std::unordered_set<const Node*> supported_compiled_nodes;

  const auto& topo_nodes = graph_viewer.GetNodesInTopologicalOrder();
  std::for_each(topo_nodes.cbegin(), topo_nodes.cend(),
                [&, this](NodeIndex node_index) {
                  const Node* node = graph_viewer.GetNode(node_index);
                  bool supported = ops_.count(node->OpType()) != 0;
                  if (supported) {
                    if (enable_static_kernels_) {
                      // we have a static kernel for Conv
                      if (node->OpType() == "Conv") {
                        supported_static_nodes.insert(node);
                      }
                    }

                    // all kernels can potentially be compiled in this test setup
                    supported_compiled_nodes.insert(node);
                  }
                });

  if (supported_static_nodes.empty() && supported_compiled_nodes.empty()) {
    return {};
  }

  std::vector<std::unique_ptr<ComputeCapability>> static_capabilities;

  if (enable_static_kernels_) {
#if !defined(ORT_MINIMAL_BUILD)
    std::unordered_set<const Node*> nodes_with_static_kernels;

    // handle any supported nodes we have a static kernel for
    for (const Node* node : supported_static_nodes) {
      bool request_node = false;
      if (node->GetExecutionProviderType() == "") {
        // unassigned node. check if we can handle it.
        if (node->Domain() == kOnnxDomain && node->OpType() == "Conv") {
          request_node = true;
        }
      } else if (node->GetExecutionProviderType() == Type()) {
        // node we selected in the first call to GetCapability. if it was layout sensitive it is now in the
        // kMSInternalNHWCDomain domain
        request_node = true;
      } else {
        // node belongs to another EP
        continue;
      }

      if (request_node) {
        // create a ComputeCapability for the individual node.
        nodes_with_static_kernels.insert(node);

        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node->Index());
        static_capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));

        // in this simple example setup we prefer static kernels over compiled nodes as that's easier to work with
        // for unit tests.
        // most likely a 'real' EP that had both would reverse the order and look for groups of nodes to compile first,
        // and remove those from supported_static_nodes before checking for nodes with static kernels.
        supported_compiled_nodes.erase(node);
      }
    }
#endif
  }

  // NOTE: GetCapability is called for all subgraphs from the bottom up, for one execution provider at a time.
  //       i.e. each execution provider will see the entire model individually.
  // If your execution provider may selectively handle a control flow node (Scan/Loop/If) if it can process all nodes
  // in the subgraph, here would be the place to check if:
  //   - you're processing a subgraph (graph_viewer.IsSubgraph() returns true)
  //   - and all nodes are handled (supported_nodes.size == graph_viewer.NumberOfNodes())
  //
  // If that is the case and you wish to take the control flow node containing the subgraph:
  //   - return an empty vector so the nodes are left as is
  //   - note the node containing the subgraph (graph_viewer.ParentNode()) so that when GetCapability is called for
  //     the graph containing the parent node you can either:
  //     - include that node in supported_nodes if your Compile implementation can handle it potentially
  //       being part of a larger partition; or
  //     - create a ComputeCapability instance for just the control flow node by calling utils::MakeComputeCapability
  //       and adding the instance to the partitions returned by CreateSupportedPartitions

  // create functor to generate a guaranteed unique metadef id
  auto generate_metadef_name = [this, &graph_viewer]() {
    HashValue model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    return ep_name_ + "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
  };

  auto compile_capabilities = utils::CreateSupportedPartitions(graph_viewer, supported_compiled_nodes, stop_ops_,
                                                               generate_metadef_name, ep_name_,
                                                               onnxruntime::utils::kInternalTestingExecutionProvider,
                                                               debug_output_);

  if (!static_capabilities.empty()) {
    std::move(std::begin(static_capabilities), std::end(static_capabilities),
              std::back_inserter(compile_capabilities));
  }

  return compile_capabilities;
}

common::Status InternalTestingExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                                         std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Create a function to generate dummy empty output for each fused node so the model can be executed.
  for (const auto& node_and_viewer : fused_nodes) {
    NodeComputeInfo compute_info;
    const Node& node = node_and_viewer.fused_node;

    if (preferred_layout_ == DataLayout::NHWC) {
      const GraphViewer& graph_viewer = node_and_viewer.filtered_graph;
      auto layout_sensitive_ops = layout_transformer::GetORTLayoutSensitiveOps();
      for (const auto& unfused_node : graph_viewer.Nodes()) {
        if (layout_sensitive_ops.count(unfused_node.OpType()) && unfused_node.Domain() != kMSInternalNHWCDomain) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Found a layout sensitive op which is still in NCHW format. Node: ",
                                 unfused_node.OpType(), " ", unfused_node.Name(),
                                 " The preferred layout for this EP is NHWC. "
                                 "This is a possible bug in layout transformer.");
        }
      }
    }

    compute_info.create_state_func = [](ComputeContext* /*context*/, FunctionState* /*state*/) {
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
    };

    compute_info.compute_func = [&node](FunctionState /*state*/, const OrtApi* c_api,
                                        OrtKernelContext* context) -> Status {
      Ort::CustomOpApi api{*c_api};  // use C++ API for convenience

      const auto outputs = node.OutputDefs();
      const size_t num_outputs = outputs.size();

      for (size_t i = 0; i < num_outputs; i++) {
        const auto* shape_proto = outputs[i]->Shape();
        if (shape_proto == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown output shapes are not supported");
        }

        TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
        if (shape.Size() < 0) {
          // arbitrarily set any unknown dim to 1
          for (size_t idx = 0, end = shape.NumDimensions(); idx < end; ++idx) {
            if (shape[idx] == -1) {
              shape[idx] = 1;
            }
          }
        }

        // create the output_tensor.
        auto* ortvalue = api.KernelContext_GetOutput(context, i, shape.GetDims().data(), shape.GetDims().size());

        // and fill with zeros
        auto* tensor = ortvalue->GetMutable<Tensor>();
        void* data = tensor->MutableDataRaw();
        auto bytes = tensor->SizeInBytes();
        memset(data, 0, bytes);
      };

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// the 'utils::' breaks the kernel registration macros
constexpr const char* internal_testing_ep = utils::kInternalTestingExecutionProvider;

class ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep, kMSInternalNHWCDomain, 11, Conv);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  // make Android build happy...
  // it doesn't count the usage of internal_testing_ep in the macros and generates a warning.
  static_cast<void>(internal_testing_ep);

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(internal_testing_ep,
                                                            kMSInternalNHWCDomain, 11, Conv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

#endif  // !defined(ORT_MINIMAL_BUILD)

std::shared_ptr<KernelRegistry> InternalTestingExecutionProvider::GetKernelRegistry() const {
#if !defined(ORT_MINIMAL_BUILD)
  if (enable_static_kernels_) {
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels();
    return registry;
  }
#endif  // !defined(ORT_MINIMAL_BUILD)
  return nullptr;
}

}  // namespace internal_testing_ep
}  // namespace onnxruntime
