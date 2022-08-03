/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "hailo_execution_provider.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "hailo_op.h"
#include "hailo_memcpy_op.h"
#include "utils.h"

namespace onnxruntime {

constexpr const char* HAILO = "Hailo";
constexpr const char* HAILO_CPU = "HailoCpu";

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    HailoOp,
    kHailoDomain,
    1,
    kHailoExecutionProvider,
    (*KernelDefBuilder::Create()),
    HailoKernel
);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kHailoExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HailoMemcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kHailoExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HailoMemcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kHailoDomain, 1, HailoOp);

HailoExecutionProvider::HailoExecutionProvider(const HailoExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kHailoExecutionProvider, true}
{
    AllocatorCreationInfo default_memory_info(
        {[](int) {
            return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(HAILO, OrtAllocatorType::OrtDeviceAllocator));
        }},
        0, info.create_arena);

    AllocatorCreationInfo cpu_memory_info(
        {[](int) {
            return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(HAILO_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0,
                                                             OrtMemTypeCPUOutput));
        }},
        0, info.create_arena);

    InsertAllocator(CreateAllocator(default_memory_info));
    InsertAllocator(CreateAllocator(cpu_memory_info));
}

HailoExecutionProvider::~HailoExecutionProvider() {}

InlinedVector<NodeIndex> HailoExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer) const
{
    InlinedVector<NodeIndex> supported_nodes;

    for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
        const auto* p_node = graph_viewer.GetNode(node_index);
        if (p_node == nullptr) {
            continue;
        }
        const auto& node = *p_node;
        const bool supported = m_op_manager.IsNodeSupported(node, graph_viewer);
        LOGS_DEFAULT(INFO) << "Operator type: [" << node.OpType()
                              << "] index: [" << node.Index()
                              << "] name: [" << node.Name()
                              << "] supported: [" << supported
                              << "]";
        if (supported) {
            supported_nodes.push_back(node.Index());
        }
    }

    return supported_nodes;
}

std::vector<std::unique_ptr<ComputeCapability>> HailoExecutionProvider::GetCapability(
    const GraphViewer& graph_viewer, const std::vector<const KernelRegistry*>& kernel_registries) const
{

    ORT_UNUSED_PARAMETER(kernel_registries);
    std::vector<std::unique_ptr<ComputeCapability>> result;

    // We do not run Hailo EP on subgraph
    if (graph_viewer.IsSubgraph()) {
        return result;
    }

    const auto supported_nodes = HailoExecutionProvider::GetSupportedNodes(graph_viewer);

    for (auto& node_index : supported_nodes) {
        auto sub_graph = IndexedSubGraph::Create();
        sub_graph->Nodes().push_back(node_index);
        result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    }

    const auto num_of_supported_nodes = supported_nodes.size();
    const auto summary_msg = MakeString(
        "HailoExecutionProvider::GetCapability,",
        " number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
        " number of nodes supported by Hailo: ", num_of_supported_nodes);

    LOGS_DEFAULT(INFO) << summary_msg;

    return result;
}

static Status RegisterHailoKernels(KernelRegistry& kernel_registry)
{
    static const BuildKernelCreateInfoFn function_table[] = {
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kHailoDomain, 1, HailoOp)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHailoExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
    };

    for (auto& function_table_entry : function_table) {
        ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
    }

    return Status::OK();
}

static std::shared_ptr<KernelRegistry>& HailoKernelRegistry()
{
    // static local variable ensures thread-safe initialization
    static std::shared_ptr<KernelRegistry> hailo_kernel_registry = []() {
        std::shared_ptr<KernelRegistry> registry = KernelRegistry::Create();
        ORT_THROW_IF_ERROR(RegisterHailoKernels(*registry));
        return registry;
    }();

    return hailo_kernel_registry;
}

void Shutdown_DeleteRegistry()
{
    HailoKernelRegistry().reset();
}

std::shared_ptr<KernelRegistry> HailoExecutionProvider::GetKernelRegistry() const
{
    return HailoKernelRegistry();
}

}  // namespace onnxruntime
