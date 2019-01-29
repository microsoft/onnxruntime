// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "trt_execution_provider.h"
#include "trt_allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/platform/env.h"
#include "onnx/shape_inference/implementation.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "gsl/pointers"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime
{
GraphProto ToGraphProto(const GraphViewer& graph)
{

    GraphProto graph_proto;
    graph_proto.clear_node();

    //Nodes is sorted in Topological Order in the GraphProto
    for (auto& node_idx : graph.GetNodesInTopologicalOrder())
    {
        const Node* p_node = graph.GetNode(node_idx);
        NodeProto* node_proto{graph_proto.add_node()};
        p_node->ToProto(*node_proto);
    }

    graph_proto.clear_input();
    graph_proto.clear_output();
    graph_proto.clear_value_info();

    for (const auto* input_arg : graph.GetInputsIncludingInitializers())
    {
        *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
    }

    for (const auto* output_arg : graph.GetOutputs())
    {
        *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
    }

    for (const auto* value_info : graph.GetValueInfo())
    {
        *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
    }

    return graph_proto;
}

//This custom kernel is a temporary implementation.
//TODO: change to the new function implementation.
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTRTExecutionProvider, kMSDomain, 1, TRTKernel);

ONNX_OPERATOR_KERNEL_EX(TRTKernel,
                        kMSDomain,
                        1,
                        kTRTExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetDefaultInputsMemoryType(OrtMemTypeCPUInput).SetDefaultOutputMemoryType(OrtMemTypeCPUOutput),
                        TRTKernel);

void RegisterTRTOperatorKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTRTExecutionProvider, kMSDomain, 1, TRTKernel)>());//kOnnxDomain
}

TRTExecutionProvider::TRTExecutionProvider()
{
    DeviceAllocatorRegistrationInfo trt_device_info({OrtMemTypeCPU, [](int)
    {
        return std::make_unique<TRTPinnedAllocator>();
    }, std::numeric_limits<size_t>::max()
                                                    });
    InsertAllocator(CreateAllocator(trt_device_info));
    DeviceAllocatorRegistrationInfo default_device_info({OrtMemTypeDefault, [](int)
    {
        return std::make_unique<TRTAllocator>();
    }, std::numeric_limits<size_t>::max()
                                                        });
    InsertAllocator(CreateAllocator(default_device_info));
}

TRTExecutionProvider::~TRTExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
        TRTExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const
{
    //Construct modelproto from graph and serialize to string
    ModelProto model_proto;
    GraphProto graph_proto = ToGraphProto(graph);
    model_proto.set_allocated_graph(&graph_proto);
    string string_buf;
    model_proto.SerializeToString(&string_buf);
    std::vector<char> onnx_buf(string_buf.begin(), string_buf.end());

    //Get supported node list
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    TRTLogger trt_logger(static_cast<nvinfer1::ILogger::Severity>(verbosity));
    const auto& trt_builder = InferObject(nvinfer1::createInferBuilder(trt_logger));
    const auto& trt_network = InferObject(trt_builder->createNetwork());
    const auto& trt_parser  = InferObject(nvonnxparser::createParser(trt_network.get(), trt_logger));

    SubGraphCollection_t SupportedNodesVector;
    trt_parser->supportsModel(onnx_buf.data(), onnx_buf.size(), SupportedNodesVector);
    model_proto.release_graph();

    std::vector<std::unique_ptr<ComputeCapability>> result;
    for (auto& group : SupportedNodesVector)
    {
        //Find inputs, initializer and outputs for each supported subgraph
        std::set<size_t> node_set(group.begin(), group.end());
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        if (!group.empty())
        {
            std::set<const NodeArg*> fused_inputs, fused_outputs, erased;
            for (auto& index : group)
            {
                sub_graph->nodes.push_back(index);
                const auto& node = graph.GetNode(index);
                for (const auto& input : node->InputDefs())
                {
                    const auto& it = fused_outputs.find(input);
                    if (it != fused_outputs.end())
                    {
                        fused_outputs.erase(it);
                        erased.insert(input);
                    }
                    else if (erased.find(input) == erased.end())
                    {
                        fused_inputs.insert(input);
                    }
                }

                for (const auto& output : node->OutputDefs())
                {
                    const auto& it = fused_inputs.find(output);
                    if (it != fused_inputs.end())
                    {
                        fused_inputs.erase(it);
                        erased.insert(output);
                    }
                    else if (erased.find(output) == erased.end())
                    {
                        fused_outputs.insert(output);
                    }
                }
            }

            //Add node's outputs to subgraph's outputs if their EdgeEnd nodes don't belong to the subgraph
            for (auto& index : node_set)
            {
                const auto& node = graph.GetNode(index);
                for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it)
                {
                    auto& node_arg = (it->GetNode()).InputDefs();
                    if (node_set.find((it->GetNode()).Index()) == node_set.end())
                    {
                        int arg_index = it->GetDstArgIndex();
                        fused_outputs.insert(node_arg[arg_index]);
                    }
                }
            }

            //Assign inputs and outputs to subgraph's meta_def
            auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
            meta_def->name = "TRTKernel";
            meta_def->domain = kMSDomain;
            for (auto& input : fused_inputs)
            {
                meta_def->inputs.push_back(input->Name());
            }
            for (auto& output : fused_outputs)
            {
                meta_def->outputs.push_back(output->Name());
            }

            meta_def->since_version = 1;
            sub_graph->SetMetaDef(meta_def);
        }
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }

    return result;
}

std::shared_ptr<KernelRegistry> GetTRTKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterTRTOperatorKernels(*kernel_registry);
  return kernel_registry;
}

std::shared_ptr<KernelRegistry> TRTExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::GetTRTKernelRegistry();
  return kernel_registry;
}

common::Status TRTExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const
{
    ORT_UNUSED_PARAMETER(src);
    ORT_UNUSED_PARAMETER(dst);
    return Status::OK();
}
}  // namespace onnxruntime

