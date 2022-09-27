#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "ExecutionProvider.h"
#include "DmlGraphFusionTransformer.h"
#include "GraphPartitioner.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_lookup.h"
#include "FusedGraphKernel.h"

namespace Dml
{
    DmlGraphFusionTransformer::DmlGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    )
        :onnxruntime::GraphTransformer(name),
         m_providerImpl(static_cast<const ExecutionProvider*>(provider)->GetImpl())
    {

    }
	
    std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>>
    GetInitializerToPartitionMap(
        const onnxruntime::GraphViewer& graph,
        gsl::span<std::unique_ptr<GraphPartition>> partitions
    )
    {
        std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>> initializerPartitionMap;

        for (uint32_t partitionIndex = 0; partitionIndex < gsl::narrow_cast<uint32_t>(partitions.size()); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            // Skip partitions which have been merged into other partitions
            if (partition->GetRootMergedPartition() != partition.get())
            {
                continue;
            }

            for (const std::string& input : partition->GetInputs())
            {
                const onnx::TensorProto* tensor = nullptr;
                if (graph.GetInitializedTensor(input, tensor))
                {
                    initializerPartitionMap[tensor].push_back(partitionIndex);
                }
            }
        }

        return initializerPartitionMap;
    }

    void CreateONNXGraphForPartition(
        onnxruntime::Graph* partitionGraph,
        onnxruntime::Graph& mainGraph,
        const onnxruntime::IndexedSubGraph& nodes_to_fuse)
    {
        auto* meta_def = nodes_to_fuse.GetMetaDef();

        int i = 0;
        std::vector<const onnxruntime::NodeArg*> partitionGraphInputs;
        partitionGraphInputs.resize(meta_def->inputs.size());
        for (auto& input : meta_def->inputs) 
        {
            auto input_arg = mainGraph.GetNodeArg(input);
            auto& function_body_graph_input_arg = partitionGraph->GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
            partitionGraphInputs[i] = &function_body_graph_input_arg;
            ++i;
        }

        i = 0;
        std::vector<const onnxruntime::NodeArg*> partitionGraphOutputs;
        partitionGraphOutputs.resize(meta_def->outputs.size());
        for (auto& output : meta_def->outputs) 
        {
            auto output_arg = mainGraph.GetNodeArg(output);
            auto& function_body_graph_output_arg = partitionGraph->GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
            partitionGraphOutputs[i] = &function_body_graph_output_arg;
            ++i;
        }

        partitionGraph->SetInputs(partitionGraphInputs);
        partitionGraph->SetOutputs(partitionGraphOutputs);

        // Add node and node args
        for (auto& node_index : nodes_to_fuse.nodes) 
        {
            auto node = mainGraph.GetNode(node_index);
            std::vector<onnxruntime::NodeArg*> inputs;
            std::vector<onnxruntime::NodeArg*> outputs;
            for (auto input : node->InputDefs()) {
                auto& n_input = partitionGraph->GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
                inputs.push_back(&n_input);
            }
            for (auto output : node->OutputDefs()) {
                auto& n_output = partitionGraph->GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
                outputs.push_back(&n_output);
            }
            partitionGraph->AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        for (const auto& input : meta_def->inputs) 
        {
            const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
            if (mainGraph.GetInitializedTensor(input, initializer)) 
            {
                // meta_def->inputs could have duplicates so make sure we only add once
                const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
                if (!partitionGraph->GetInitializedTensor(input, subgraph_initializer)) 
                {
                    partitionGraph->AddInitializedTensor(*initializer);
                }
            }
        }

        for (const auto& constant_initializer : meta_def->constant_initializers) 
        {
            const ONNX_NAMESPACE::TensorProto* initializer = mainGraph.GetConstantInitializer(constant_initializer, true);
            ORT_ENFORCE(initializer != nullptr, "Initializer " + constant_initializer + " is not found or is not constant initializer.");
            // meta_def->constant_initializers could have duplicates so make sure we only add once
            const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
            if (!partitionGraph->GetInitializedTensor(constant_initializer, subgraph_initializer)) 
            {
                partitionGraph->AddInitializedTensor(*initializer);
            }
        }

        //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
        auto status = partitionGraph->Resolve();
        ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    }

    PartitionGraphDetails CreateDmlGraphForPartition(
        onnxruntime::Graph& graph, 
        const onnxruntime::IndexedSubGraph& nodes_to_fuse/*,
        const std::unordered_map<std::string, GraphNodeProperties>& partitionNodePropsMap,
        std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap,
        std::vector<std::string>& fusedNodeInputArgOriginalNames,
        std::vector<std::string>& fusedNodeOutputArgOriginalNames*/)
    {
        ONNX_NAMESPACE::GraphProto function_storage_proto_;
        std::shared_ptr<onnxruntime::Graph> partitionONNXGraph = std::make_shared<onnxruntime::Graph>(
            graph.GetModel(), 
            graph.GetSchemaRegistry(), 
            function_storage_proto_,
            graph.DomainToVersionMap(),
            graph.GetLogger(),
            graph.StrictShapeTypeInference());

        /*onnxruntime::Graph partitionONNXGraph(
            graph.GetModel(), 
            graph.GetSchemaRegistry(), 
            function_storage_proto_,
            graph.DomainToVersionMap(),
            graph.GetLogger(),
            graph.StrictShapeTypeInference());*/

        CreateONNXGraphForPartition(partitionONNXGraph.get(), graph, nodes_to_fuse);
        PartitionGraphDetails partitionGraphDetails{function_storage_proto_, partitionONNXGraph};
        return partitionGraphDetails;
    }

    void FusePartitionAndRegisterKernel(
        GraphPartition* partition,
        uint32_t partitionIndex,
        onnxruntime::Graph& graph,
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& graphNodePropertyMap,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::string& partitionKernelPrefix,
        std::shared_ptr<std::unordered_map<std::string, onnx::TensorProto>> transferredInitializerMap)
    {
        assert(partition->IsDmlGraphPartition());

        onnxruntime::IndexedSubGraph subGraph;
        // Create a definition for the node.  The name must be unique.
        auto def = std::make_unique<onnxruntime::IndexedSubGraph::MetaDef>();
        def->name = std::string("DmlFusedNode_") + partitionKernelPrefix + std::to_string(partitionIndex);
        def->domain = "DmlFusedNodeDomain";
        def->since_version = 1;
        def->inputs.insert(def->inputs.begin(), partition->GetInputs().begin(), partition->GetInputs().end());
        def->outputs.insert(def->outputs.begin(), partition->GetOutputs().begin(), partition->GetOutputs().end());

        subGraph.SetMetaDef(std::move(def));
        subGraph.nodes = std::move(partition->GetNodeIndices());
        auto& fusedNode = graph.BeginFuseSubGraph(subGraph, subGraph.GetMetaDef()->name);
        fusedNode.SetExecutionProviderType(onnxruntime::kDmlExecutionProvider);
        //graph.FuseSubGraph(subGraph, subGraph.GetMetaDef()->name);
        










        // Create and register partition kernel

        // Populate properties which will be passed to OpKernel for this graph via the function below
        std::unordered_map<std::string, GraphNodeProperties> partitionNodePropsMap;
        for (auto nodeIndex : subGraph.nodes)
        {
            const onnxruntime::Node* node = graph.GetNode(nodeIndex);

#ifdef PRINT_PARTITON_INFO
            printf("Partition %u\t%s\n", partitionIndex, GraphDescBuilder::GetUniqueNodeName(*node).c_str());
#endif
            partitionNodePropsMap.insert(std::make_pair(
                GraphDescBuilder::GetUniqueNodeName(*node), std::move(graphNodePropertyMap[node])));
        }

#ifdef PRINT_PARTITON_INFO
        printf("\n");
#endif

        
        // These nodeArgNames will be used while creating DML Graph inside FusedGraphKernel.cpp
        // Ordering of input/output nodeArgs in below vector will be same as Node::Definitions::input_defs because
        // ORT is populating these args as it is while creating the FusedNode at Graph::CreateFusedSubGraphNode()
        // Why we need these names?
        //      After Partitioning and before reaching to FusedGraphKernel, ORT may modify the input/output nodeArg names
        //      present in FusedNode (Node::Definitions::input_defs) as part of some transformers like memcopy, or L1/L2/L3 transformers.
        std::vector<std::string> fusedNodeInputArgOriginalNames = subGraph.GetMetaDef()->inputs;
        std::vector<std::string> fusedNodeOutputArgOriginalNames = subGraph.GetMetaDef()->outputs;
        /*CreateDmlGraphForPartition(graph, subGraph, 
            partitionNodePropsMap, *transferredInitializerMap, fusedNodeInputArgOriginalNames, fusedNodeOutputArgOriginalNames);*/
        PartitionGraphDetails partitionGraphDetails = CreateDmlGraphForPartition(graph, subGraph);

        auto fused_kernel_func = [partitionNodePropsMap, transferredInitializerMap, partitionGraphDetails,
            fusedNodeInputArgOriginalNames, fusedNodeOutputArgOriginalNames](onnxruntime::FuncManager& func_mgr, 
                const onnxruntime::OpKernelInfo& info, 
                std::unique_ptr<onnxruntime::OpKernel>& out) mutable ->onnxruntime::Status
        {
            out.reset(CreateFusedGraphKernel(
                info, 
                partitionGraphDetails, 
                std::move(partitionNodePropsMap), 
                *transferredInitializerMap, 
                fusedNodeInputArgOriginalNames, 
                fusedNodeOutputArgOriginalNames));
			return Status::OK();
        };

        // build the kernel definition on the fly, and register it to the fused_kernel_regisitry.
        onnxruntime::KernelDefBuilder builder;

        builder.SetName(subGraph.GetMetaDef()->name)
            .SetDomain(subGraph.GetMetaDef()->domain)
            .SinceVersion(subGraph.GetMetaDef()->since_version)
            .Provider(onnxruntime::kDmlExecutionProvider);

        ORT_THROW_IF_ERROR(registryForPartitionKernels->Register(builder, fused_kernel_func));

        
        graph.FinalizeFuseSubGraph(subGraph, fusedNode);

        
        /*return std::make_unique<onnxruntime::ComputeCapability>(std::move(subGraph));*/
    }

    onnxruntime::common::Status DmlGraphFusionTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger) const
    {
        onnxruntime::ProviderType provider_type = onnxruntime::kDmlExecutionProvider;
        const gsl::not_null<const onnxruntime::KernelRegistry*> registry = m_providerImpl->GetKernelRegistry().get();
        const auto kernel_type_str_resolver = onnxruntime::OpSchemaKernelTypeStrResolver{};
        const auto kernel_lookup = onnxruntime::KernelLookup{provider_type,
                                                             gsl::make_span(&registry, 1),
                                                             kernel_type_str_resolver};

        //std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> result;

        // Initializers needed by any graph partition
        std::unordered_set<std::string> requiredInitializerMap;

        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
        onnxruntime::GraphViewer graphViewer(graph);
        std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
            graphViewer,
            *m_providerImpl->GetInternalRegistrationInfoMap(), 
            kernel_lookup,
            m_providerImpl->GetSupportedDeviceDataTypeMask(),
            graphNodePropertyMap, 
            requiredInitializerMap);

        // Create a map between each initialized tensor and the partition(s) it is part of.
        auto initializerPartitionMap = GetInitializerToPartitionMap(graphViewer, partitions);

        for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            if (partition->GetRootMergedPartition() != partition.get() ||
                !partition->IsDmlPartition())
            {
                continue;
            }

            // Create a map which will store by name each initializer which should be transferred to the 
            // partition.  This prevents OnnxRuntime from allocating GPU resources and uploading those initializers,
            // so the partiton's kernel can do so.  In the process, it will pre-process weights while consuming a CPU
            // backed resource, avoiding an extra set of GPU resources in memory.
            // A shared pointer is used so the functor and contained initializer captures can be cheaply copied within ORT.
            auto transferredInitializerMap = std::make_shared<std::unordered_map<std::string, onnx::TensorProto>>();
            
            if (partition->IsDmlGraphPartition())
            {
                // populate transferredInitializerMap
                for (const auto& input : partition->GetInputs())
                {
                    const onnx::TensorProto* tensor = nullptr;
                    if (graph.GetInitializedTensor(input, tensor))
                    {
                        // It's only safe to transfer tensors which are used by this partition alone.
                        auto iter = initializerPartitionMap.find(tensor);
                        assert(iter != initializerPartitionMap.end());
                        if (iter->second.size() > 1)
                        {
                            if (requiredInitializerMap.find(input) != requiredInitializerMap.end())
                            {
                                // The kernel relies on this input to be initialized, and it should be small enough to copy
                                // cheaply. FusedGraphKernel only handles constant CPU inputs through transferred initializers,
                                // rather than ORT, to avoid mismatches in policy or implementation causing failures.
                                (*transferredInitializerMap)[input] = *tensor;
                            }

                            continue;
                        }

                        //// Transfer the initializer
                        //auto& graphTensor = const_cast<onnx::TensorProto&>(*tensor);

                        //onnx::TensorProto partitionTensor;
                        //graphTensor.Swap(&partitionTensor);
                        //(*transferredInitializerMap)[input] = std::move(partitionTensor);
                        (*transferredInitializerMap)[input] = *tensor;

                        graph.RemoveInitializedTensor(input);
                
                        //const_cast<onnxruntime::InitializedTensorSet&>(graph.GetAllInitializedTensors()).erase(graph.GetAllInitializedTensors().find(input));
                    }
                }

                std::string partitionKernelPrefix = std::to_string(m_providerImpl->GetPartitionKernelPrefixVal()) + "_";
                m_providerImpl->IncreasePartitionKernelPrefixVal();

                FusePartitionAndRegisterKernel(
                    partition.get(), 
                    partitionIndex, 
                    graph, 
                    graphNodePropertyMap,
                    m_providerImpl->GetKernelRegistry().get(),
                    partitionKernelPrefix,
                    transferredInitializerMap
                );
            }
        }

        return onnxruntime::common::Status::OK();
    }
}
