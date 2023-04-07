// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "IExecutionProvider.h"
#include "ExecutionProvider.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "FusedGraphKernel.h"
#include "GraphDescBuilder.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/framework/compute_capability.h"
#include <wil/wrl.h>
#ifndef _GAMING_XBOX
#include <dxgi1_6.h>
#endif
#include "GraphPartitioner.h"

//#define PRINT_PARTITON_INFO

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml
{
    GraphPartition* GraphPartition::GetRootMergedPartition()
    {
        return m_mergedPartition ? m_mergedPartition->GetRootMergedPartition() : this;
    }

    std::vector<onnxruntime::NodeIndex>& GraphPartition::GetNodeIndices()
    {
        assert(this == GetRootMergedPartition());
        return m_nodeIndices;
    }

    std::set<std::string>& GraphPartition::GetInputs()
    {
        assert(this == GetRootMergedPartition());
        return m_inputs;
    }

    std::set<std::string>& GraphPartition::GetOutputs()
    {
        assert(this == GetRootMergedPartition());
        return m_outputs;
    }

    bool GraphPartition::IsFinalized()
    {
        assert(this == GetRootMergedPartition());
        return m_finalized;
    }

    void GraphPartition::SetFinalized()
    {
        m_finalized = true;
    }

    bool GraphPartition::IsDmlPartition()
    {
        assert(this == GetRootMergedPartition());
        return m_isDmlPartition;
    }

    bool GraphPartition::IsDmlGraphPartition()
    {
        assert(this == GetRootMergedPartition());
        return m_isDmlGraphPartition;
    }

    void GraphPartition::SetIsDmlPartition(bool isDmlPartition)
    {
        assert(this == GetRootMergedPartition());
        m_isDmlPartition = isDmlPartition;
    }

    void GraphPartition::SetIsDmlGraphPartition(bool isDmlGraphPartition)
    {
        assert(this == GetRootMergedPartition());
        m_isDmlGraphPartition = isDmlGraphPartition;
    }

    void GraphPartition::AddNodeIndex(onnxruntime::NodeIndex index)
    {
        assert(!IsFinalized());
        assert(std::find(m_nodeIndices.begin(), m_nodeIndices.end(), index) == m_nodeIndices.end());

        m_nodeIndices.push_back(index);
    }

    void GraphPartition::AddInput(const std::string& name)
    {
        assert(!IsFinalized());
        assert(this == GetRootMergedPartition());
        m_inputs.insert(name);
    }

    void GraphPartition::AddOutput(const std::string& name)
    {
        assert(this == GetRootMergedPartition());
        m_outputs.insert(name);
    }

    void GraphPartition::Merge(gsl::span<GraphPartition*> partitionsToMerge)
    {
        assert(this == GetRootMergedPartition());

        for (GraphPartition* partitionToMerge : partitionsToMerge)
        {
            if (partitionToMerge->GetRootMergedPartition() == this)
            {
                continue;
            }

            assert(!partitionToMerge->IsFinalized());
            assert(partitionToMerge->IsDmlPartition() == IsDmlPartition());
            assert(partitionToMerge->IsDmlGraphPartition() == IsDmlGraphPartition());

            partitionToMerge->m_mergedPartition = this;

            m_nodeIndices.insert(m_nodeIndices.begin(), partitionToMerge->m_nodeIndices.begin(), partitionToMerge->m_nodeIndices.end());
            m_inputs.insert(partitionToMerge->m_inputs.begin(), partitionToMerge->m_inputs.end());
            m_outputs.insert(partitionToMerge->m_outputs.begin(), partitionToMerge->m_outputs.end());
        }
    }

    // Adds the outputs of a node to the specified partition
    void AddNodeOutputsToPartitionMap(
        const onnxruntime::Node& node,
        GraphPartition* partition,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        for (uint32_t i = 0; i < node.OutputDefs().size(); ++i)
        {
            const auto* arg = node.OutputDefs()[i];
            if (arg->Exists())
            {
                nodeNameToPartitionMap[arg->Name()] = partition;
            }
        }
    };

    // Gets properties of the registration for a node
    void GetRegistrationProperties(
        const onnxruntime::GraphViewer& graph,
        const onnxruntime::Node& node,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        const InternalRegistrationInfoMap& internalRegInfoMap,
        _In_opt_ const std::unordered_map<std::string, GraphPartition*>* nodeNameToPartitionMap,
        _Inout_ std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& dmlNodePropertyMap,
        _Inout_ std::unordered_set<std::string>& requiredInitializerMap,
        _Out_ bool* isDmlGraphNode
        )
    {
        *isDmlGraphNode = false;

        // Get the kernel creation info for the registration, and check if it carries the property
        // set during registration of kernels that support DML graph node usage.
        auto graphNodeProperty = dmlNodePropertyMap.insert(std::make_pair(&node, GraphNodeProperties()));

        // Ensure that shape information is known statically for the inputs and outputs of the node,
        // which is required for MLGraph compilation.
        const onnxruntime::KernelCreateInfo* createInfo = kernel_lookup.LookUpKernel(node);
        assert(createInfo != nullptr);  // since GetRegistrationProperties is called only when node is a DML node

        auto regInfoIter = internalRegInfoMap.find(createInfo->kernel_def.get());
        if (regInfoIter != internalRegInfoMap.end())
        {
            auto internalRegInfo = regInfoIter->second;

            if (internalRegInfo && internalRegInfo->graphNodeFactoryRegistration)
            {
                bool requiredCpuInputsConstant = true;
                for (uint32_t inputIndex : internalRegInfo->requiredConstantCpuInputs)
                {
                    if (inputIndex >= node.InputDefs().size() || !node.InputDefs()[inputIndex]->Exists())
                    {
                        continue;
                    }

                    const onnx::TensorProto* tensor = nullptr;
                    const std::string& inputName = node.InputDefs()[inputIndex]->Name();

                    if (!graph.GetInitializedTensor(inputName, tensor))
                    {
                        requiredCpuInputsConstant = false;
                        break;
                    }

                    requiredInitializerMap.insert(inputName);
                }

                std::optional<uint32_t> requiredInputCount = internalRegInfo->graphNodeFactoryRegistration->requiredInputCount;
                if (requiredCpuInputsConstant &&
                    TryGetStaticInputShapes( node, graphNodeProperty.first->second.inputShapes) &&
                    !ContainsEmptyDimensions(graphNodeProperty.first->second.inputShapes, internalRegInfo->requiredConstantCpuInputs) &&
                    TryGetStaticOutputShapes(node, graphNodeProperty.first->second.outputShapes) &&
                    !ContainsEmptyDimensions(graphNodeProperty.first->second.outputShapes, internalRegInfo->requiredConstantCpuInputs) &&
                    (requiredInputCount == std::nullopt || *requiredInputCount == node.InputDefs().size()))
                {
                    *isDmlGraphNode = true;
                    graphNodeProperty.first->second.internalRegInfo = internalRegInfo;
                }
            }
        }
    }

    // Creates a partition for a node which is not a DML graph node, and finalizes partitions
    // which are inputs of the new partition.
    std::unique_ptr<GraphPartition> CreateNonGraphNodePartitionAndFinalizeInputs(
        const onnxruntime::Node& node,
        bool isDmlNode,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::unique_ptr<GraphPartition> partition = std::make_unique<GraphPartition>();
        partition->SetIsDmlGraphPartition(false);
        partition->SetIsDmlPartition(isDmlNode);
        partition->AddNodeIndex(node.Index());

        for (uint32_t i = 0; i < node.InputDefs().size(); ++i)
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                const std::string& argName = arg->Name();

                if (nodeNameToPartitionMap.find(argName) != nodeNameToPartitionMap.end())
                {
                    // Finalize the partition which contains an input to a non-DML-graph partition.
                    // The connections from that partition to other partitions, such as this one,
                    // must become outputs of that partition.  As subsequent downstream nodes of
                    // the finalized partition are visited, other outputs will subsequently be
                    // added to the partition, too.
                    GraphPartition* inputPartition = nodeNameToPartitionMap[argName]->GetRootMergedPartition();
                    inputPartition->SetFinalized();
                    inputPartition->AddOutput(argName);
                }

                partition->AddInput(argName);
            }
        }

        partition->SetFinalized();
        AddNodeOutputsToPartitionMap(node, partition.get(), nodeNameToPartitionMap);

        return partition;
    }

    // Get the partitions which are inputs to the specified node and which are not finalized.
    std::vector<GraphPartition*> GetNonFinalizedInputPartitions(
        const onnxruntime::Node& node,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::vector<GraphPartition*> inputNonFinalPartitions;

        for (uint32_t i = 0; i < node.InputDefs().size(); ++i)
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                const std::string& argName = arg->Name();

                if (nodeNameToPartitionMap.find(argName) == nodeNameToPartitionMap.end())
                {
                    // Must be source node
                    continue;
                }

                GraphPartition* inputPartition = nodeNameToPartitionMap[argName]->GetRootMergedPartition();

                if (!inputPartition->IsFinalized())
                {
                    inputNonFinalPartitions.push_back(inputPartition);
                }
            }
        }

        return inputNonFinalPartitions;
    }

    // Add graph outputs of the new node to a partition.
    void AddGraphOutputsFromNodeToPartition(
        const onnxruntime::Node& node,
        const std::set<std::string>& graphOutputs,
        GraphPartition* partition
    )
    {
        for (uint32_t i = 0; i < node.OutputDefs().size(); ++i)
        {
            const auto* arg = node.OutputDefs()[i];
            if (arg->Exists())
            {
                if (graphOutputs.find(arg->Name()) != graphOutputs.end())
                {
                    partition->AddOutput(arg->Name());
                }
            }
        }
    }

    std::unique_ptr<GraphPartition> CreateNewPartitionWithFinalizedInputPartitions(
        const onnxruntime::Node& node,
        const std::set<std::string>& graphOutputs,
        std::unordered_map<std::string, GraphPartition*>& nodeNameToPartitionMap
    )
    {
        std::unique_ptr<GraphPartition> partition = std::make_unique<GraphPartition>();
        partition->SetIsDmlGraphPartition(true);
        partition->SetIsDmlPartition(true);
        partition->AddNodeIndex(node.Index());

        // Inputs of the partition are added when partitions are created and extended when
        // nodes are added with inputs which are not inside the partition
        for (uint32_t i = 0; i < node.InputDefs().size(); ++i)
        {
            const auto* arg = node.InputDefs()[i];
            if (arg->Exists())
            {
                partition->AddInput(arg->Name());

                auto inputPartition = nodeNameToPartitionMap.find(arg->Name());
                if (inputPartition != nodeNameToPartitionMap.end())
                {
                    inputPartition->second->GetRootMergedPartition()->AddOutput(arg->Name());
                }
            }
        }

        // Outputs of the partition are initially set to node outputs which are also
        // graph outputs.  They are extended when adding other node with the graph
        // outputs from those nodes.  They are also extended when a partition
        // consumes an input from the current partition.
        AddGraphOutputsFromNodeToPartition(node, graphOutputs, partition.get());

        AddNodeOutputsToPartitionMap(node, partition.get(), nodeNameToPartitionMap);

        return partition;
    }

    // Whether any operator in the model contains a subgraph.  This is true
    // if the graph being partitioned is itself within a subgraph, or contains
    // an operator with a subgraph.
    bool ModelUsesSubgraph(const onnxruntime::GraphViewer& graph)
    {
        if (graph.IsSubgraph())
        {
            return true;
        }

        const std::vector<onnxruntime::NodeIndex>& toplogicalOrder = graph.GetNodesInTopologicalOrder();

        for (size_t nodeIndex : toplogicalOrder)
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);
            if (node.ContainsSubgraph())
            {
                return true;
            }
        }

        return false;
    }

    //
    // A simple graph partitioning algorithm is used:
    //
    // - If a node has any input which is already in a graph, and that graph is not finalized,
    //   then the node and all such input graphs are merged.
    //
    // - Once a node has an output which cannot be merged with its graph, its graph is marked
    //   as final, which disallows its future extensions.  This ensures that no indirect
    //   downstream dependencies of the external output node are later merged.
    //
    std::vector<std::unique_ptr<GraphPartition>>
    BuildPartitions(
        const onnxruntime::GraphViewer& graph,
        const InternalRegistrationInfoMap& internalRegInfoMap,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& graphNodePropertyMap,
        std::unordered_set<std::string>& requiredInitializerMap,
        std::function<void(const onnxruntime::Node&)> onNodeUnsupportedInGraph)
    {
        // Nodes are uniquely identified by the name of their first output argument
        std::vector<std::unique_ptr<GraphPartition>> partitions;
        std::unordered_map<std::string, GraphPartition*> nodeNameToPartitionMap;

        // Get the list of node indices in toplogical order, so nodes are visited before.
        // downstream nodes consuming them.
        const std::vector<onnxruntime::NodeIndex>& toplogicalOrder = graph.GetNodesInTopologicalOrder();

        // Construct sets with graph inputs and outputs for fast lookup later.
        std::set<std::string> graphInputs;
        std::set<std::string> graphOutputs;

        for (const auto* arg : graph.GetInputsIncludingInitializers())
        {
            graphInputs.insert(arg->Name());
        }

        // If a model contains an intializer which is not also a graph input, it will not be returned
        // by GetInputsIncludingInitializers above.  Such models would be invalid, however they loaded
        // in RS5.  For compatibility, this ensures that such models continue to load.  This is
        // verified by an ONNX conformance test for Add.
        for (const auto& arg : graph.GetAllInitializedTensors())
        {
            // This adds the initializer to the input set if it didn't already exist.
            graphInputs.insert(arg.first);
        }

        for (const auto* arg : graph.GetOutputs())
        {
            graphOutputs.insert(arg->Name());
        }

        // Check whether this graph is a subgraph, or contains any node with a subgraph.
        bool modelUsesSubgraph = ModelUsesSubgraph(graph);

        // Build up partitions while traversing the graph.
        for (size_t nodeIndex : toplogicalOrder)
        {
            const onnxruntime::Node& node = *graph.GetNode(nodeIndex);

            // Whether the node is implemented through DML.
            bool isDmlNode = node.GetExecutionProviderType() == onnxruntime::kDmlExecutionProvider;

            // Whether the node is implemented through DML and as a graph node, meaning it
            // can generate DML operations through a private interface for use as an MLGraph node.
            bool isDmlGraphNode = false;

            // Get the registration properties above and populate nodeNameToPartitionMap.
            if (isDmlNode)
            {
                GetRegistrationProperties(
                    graph,
                    node,
                    kernel_lookup,
                    supportedDeviceDataTypeMask,
                    internalRegInfoMap,
                    &nodeNameToPartitionMap,
                    graphNodePropertyMap,
                    requiredInitializerMap,
                    /*out*/ &isDmlGraphNode
                );
            }

            // Add a unique partition if graph node usage is not supported.
            //
            // Partitioning is disabled in models with subgraphs to work around issues with implicit inputs.
            // The partitioning algorithm does not currently consider such inputs.  Transfering shared initializers
            // for partitions could also cause problems.  Note, operators with subgraphs are currently not efficient
            // anyhow due to CPU/GPU copies.
            if (modelUsesSubgraph || !isDmlGraphNode)
            {
                if (onNodeUnsupportedInGraph)
                {
                    onNodeUnsupportedInGraph(node);
                }

                partitions.push_back(CreateNonGraphNodePartitionAndFinalizeInputs(node, isDmlNode, nodeNameToPartitionMap));
                continue;
            }

            std::vector<GraphPartition*> inputNonFinalPartitions = GetNonFinalizedInputPartitions(node, nodeNameToPartitionMap);

            if (inputNonFinalPartitions.empty())
            {
                partitions.push_back(CreateNewPartitionWithFinalizedInputPartitions(node, graphOutputs, nodeNameToPartitionMap));
            }
            else
            {
                // Arbitrarily pick the first non-final partition found among the inputs, and add this node
                // and its output arguments to that partition.
                GraphPartition* firstNonFinalInputPartition = inputNonFinalPartitions[0]->GetRootMergedPartition();
                firstNonFinalInputPartition->AddNodeIndex(node.Index());
                AddNodeOutputsToPartitionMap(node, firstNonFinalInputPartition, nodeNameToPartitionMap);

                // Add inputs for the new node which span partitions
                for (uint32_t i = 0; i < node.InputDefs().size(); ++i)
                {
                    const auto* arg = node.InputDefs()[i];
                    if (arg->Exists())
                    {
                        auto inputPartition = nodeNameToPartitionMap.find(arg->Name());

                        // Add the input of the current node into the partition which the node will be merged into.
                        // Skip this if the input is already merged into the same partition or is not finalized,
                        // and so will be subsequently merged below.
                        if (inputPartition != nodeNameToPartitionMap.end() &&
                            inputPartition->second->GetRootMergedPartition() != firstNonFinalInputPartition &&
                            inputPartition->second->GetRootMergedPartition()->IsFinalized())
                        {
                            // Add this input of the current node as an output of the final partition to which
                            // it belongs.
                            inputPartition->second->GetRootMergedPartition()->AddOutput(arg->Name());
                            firstNonFinalInputPartition->AddInput(arg->Name());
                        }

                        if (graphInputs.find(arg->Name()) != graphInputs.end())
                        {
                            firstNonFinalInputPartition->AddInput(arg->Name());
                        }
                    }
                }

                // Add graph outputs of the new node
                AddGraphOutputsFromNodeToPartition(node, graphOutputs, firstNonFinalInputPartition);

                // Merge each other non-finalized input partition into the first one
                if (inputNonFinalPartitions.size() > 1)
                {
                    firstNonFinalInputPartition->Merge(gsl::span<GraphPartition*>(&inputNonFinalPartitions[1], inputNonFinalPartitions.size() - 1));
                }
            }
        }

        return partitions;
    }
} // namespace Dml
