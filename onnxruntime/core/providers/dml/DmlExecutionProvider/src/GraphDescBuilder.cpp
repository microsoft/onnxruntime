// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "GraphDescBuilder.h"
#include <stack>

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml::GraphDescBuilder
{

    #pragma warning(push)
    #pragma warning(disable:4702)
    const std::string& GetUniqueNodeName(const onnxruntime::Node& node)
    {
        // The node's name is optional, and it might be re-created with a different index
        // and pointer after partitioning occurs.  Use the name of the node's first valid
        // output as the unique identifier for the node itself.
        for (const auto* arg : node.OutputDefs())
        {
            if (arg->Exists())
            {
                return arg->Name();
            }
        }

        assert(false);
        ORT_THROW_HR(E_UNEXPECTED);
        const onnxruntime::NodeArg* arg = node.OutputDefs()[0];
        return arg->Name();
    }
    #pragma warning(pop)

    static void RemoveUnconnectedNodes(
        std::vector<DmlSerializedGraphNode>& graphNodes,
        std::vector<DmlInputSerializedGraphEdge>& graphInputEdges,
        std::vector<DmlIntermediateSerializedGraphEdge>& graphIntermediateEdges,
        std::vector<DmlOutputSerializedGraphEdge>& graphOutputEdges)
    {
        enum class NodeState
        {
            NotVisited,
            Visiting,
            Visited,
        };

        struct NodeData
        {
            std::vector<uint32_t> predecessorIndices;
            NodeState state = {};
        };

        std::vector<NodeData> nodesData(graphNodes.size());
        for (const DmlIntermediateSerializedGraphEdge& intermediateEdge : graphIntermediateEdges)
        {
            nodesData[intermediateEdge.ToNodeIndex].predecessorIndices.push_back(intermediateEdge.FromNodeIndex);
        }

        std::stack<uint32_t> nodeIndicesToVisit;

        // Start from the outputs of the graph and traverse upwards
        for (const DmlOutputSerializedGraphEdge& outputEdge : graphOutputEdges)
        {
            nodeIndicesToVisit.push(outputEdge.FromNodeIndex);
        }

        while (!nodeIndicesToVisit.empty())
        {
            const uint32_t nodeIndex = nodeIndicesToVisit.top();
            NodeData* node = &nodesData[nodeIndex];

            if (node->state == NodeState::Visited)
            {
                nodeIndicesToVisit.pop();
                continue;
            }

            if (node->state == NodeState::Visiting)
            {
                // The stack has been popped all the way back to this node, which means all its predecessors have been
                // visited. That means we're done visiting this node too.
                node->state = NodeState::Visited;
                nodeIndicesToVisit.pop();
                continue;
            }

            node->state = NodeState::Visiting;

            for (uint32_t predecessorNodeIndex : node->predecessorIndices)
            {
                // If we're already visiting that node, we are in a cycle and we should fail early
                ORT_THROW_HR_IF(E_INVALIDARG, nodesData[predecessorNodeIndex].state == NodeState::Visiting);
                nodeIndicesToVisit.push(predecessorNodeIndex);
            }
        }

        // Delete the edges that reference nodes that are not reachable before removing the nodes themselves
        graphIntermediateEdges.erase(
            std::remove_if(graphIntermediateEdges.begin(), graphIntermediateEdges.end(), [&nodesData](const auto& intermediateEdge){
                return nodesData[intermediateEdge.FromNodeIndex].state == NodeState::NotVisited || nodesData[intermediateEdge.ToNodeIndex].state == NodeState::NotVisited;
            }),
            graphIntermediateEdges.end());

        // Mapping from the old indices to the new indices that have been shifted after removing earlier nodes
        std::vector<uint32_t> shiftedIndicesMapping(graphNodes.size());

        uint32_t shift = 0;
        for (uint32_t nodeIndex = 0; nodeIndex < graphNodes.size(); ++nodeIndex)
        {
            if (nodesData[nodeIndex].state == NodeState::NotVisited)
            {
                // The node is not connected, so we simply increase the shift value (the node will be overwritten by the following nodes)
                ++shift;
            }
            else
            {
                // The node is connected, so we keep it and adjust its mapping
                graphNodes[nodeIndex - shift] = std::move(graphNodes[nodeIndex]);
                shiftedIndicesMapping[nodeIndex] = nodeIndex - shift;
            }
        }

        graphNodes.resize(graphNodes.size() - shift);

        // Adjust the node indices in the input edges
        for (auto& inputEdge : graphInputEdges)
        {
            inputEdge.ToNodeIndex = shiftedIndicesMapping[inputEdge.ToNodeIndex];
        }

        // Adjust the node indices in the output edges
        for (auto& outputEdge : graphOutputEdges)
        {
            outputEdge.FromNodeIndex = shiftedIndicesMapping[outputEdge.FromNodeIndex];
        }

        // Adjust the node indices in the intermediate edges
        for (auto& intermediateEdge : graphIntermediateEdges)
        {
            intermediateEdge.FromNodeIndex = shiftedIndicesMapping[intermediateEdge.FromNodeIndex];
            intermediateEdge.ToNodeIndex = shiftedIndicesMapping[intermediateEdge.ToNodeIndex];
        }
    }

    uint32_t SetAndGetMainDmlGraphNodeIndex(
        const uint32_t operatorDmlGraphNodeIndex,
        const std::string& nodeNamePrefix,
        AbstractOperatorDesc& operatorDesc,
        /*in_out*/std::unordered_map<uint32_t, uint32_t>& operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap,
        /*in_out*/std::vector<DmlSerializedGraphNode>& dmlGraphNodes)
    {
        auto iter = operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap.find(operatorDmlGraphNodeIndex);
        if (iter != operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap.end())
        {
            return iter->second;
        }
        operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap[operatorDmlGraphNodeIndex] = static_cast<uint32_t>(dmlGraphNodes.size());
        dmlGraphNodes.push_back({operatorDesc, nodeNamePrefix + std::to_string(operatorDmlGraphNodeIndex)});
        return operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap[operatorDmlGraphNodeIndex];
    }

    // Terminology:
    //   Subgraph: partitioned ONNX graph from the original (main) ONNX graph
    //   DmlGraph: a graph in DML currency converted from subgraph.
    //   operatorDmlGraph: a graph in DML currency for a given node or operator
    // DmlGraph aka mainDmlGraph to distinguish beetween operatorDmlGraph and DmlGraph.
    // Main Points to note:
    //   - GraphDesc will always has sequential indices for input and intermediate edges.
    GraphDesc BuildDmlGraphDesc(
        const uint8_t* isConstGpuGraphInput,
        const size_t isConstGpuGraphInputCount,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
        const onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubgraph,
        const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
        IDMLDevice* device,
        const ExecutionProviderImpl* providerImpl,
        std::unordered_map<uint32_t, uint32_t>& serializedGraphInputIndexToMainGraphInputIndex,
        std::unordered_map<std::string_view, uint32_t>& serializedGraphConstantNameToMainGraphInputIndex,
        /*out*/std::vector<std::unique_ptr<std::byte[]>>& smallConstantData)
    {
        const gsl::span<const std::string> subgraphInputArgNames = indexedSubgraph.GetMetaDef()->inputs;
        const gsl::span<const std::string> subgraphOutputArgNames = indexedSubgraph.GetMetaDef()->outputs;
        struct NodeAndIndex
        {
            uint32_t nodeIndex; // The index of the node itself
            uint32_t targetIndex; // The index of the input/output on the node (e.g. 1 for the second input on a node)
        };

        // Map from ORT node argument names to input indices of the DML graph (fused kernel node).
        std::unordered_map<std::string_view, uint32_t> subGraphInputNameToInputIndexMap;
        for (size_t inputIndex = 0; inputIndex < subgraphInputArgNames.size(); ++inputIndex)
        {
            const onnxruntime::NodeArg* graphInput = graph.GetNodeArg(subgraphInputArgNames[inputIndex]);
            if (!graphInput)
            {
                // This is a workaround for when node inputs get manipulated by transformers outside of our control,
                // which then causes them to have a different name. If that happens we can't figure out how to
                // correlate inputs to the fused graph index. This likely requires a higher-level fix, but for now
                // just bail early.
                ORT_THROW_HR(E_UNEXPECTED);
            }
            subGraphInputNameToInputIndexMap.emplace(graphInput->Name(), gsl::narrow_cast<uint32_t>(inputIndex));
        }

        auto modelPath = graph.ModelPath();
        auto constantCpuGraphInputGetter = [&isInitializerTransferable, &modelPath](const std::string& argName)
        {
            ComPtr<OnnxTensorWrapper> tensorWrapper;

            auto iter = isInitializerTransferable.find(argName);
            if (iter != isInitializerTransferable.end())
            {
                // Using const_cast here is simpler than making surrounding code const correct.
                tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(const_cast<ONNX_NAMESPACE::TensorProto*>(iter->second.first), modelPath);
            }

            return tensorWrapper;
        };

        // - Map from ORT node's output names to DML graph <NodeAndIndex>.
        // - Once a given ORT node (or operator) will be transformed into a operatorDmlGraph,
        //   then ORT node's output names will become output edges for the operatorDmlGraph.
        // - This map will be populated for those output edges.
        std::unordered_map<std::string, NodeAndIndex> outputEdgeNameToDmlGraphNodeAndIndexMap;

        // This map will be used to re-index an subGraphInputIndex to sequential input index for
        // DmlGraph
        std::unordered_map<uint32_t, uint32_t> subGraphInputIndexToDmlGraphInputIndex;

        std::vector<DmlSerializedGraphNode> dmlGraphNodes;
        std::vector<DmlInputSerializedGraphEdge> dmlGraphInputEdges;
        std::vector<DmlIntermediateSerializedGraphEdge> dmlGraphIntermediateEdges;
        std::vector<DmlOutputSerializedGraphEdge> dmlGraphOutputEdges;
        // Iterate through each node and create a corresponding node in the new graph
        // We can iterate the nodes in any order because the edge connectivity will take care of the topological order
        for (size_t sortedNodeIndex : indexedSubgraph.nodes)
        {
            const onnxruntime::Node& node = *graph.GetNode(sortedNodeIndex);

            const GraphNodeProperties& graphNodeProps = graphNodePropertyMap.find(GetUniqueNodeName(node))->second;
            const auto& requiredConstantCpuInputs = graphNodeProps.internalRegInfo->requiredConstantCpuInputs;

            MLOperatorTensorGetter constantCpuNodeInputGetter = [&node, &constantCpuGraphInputGetter, &requiredConstantCpuInputs](uint32_t inputIndex)
            {
                ComPtr<IMLOperatorTensor> tensor = nullptr;

                auto inputDefs = node.InputDefs();

                if (inputIndex < inputDefs.size())
                {
                    const onnxruntime::NodeArg* arg = inputDefs[inputIndex];
                    tensor = constantCpuGraphInputGetter(arg->Name());

                    if (tensor == nullptr)
                    {
                        bool inputRequiredAsConstant = std::find(
                            requiredConstantCpuInputs.begin(),
                            requiredConstantCpuInputs.end(),
                            inputIndex) != requiredConstantCpuInputs.end();

                        // This shouldn't happen since kernel creation is deferred and repeated when required constant inputs are not present.
                        ORT_THROW_HR_IF(E_UNEXPECTED, inputRequiredAsConstant);
                    }
                }

                return tensor;
            };

            DmlGraphNodeCreateInfo operatorDmlGraphNodeCreateInfo;
            graphNodeProps.internalRegInfo->graphNodeFactoryRegistration->factory(
                node,
                constantCpuNodeInputGetter,
                providerImpl,
                /*out*/ &operatorDmlGraphNodeCreateInfo);

            // Create a map between operatorDmlGraphNodeIndex to mainDmlGraphNodeIndex.
            std::unordered_map<uint32_t, uint32_t> operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap;
            const bool isNodeAsOpDesc = operatorDmlGraphNodeCreateInfo.nodesAsOperatorDesc.size() > 0;

            // Algorithm:
            //  1. Create constant nodes by iterating through operatorDmlGraph's input edges and keep a map of it,
            //     because there would be an intermediate edge from the constantNode and source of the intermediate edge
            //     should come before the destination.
            //  2. Again iterate through operatorDmlGraph's input edges to create mainGraph's input and intermediate edges.
            //  3. Iterate through operatorDmlGraph's intermediate edges to create mainGraph's intermediate edges.
            //  4. Iterate through operatorDmlGraph's output edges to populate outputEdgeNameToDmlGraphNodeAndIndex
            //  5. While performing step 2, 3, and 4, insert operatorDmlGraphNode to the mainDmlGraphNode list.

            for (auto& operatorDmlGraphInputEdge : operatorDmlGraphNodeCreateInfo.inputEdges)
            {
                const onnxruntime::NodeArg* arg = node.InputDefs()[operatorDmlGraphInputEdge.GraphInputIndex];
                if (arg->Exists())
                {
                    auto iter = subGraphInputNameToInputIndexMap.find(arg->Name());
                    if (iter != subGraphInputNameToInputIndexMap.end() &&
                        iter->second < isConstGpuGraphInputCount &&
                        isConstGpuGraphInput[iter->second])
                    {
                        DmlSerializedGraphNode constantNode = {};
                        constantNode.Name = arg->Name();

                        // This is a highly inefficient approach to generating constant nodes.  It duplicates constant data 
                        // across the graph input as well as every consumer's unique constant node.  However it is currently 
                        // only used for small inputs.
                        
                        // TODO: Rework this to create DML constant nodes with the minimum data size actually used by consuming
                        // nodes.  This would allow this size to be reduced while handling the case that 1D scale and zero point
                        // values that have been de-duplicated with conversion to scalars in kernels.
                        ComPtr<OnnxTensorWrapper> constantInput = constantCpuGraphInputGetter(arg->Name());

                        if (constantInput && constantInput->GetTensorByteSize() < c_maxConstNodeDataSize)
                        {
                            smallConstantData.push_back(std::make_unique<std::byte[]>(constantInput->GetTensorByteSize()));
                            std::transform(
                                static_cast<const uint8_t*>(constantInput->GetData()),
                                static_cast<const uint8_t*>(constantInput->GetData()) + constantInput->GetTensorByteSize(),
                                smallConstantData.back().get(),
                                [](uint8_t b) {return static_cast<std::byte>(b);});

                            ConstantData constantData = {smallConstantData.back().get(), constantInput->GetTensorByteSize()};
                            constantNode.Desc = constantData;
                        }
                        else
                        {
                            ConstantName constantFileName = {GetSanitizedFileName(arg->Name())};
                            constantNode.Desc = constantFileName;
                        }
                        outputEdgeNameToDmlGraphNodeAndIndexMap[arg->Name()] = {static_cast<uint32_t>(dmlGraphNodes.size()), 0};
                        dmlGraphNodes.push_back(constantNode);
                    }
                }
            }

            // map operatorDmlGraphInputEdge as either mainDmlGraphInputEdge or mainDmlGraphIntermediateEdge
            for (auto& operatorDmlGraphInputEdge : operatorDmlGraphNodeCreateInfo.inputEdges)
            {
                // operatorDmlGraphInputEdge.GraphInputIndex will be the ONNX input index.
                const onnxruntime::NodeArg* arg = node.InputDefs()[operatorDmlGraphInputEdge.GraphInputIndex];
                if (arg->Exists())
                {
                    auto iter = subGraphInputNameToInputIndexMap.find(arg->Name());
                    uint32_t mainDmlGraphNodeIndex = SetAndGetMainDmlGraphNodeIndex(
                        operatorDmlGraphInputEdge.ToNodeIndex,
                        node.Name(),
                        *operatorDmlGraphNodeCreateInfo.nodesAsOperatorDesc[operatorDmlGraphInputEdge.ToNodeIndex],
                        operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap,
                        dmlGraphNodes);

                    if (iter != subGraphInputNameToInputIndexMap.end())
                    {
                        // If this is a constant input, then it will be an intermediate edge, otherwise
                        // it will be a mainDmlGraphInputEdge.
                        const uint32_t mainDmlGraphInputIndex = iter->second;

                        // If this is a constant input, also set the appropriate flags on the desc
                        if (mainDmlGraphInputIndex < isConstGpuGraphInputCount &&
                            isConstGpuGraphInput[mainDmlGraphInputIndex])
                        {
                            const auto& constantNodeAndIndex = outputEdgeNameToDmlGraphNodeAndIndexMap.at(arg->Name());
                            // if it is large constant tensor then only set the OWNED_BY_DML flag.
                            auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(dmlGraphNodes[constantNodeAndIndex.nodeIndex].Desc);
                            if (std::holds_alternative<ConstantName>(constantNodeVariant))
                            {
                                auto& mainDmlGraphNode = dmlGraphNodes[mainDmlGraphNodeIndex];
                                AbstractOperatorDesc& abstractOperatorDesc = std::get<AbstractOperatorDesc>(mainDmlGraphNode.Desc);
                                std::vector<DmlBufferTensorDesc*> toNodeInputTensorDescs = abstractOperatorDesc.GetInputTensors();
                                DmlBufferTensorDesc* tensorDesc = toNodeInputTensorDescs[operatorDmlGraphInputEdge.ToNodeInputIndex];
                                tensorDesc->flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                            }

                            DmlIntermediateSerializedGraphEdge edge = {};
                            edge.FromNodeIndex = constantNodeAndIndex.nodeIndex;
                            edge.FromNodeOutputIndex = constantNodeAndIndex.targetIndex;
                            edge.ToNodeIndex = mainDmlGraphNodeIndex;
                            edge.ToNodeInputIndex = operatorDmlGraphInputEdge.ToNodeInputIndex;
                            edge.Name = arg->Name() + "-nodeIdx:" + std::to_string(edge.FromNodeIndex) + "-outputIdx:" + std::to_string(edge.FromNodeOutputIndex);
                            serializedGraphConstantNameToMainGraphInputIndex[iter->first] = mainDmlGraphInputIndex;
                            dmlGraphIntermediateEdges.push_back(edge);
                        }
                        else
                        {
                            DmlInputSerializedGraphEdge edge = {};
                            if (subGraphInputIndexToDmlGraphInputIndex.find(mainDmlGraphInputIndex) == subGraphInputIndexToDmlGraphInputIndex.end())
                            {
                                subGraphInputIndexToDmlGraphInputIndex[mainDmlGraphInputIndex] = static_cast<uint32_t>(subGraphInputIndexToDmlGraphInputIndex.size());
                            }

                            edge.GraphInputIndex = subGraphInputIndexToDmlGraphInputIndex[mainDmlGraphInputIndex];
                            edge.ToNodeIndex = mainDmlGraphNodeIndex;
                            edge.ToNodeInputIndex = operatorDmlGraphInputEdge.ToNodeInputIndex;  // ?? might need to point inputIndex
                            edge.Name = arg->Name();

                            serializedGraphInputIndexToMainGraphInputIndex[edge.GraphInputIndex] = mainDmlGraphInputIndex;
                            dmlGraphInputEdges.push_back(edge);
                        }

                    }
                    else
                    {
                        const auto& inputNodeAndIndex = outputEdgeNameToDmlGraphNodeAndIndexMap.at(arg->Name());

                        DmlIntermediateSerializedGraphEdge edge = {};
                        edge.FromNodeIndex = inputNodeAndIndex.nodeIndex;
                        edge.FromNodeOutputIndex = inputNodeAndIndex.targetIndex;
                        edge.ToNodeIndex = mainDmlGraphNodeIndex;
                        edge.ToNodeInputIndex = operatorDmlGraphInputEdge.ToNodeInputIndex;
                                                edge.Name = arg->Name();
                        dmlGraphIntermediateEdges.push_back(edge);
                    }
                }
            }

            // map operatorGraphIntermediateEdges as mainGraphIntermediateEdge
            for (auto& operatorGraphIntermediateEdge : operatorDmlGraphNodeCreateInfo.intermediateEdges)
            {
                DmlIntermediateSerializedGraphEdge edge = {};
                uint32_t shiftedFromNodeIndex = SetAndGetMainDmlGraphNodeIndex(
                        operatorGraphIntermediateEdge.FromNodeIndex,
                        node.Name(),
                        *operatorDmlGraphNodeCreateInfo.nodesAsOperatorDesc[operatorGraphIntermediateEdge.FromNodeIndex],
                        operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap,
                        dmlGraphNodes);
                uint32_t shiftedToNodeIndex = SetAndGetMainDmlGraphNodeIndex(
                        operatorGraphIntermediateEdge.ToNodeIndex,
                        node.Name(),
                        *operatorDmlGraphNodeCreateInfo.nodesAsOperatorDesc[operatorGraphIntermediateEdge.ToNodeIndex],
                        operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap,
                        dmlGraphNodes);

                edge.FromNodeIndex = shiftedFromNodeIndex;
                edge.FromNodeOutputIndex = operatorGraphIntermediateEdge.FromNodeOutputIndex;
                edge.ToNodeIndex = shiftedToNodeIndex;
                edge.ToNodeInputIndex = operatorGraphIntermediateEdge.ToNodeInputIndex;
                edge.Name = "nodeIdx:" + std::to_string(shiftedFromNodeIndex) + "-outputIdx:" + std::to_string(operatorGraphIntermediateEdge.FromNodeOutputIndex);
                dmlGraphIntermediateEdges.push_back(edge);
            }

            // populate nameToNodeAndIndexMap (which will be used by above loop) for operatorGraphOutputEdges
            for (auto& operatorGraphOutputEdge : operatorDmlGraphNodeCreateInfo.outputEdges)
            {
                const onnxruntime::NodeArg* arg = node.OutputDefs()[operatorGraphOutputEdge.GraphOutputIndex];
                if (arg->Exists())
                {
                    uint32_t shiftedNodeIndex = SetAndGetMainDmlGraphNodeIndex(
                            operatorGraphOutputEdge.FromNodeIndex,
                            node.Name(),
                            *operatorDmlGraphNodeCreateInfo.nodesAsOperatorDesc[operatorGraphOutputEdge.FromNodeIndex],
                            operatorDmlGraphNodeIndexToMainDmlGraphNodeIndexMap,
                            dmlGraphNodes);
                    outputEdgeNameToDmlGraphNodeAndIndexMap[arg->Name()] = {shiftedNodeIndex, operatorGraphOutputEdge.FromNodeOutputIndex};
                }
            }
        }

        // Add graph output nodes, which might be in a different order from the encapsulating node
        for (size_t outputIndex = 0; outputIndex < subgraphOutputArgNames.size(); ++outputIndex)
        {
            const onnxruntime::NodeArg* graphOutput = graph.GetNodeArg(subgraphOutputArgNames[outputIndex]);

            ORT_THROW_HR_IF_NULL_MSG(E_POINTER, graphOutput, "FusedNode's nodeArgList does not contain one of the nodeArg");
            const auto& outputNodeAndIndex = outputEdgeNameToDmlGraphNodeAndIndexMap.at(graphOutput->Name());

            DmlOutputSerializedGraphEdge edge = {};
            edge.FromNodeIndex = outputNodeAndIndex.nodeIndex;
            edge.FromNodeOutputIndex = outputNodeAndIndex.targetIndex;
            edge.GraphOutputIndex = gsl::narrow_cast<uint32_t>(outputIndex);
            edge.Name = graphOutput->Name();
            dmlGraphOutputEdges.push_back(edge);
        }

        RemoveUnconnectedNodes(dmlGraphNodes, dmlGraphInputEdges, dmlGraphIntermediateEdges, dmlGraphOutputEdges);

        GraphDesc graphDesc{};
        graphDesc.InputCount = static_cast<uint32_t>(dmlGraphInputEdges.size());
        graphDesc.OutputCount = static_cast<uint32_t>(subgraphOutputArgNames.size());
        graphDesc.Nodes = std::move(dmlGraphNodes);
        graphDesc.InputEdges = std::move(dmlGraphInputEdges);
        graphDesc.OutputEdges = std::move(dmlGraphOutputEdges);
        graphDesc.IntermediateEdges = std::move(dmlGraphIntermediateEdges);
        // Avoid using separate command lists for small graphs. This value can be reduced by tuning the
        // flushing behavior of DmlCommandRecorder.  Its current behavior is to assume that graphs contain
        // enough GPU work to be worth flushing immediately.
        graphDesc.reuseCommandList = ((indexedSubgraph.nodes.size() >= minNodeCountToReuseCommandList) || providerImpl->IsMcdmDevice());
        return graphDesc;
    }
}
