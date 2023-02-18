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
        std::vector<NodeInfo>& graphNodes,
        std::vector<DML_INPUT_GRAPH_EDGE_DESC>& graphInputEdges,
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC>& graphIntermediateEdges,
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC>& graphOutputEdges)
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
        for (const DML_INTERMEDIATE_GRAPH_EDGE_DESC& intermediateEdge : graphIntermediateEdges)
        {
            nodesData[intermediateEdge.ToNodeIndex].predecessorIndices.push_back(intermediateEdge.FromNodeIndex);
        }

        std::stack<uint32_t> nodeIndicesToVisit;

        // Start from the outputs of the graph and traverse upwards
        for (const DML_OUTPUT_GRAPH_EDGE_DESC& outputEdge : graphOutputEdges)
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

    GraphDesc BuildGraphDesc(
        const uint8_t* isConstGpuGraphInput,
        const size_t isConstGpuGraphInputCount,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
        const onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
        IDMLDevice* device,
        const void* executionHandle)
    {
        const gsl::span<const std::string> subGraphInputArgNames = indexedSubGraph.GetMetaDef()->inputs;
        const gsl::span<const std::string> subGraphOutputArgNames = indexedSubGraph.GetMetaDef()->outputs;
        struct NodeAndIndex
        {
            uint32_t nodeIndex; // The index of the node itself
            uint32_t targetIndex; // The index of the input/output on the node (e.g. 1 for the second input on a node)
        };

        // Map from Lotus node argument names to the new node and index where it will be produced
        std::unordered_map<std::string, NodeAndIndex> nameToNodeAndIndexMap;

        // Map from Lotus node argument names to input indices of the fused kernel node.
        std::unordered_map<std::string, uint32_t> nameToDmlFusedNodeInputIndex;

        for (size_t inputIndex = 0; inputIndex < subGraphInputArgNames.size(); ++inputIndex)
        {
            const onnxruntime::NodeArg* graphInput = graph.GetNodeArg(subGraphInputArgNames[inputIndex]);

            if (!graphInput)
            {
                // This is a workaround for when node inputs get manipulated by transformers outside of our control,
                // which then causes them to have a different name. If that happens we can't figure out how to
                // correlate inputs to the fused graph index. This likely requires a higher-level fix, but for now
                // just bail early.
                ORT_THROW_HR(E_UNEXPECTED);
            }

            nameToDmlFusedNodeInputIndex.emplace(graphInput->Name(), gsl::narrow_cast<uint32_t>(inputIndex));
        }

        StackAllocator<1024> allocator; // Used for converting abstract operator descs into DML_OPERATOR_DESC

        std::vector<NodeInfo> graphNodes;
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> graphInputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> graphIntermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> graphOutputEdges;

        // Avoid using separate command lists for small graphs. This value can be reduced by tuning the
        // flushing behavior of DmlCommandRecorder.  Its current behavior is to assume that graphs contain
        // enough GPU work to be worth flushing immediately.
        const uint32_t minNodeCountToReuseCommandList = 5;
        bool reuseCommandList = false;

        if (indexedSubGraph.nodes.size() >= minNodeCountToReuseCommandList)
        {
            reuseCommandList = true;
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

        // Iterate through each node and create a corresponding node in the new graph
        // We can iterate the nodes in any order because the edge connectivity will take care of the topological order
        for (size_t sortedNodeIndex : indexedSubGraph.nodes)
        {
            const onnxruntime::Node& node = *graph.GetNode(sortedNodeIndex);

            const GraphNodeProperties& graphNodeProps = graphNodePropertyMap.find(GetUniqueNodeName(node))->second;
            const auto& requiredConstantCpuInputs = graphNodeProps.internalRegInfo->requiredConstantCpuInputs;

            MLOperatorTensorGetter constantCpuNodeInputGetter = [&node, &constantCpuGraphInputGetter, &requiredConstantCpuInputs](uint32_t inputIndex)
            {
                ComPtr<IMLOperatorTensor> tensor = nullptr;

                // Check whether this specific node requested support for constant CPU inputs
                if (std::find(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end(), inputIndex) != requiredConstantCpuInputs.end())
                {
                    auto inputDefs = node.InputDefs();
                    if (inputIndex < inputDefs.size())
                    {
                        const onnxruntime::NodeArg* arg = inputDefs[inputIndex];
                        tensor = constantCpuGraphInputGetter(arg->Name());
                    }
                }

                return tensor;
            };

            DmlGraphNodeCreateInfo graphNodeCreateInfo;
            graphNodeProps.internalRegInfo->graphNodeFactoryRegistration->factory(
                node,
                constantCpuNodeInputGetter,
                executionHandle,
                /*out*/ &graphNodeCreateInfo
            );

            // Create a map between operatorGraphNodeIndex to mainGraphNodeIndex.
            std::unordered_map<uint32_t, uint32_t> operatorGraphNodeIndexToMainGraphNodeIndexMap;
            uint32_t graphNodeCount = gsl::narrow_cast<uint32_t>(graphNodes.size());
            const bool isNodeAsOpDesc = graphNodeCreateInfo.nodesAsOperatorDesc.size() > 0;

            if (isNodeAsOpDesc)
            {
                // Can't populate graphNodes vector at this point, because operatorDesc may get modified later.
                for (uint32_t nodeIndex = 0; nodeIndex < graphNodeCreateInfo.nodeCount; nodeIndex++)
                {
                    ORT_THROW_HR_IF(E_UNEXPECTED, !graphNodeCreateInfo.nodesAsOperatorDesc[nodeIndex]);
                    operatorGraphNodeIndexToMainGraphNodeIndexMap.emplace(nodeIndex, graphNodeCount++);
                }
            }
            else
            {
                for (uint32_t nodeIndex = 0; nodeIndex < graphNodeCreateInfo.nodeCount; nodeIndex++)
                {
                    ORT_THROW_HR_IF(E_UNEXPECTED, !graphNodeCreateInfo.nodesAsIDMLOperator[nodeIndex].Get());
                    operatorGraphNodeIndexToMainGraphNodeIndexMap.emplace(nodeIndex, graphNodeCount++);
                    NodeInfo nodeInfo = {};
                    nodeInfo.op = std::move(graphNodeCreateInfo.nodesAsIDMLOperator[nodeIndex]);
                    graphNodes.push_back(std::move(nodeInfo));
                }
            }

            // map operatorGraphInputEdge as either mainGraphInputEdge or mainGraphIntermediateEdge
            for (auto& operatorGraphInputEdge : graphNodeCreateInfo.inputEdges)
            {
                // operatorGraphInputEdge.GraphInputIndex will be the ONNX input index.
                const onnxruntime::NodeArg* arg = node.InputDefs()[operatorGraphInputEdge.GraphInputIndex];

                if (arg->Exists())
                {
                    auto iter = nameToDmlFusedNodeInputIndex.find(arg->Name());
                    uint32_t mainGraphNodeIndex = operatorGraphNodeIndexToMainGraphNodeIndexMap[operatorGraphInputEdge.ToNodeIndex];

                    if (iter != nameToDmlFusedNodeInputIndex.end())
                    {
                        // This is a graph input

                        const uint32_t dmlFusedNodeInputIndex = iter->second;

                        DML_INPUT_GRAPH_EDGE_DESC edge = {};
                        edge.GraphInputIndex = dmlFusedNodeInputIndex;
                        edge.ToNodeIndex = mainGraphNodeIndex;
                        edge.ToNodeInputIndex = operatorGraphInputEdge.ToNodeInputIndex;  // ?? might need to point inputIndex
                        graphInputEdges.push_back(edge);

                        // If this is a constant input, set the appropriate flags on the desc
                        if (isNodeAsOpDesc &&
                            dmlFusedNodeInputIndex < isConstGpuGraphInputCount &&
                            isConstGpuGraphInput[dmlFusedNodeInputIndex])
                        {
                            auto& graphInputNode = graphNodeCreateInfo.nodesAsOperatorDesc[operatorGraphInputEdge.ToNodeIndex];
                            std::vector<DmlBufferTensorDesc*> toNodeInputTensorDescs = graphInputNode->GetInputTensors();
                            DmlBufferTensorDesc* tensorDesc = toNodeInputTensorDescs[operatorGraphInputEdge.ToNodeInputIndex];
                            tensorDesc->flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                        }
                    }
                    else
                    {
                        const auto& inputNodeAndIndex = nameToNodeAndIndexMap.at(arg->Name());

                        DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
                        edge.FromNodeIndex = inputNodeAndIndex.nodeIndex;
                        edge.FromNodeOutputIndex = inputNodeAndIndex.targetIndex;
                        edge.ToNodeIndex = mainGraphNodeIndex;
                        edge.ToNodeInputIndex = operatorGraphInputEdge.ToNodeInputIndex;
                        graphIntermediateEdges.push_back(edge);
                    }
                }
            }

            // map operatorGraphIntermediateEdges as mainGraphIntermediateEdge
            for (auto& operatorGraphIntermediateEdge : graphNodeCreateInfo.intermediateEdges)
            {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
                edge.FromNodeIndex = operatorGraphNodeIndexToMainGraphNodeIndexMap[operatorGraphIntermediateEdge.FromNodeIndex];
                edge.FromNodeOutputIndex = operatorGraphIntermediateEdge.FromNodeOutputIndex;
                edge.ToNodeIndex = operatorGraphNodeIndexToMainGraphNodeIndexMap[operatorGraphIntermediateEdge.ToNodeIndex];
                edge.ToNodeInputIndex = operatorGraphIntermediateEdge.ToNodeInputIndex;
                graphIntermediateEdges.push_back(edge);
            }

            // populate nameToNodeAndIndexMap (which will be used by above loop) for operatorGraphOutputEdges
            for (auto& operatorGraphOutputEdge : graphNodeCreateInfo.outputEdges)
            {
                const onnxruntime::NodeArg* arg = node.OutputDefs()[operatorGraphOutputEdge.GraphOutputIndex];
                if (arg->Exists())
                {
                    nameToNodeAndIndexMap[arg->Name()] = NodeAndIndex {
                        operatorGraphNodeIndexToMainGraphNodeIndexMap[operatorGraphOutputEdge.FromNodeIndex],
                        operatorGraphOutputEdge.FromNodeOutputIndex
                    };
                }
            }

            if (isNodeAsOpDesc)
            {
                for (auto& opDesc : graphNodeCreateInfo.nodesAsOperatorDesc)
                {
                    DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc(*opDesc, &allocator);
                    ComPtr<IDMLOperator> op;
                    ORT_THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
                    allocator.Reset();

                    NodeInfo nodeInfo = {};
                    nodeInfo.op = std::move(op);
                    nodeInfo.name = node.Name();
                    graphNodes.push_back(std::move(nodeInfo));
                }
            }
        }

        // Add graph output nodes, which might be in a different order from the encapsulating node
        for (size_t outputIndex = 0; outputIndex < subGraphOutputArgNames.size(); ++outputIndex)
        {
            const onnxruntime::NodeArg* graphOutput = graph.GetNodeArg(subGraphOutputArgNames[outputIndex]);

            ORT_THROW_HR_IF_NULL_MSG(E_POINTER, graphOutput, "FusedNode's nodeArgList does not contain one of the nodeArg");
            const auto& outputNodeAndIndex = nameToNodeAndIndexMap.at(graphOutput->Name());

            DML_OUTPUT_GRAPH_EDGE_DESC edge = {};
            edge.FromNodeIndex = outputNodeAndIndex.nodeIndex;
            edge.FromNodeOutputIndex = outputNodeAndIndex.targetIndex;
            edge.GraphOutputIndex = gsl::narrow_cast<uint32_t>(outputIndex);
            graphOutputEdges.push_back(edge);
        }

        RemoveUnconnectedNodes(graphNodes, graphInputEdges, graphIntermediateEdges, graphOutputEdges);

        GraphDesc graphDesc{};
        graphDesc.nodes = std::move(graphNodes);
        graphDesc.inputEdges = std::move(graphInputEdges);
        graphDesc.outputEdges = std::move(graphOutputEdges);
        graphDesc.intermediateEdges = std::move(graphIntermediateEdges);
        graphDesc.reuseCommandList = reuseCommandList;
        return graphDesc;
    }
}
