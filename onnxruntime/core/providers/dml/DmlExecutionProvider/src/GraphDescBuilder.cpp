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

    uint32_t GetElementSize(DML_TENSOR_DATA_TYPE dataType)
    {
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_UINT8:
        case DML_TENSOR_DATA_TYPE_INT8:
            return 1;
            
        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
        case DML_TENSOR_DATA_TYPE_INT16:
            return 2;
            
        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
        case DML_TENSOR_DATA_TYPE_INT32:
            return 4;
            
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_UINT64:
        case DML_TENSOR_DATA_TYPE_INT64:
            return 8;
            
        default:
            return 0;
        }
    }

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
        const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
        IDMLDevice* device,
        const ExecutionProviderImpl* executionHandle,
        const onnxruntime::Path& modelPath,
        gsl::span<const onnxruntime::Node* const> subgraphNodes,
        gsl::span<const onnxruntime::NodeArg* const> subgraphInputs,
        gsl::span<const onnxruntime::NodeArg* const> subgraphOutputs)
    {
        struct NodeAndIndex
        {
            uint32_t nodeIndex; // The index of the node itself
            uint32_t targetIndex; // The index of the input/output on the node (e.g. 1 for the second input on a node)
        };

        // Map from Lotus node argument names to the new node and index where it will be produced
        std::unordered_map<std::string, NodeAndIndex> nameToNodeAndIndexMap;

        std::unordered_map<std::string, EdgeShapes> nodeOutputShapes;

        // Map from Lotus node argument names to input indices of the fused kernel node.
        std::unordered_map<std::string, uint32_t> nameToDmlFusedNodeInputIndex;

        for (size_t inputIndex = 0; inputIndex < subgraphInputs.size(); ++inputIndex)
        {
            const onnxruntime::NodeArg* graphInput = subgraphInputs[inputIndex];

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

        if (subgraphNodes.size() >= minNodeCountToReuseCommandList || executionHandle->IsMcdmDevice())
        {
            reuseCommandList = true;
        }

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
        std::unordered_map<std::string, std::vector<uint32_t>> inferredOutputShapes;

        for (const onnxruntime::Node* subgraphNode : subgraphNodes)
        {
            const onnxruntime::Node& node = *subgraphNode;

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

            EdgeShapes inputShapesOverrides(node.InputDefs().size());

            // Override the input shapes with shapes that were previously inferred
            for (int inputIndex = 0; inputIndex < node.InputDefs().size(); ++inputIndex)
            {
                auto inputDef = node.InputDefs()[inputIndex];

                auto outputShapesIter = inferredOutputShapes.find(inputDef->Name());
                if (outputShapesIter != inferredOutputShapes.end())
                {
                    inputShapesOverrides.GetMutableShape(inputIndex) = outputShapesIter->second;
                }
                else if (inputDef->HasTensorOrScalarShape())
                {
                    for (int i = 0; i < inputDef->Shape()->dim_size(); ++i)
                    {
                        ORT_THROW_HR_IF(E_INVALIDARG, !inputDef->Shape()->dim(i).has_dim_value());
                        inputShapesOverrides.GetMutableShape(inputIndex).push_back(gsl::narrow_cast<uint32_t>(inputDef->Shape()->dim(i).dim_value()));
                    }
                }
            }

            EdgeShapes outputShapes;
            DmlGraphNodeCreateInfo graphNodeCreateInfo;
            graphNodeProps.internalRegInfo->graphNodeFactoryRegistration->factory(
                node,
                constantCpuNodeInputGetter,
                executionHandle,
                &inputShapesOverrides,
                /*out*/ &outputShapes,
                /*out*/ &graphNodeCreateInfo
            );

            ORT_THROW_HR_IF(E_UNEXPECTED, outputShapes.EdgeCount() != node.OutputDefs().size());
            for (int i = 0; i < node.OutputDefs().size(); ++i)
            {
                inferredOutputShapes[node.OutputDefs()[i]->Name()] = outputShapes.GetShape(i);
            }

            // Create a map between operatorGraphNodeIndex to mainGraphNodeIndex.
            std::unordered_map<uint32_t, uint32_t> operatorGraphNodeIndexToMainGraphNodeIndexMap;
            uint32_t graphNodeCount = gsl::narrow_cast<uint32_t>(graphNodes.size());
            const bool isNodeAsOpDesc = graphNodeCreateInfo.nodesAsOperatorDesc.size() > 0;
            size_t firstOpDescGraphNode = graphNodes.size();

            if (isNodeAsOpDesc)
            {
                // Can't populate graphNodes vector at this point, because operatorDesc may get modified later.
                for (uint32_t nodeIndex = 0; nodeIndex < graphNodeCreateInfo.nodeCount; nodeIndex++)
                {
                    ORT_THROW_HR_IF(E_UNEXPECTED, !graphNodeCreateInfo.nodesAsOperatorDesc[nodeIndex]);
                    operatorGraphNodeIndexToMainGraphNodeIndexMap.emplace(nodeIndex, graphNodeCount++);
                }

                graphNodes.resize(graphNodes.size() + graphNodeCreateInfo.nodeCount);
            }
            else
            {
                for (uint32_t nodeIndex = 0; nodeIndex < graphNodeCreateInfo.nodeCount; nodeIndex++)
                {
                    ORT_THROW_HR_IF(E_UNEXPECTED, !graphNodeCreateInfo.nodesAsIDMLOperator[nodeIndex].Get());
                    operatorGraphNodeIndexToMainGraphNodeIndexMap.emplace(nodeIndex, graphNodeCount++);
                    NodeInfo nodeInfo = {};
                    nodeInfo.nodeDef = std::move(graphNodeCreateInfo.nodesAsIDMLOperator[nodeIndex]);
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

                        // If this is a constant input, set the appropriate flags on the desc
                        if (isNodeAsOpDesc &&
                            dmlFusedNodeInputIndex < isConstGpuGraphInputCount &&
                            isConstGpuGraphInput[dmlFusedNodeInputIndex])
                        {
                            auto& operatorGraphInputNode = graphNodeCreateInfo.nodesAsOperatorDesc[operatorGraphInputEdge.ToNodeIndex];
                            std::vector<DmlBufferTensorDesc*> toNodeInputTensorDescs = operatorGraphInputNode->GetInputTensors();
                            DmlBufferTensorDesc* tensorDesc = toNodeInputTensorDescs[operatorGraphInputEdge.ToNodeInputIndex];
                            size_t c_maxConstNodeDataSize = 8;

                            ComPtr<OnnxTensorWrapper> constantInput = constantCpuGraphInputGetter(arg->Name());

                            if (constantInput)
                            {
                                std::vector<uint8_t> tensorData;
                                tensorData.insert(
                                    tensorData.begin(), 
                                    static_cast<const uint8_t*>(constantInput->GetData()), 
                                    static_cast<const uint8_t*>(constantInput->GetData()) + GetElementSize(tensorDesc->dataType));
                                if (tensorData.size() <= c_maxConstNodeDataSize)
                                {
                                    NodeInfo nodeInfo = {};
                                    nodeInfo.nodeDef = std::move(tensorData);
                                    graphNodes.push_back(std::move(nodeInfo));

                                    DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
                                    edge.FromNodeIndex = static_cast<UINT>(graphNodes.size() - 1);
                                    edge.FromNodeOutputIndex = 0;
                                    edge.ToNodeIndex = mainGraphNodeIndex;
                                    edge.ToNodeInputIndex = operatorGraphInputEdge.ToNodeInputIndex;
                                    graphIntermediateEdges.push_back(edge);
                                }
                            }
                            else
                            {
                                DML_INPUT_GRAPH_EDGE_DESC edge = {};
                                edge.GraphInputIndex = dmlFusedNodeInputIndex;
                                edge.ToNodeIndex = mainGraphNodeIndex;
                                edge.ToNodeInputIndex = operatorGraphInputEdge.ToNodeInputIndex;
                                graphInputEdges.push_back(edge);

                                tensorDesc->flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                            }
                        }
                        else
                        {
                            DML_INPUT_GRAPH_EDGE_DESC edge = {};
                            edge.GraphInputIndex = dmlFusedNodeInputIndex;
                            edge.ToNodeIndex = mainGraphNodeIndex;
                            edge.ToNodeInputIndex = operatorGraphInputEdge.ToNodeInputIndex;
                            graphInputEdges.push_back(edge);
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

                    nodeOutputShapes[arg->Name()] = outputShapes;
                }
            }

            if (isNodeAsOpDesc)
            {
                for (uint32_t i = 0; i < graphNodeCreateInfo.nodesAsOperatorDesc.size(); ++i)
                {
                    auto& opDesc = graphNodeCreateInfo.nodesAsOperatorDesc[i];

                    DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc(*opDesc, &allocator);

                    // TODO: Change as new header is ingested
                    if (dmlDesc.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING)
                        dmlDesc.Type = (DML_OPERATOR_TYPE) 169;
                
                    // TODO: Change as new header is ingested
                    if (dmlDesc.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT)
                        dmlDesc.Type = (DML_OPERATOR_TYPE) 170;

                    ComPtr<IDMLOperator> op;
                    ORT_THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
                    allocator.Reset();

                    NodeInfo nodeInfo = {};
                    nodeInfo.nodeDef = std::move(op);
                    nodeInfo.name = node.Name();
                    graphNodes[firstOpDescGraphNode + i] = std::move(nodeInfo);
                }
            }
        }

        EdgeShapes graphOutputShapes(subgraphOutputs.size());

        // Add graph output nodes, which might be in a different order from the encapsulating node
        for (size_t outputIndex = 0; outputIndex < subgraphOutputs.size(); ++outputIndex)
        {
            const onnxruntime::NodeArg* graphOutput = subgraphOutputs[outputIndex];

            ORT_THROW_HR_IF_NULL_MSG(E_POINTER, graphOutput, "FusedNode's nodeArgList does not contain one of the nodeArg");
            const auto& outputNodeAndIndex = nameToNodeAndIndexMap.at(graphOutput->Name());

            DML_OUTPUT_GRAPH_EDGE_DESC edge = {};
            edge.FromNodeIndex = outputNodeAndIndex.nodeIndex;
            edge.FromNodeOutputIndex = outputNodeAndIndex.targetIndex;
            edge.GraphOutputIndex = gsl::narrow_cast<uint32_t>(outputIndex);
            graphOutputEdges.push_back(edge);
            graphOutputShapes.GetMutableShape(outputIndex) = nodeOutputShapes[graphOutput->Name()].GetShape(outputNodeAndIndex.targetIndex);
        }

        RemoveUnconnectedNodes(graphNodes, graphInputEdges, graphIntermediateEdges, graphOutputEdges);

        GraphDesc graphDesc{};
        graphDesc.nodes = std::move(graphNodes);
        graphDesc.inputEdges = std::move(graphInputEdges);
        graphDesc.outputEdges = std::move(graphOutputEdges);
        graphDesc.intermediateEdges = std::move(graphIntermediateEdges);
        graphDesc.reuseCommandList = reuseCommandList;
        graphDesc.outputShapes = std::move(graphOutputShapes);
        return graphDesc;
    }
}
