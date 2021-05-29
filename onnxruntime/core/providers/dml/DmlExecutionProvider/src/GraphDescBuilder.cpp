// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "GraphKernelHelper.h"

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml::GraphDescBuilder
{

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
        THROW_HR(E_UNEXPECTED);
    }

    GraphDesc BuildGraphDesc(
        const onnxruntime::OpKernelInfo& kernelInfo,
        const uint8_t* isConstGpuGraphInput,
        const size_t isConstGpuGraphInputCount,
        std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap,
        const onnxruntime::Graph& graph,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeInputDefs,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeOutputDefs,
        const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
        IDMLDevice* device,
        const void* executionHandle)
    {
        struct NodeAndIndex
        {
            uint32_t nodeIndex; // The index of the node itself
            uint32_t targetIndex; // The index of the input/output on the node (e.g. 1 for the second input on a node)
        };

        // Map from Lotus node argument names to the new node and index where it will be produced
        std::unordered_map<std::string, NodeAndIndex> nameToNodeAndIndexMap;

        // Map from Lotus node argument names to input indices of the fused kernel node.
        std::unordered_map<std::string, uint32_t> nameToFusedNodeInputIndex;

        for (size_t inputIndex = 0; inputIndex < fusedNodeInputDefs.size(); ++inputIndex)
        {
            const onnxruntime::NodeArg* graphInput = graph.GetNodeArg(
                GraphKernelHelper::GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[inputIndex]->Name()));

            if (!graphInput)
            {
                // This is a workaround for when node inputs get manipulated by transformers outside of our control,
                // which then causes them to have a different name. If that happens we can't figure out how to
                // correlate inputs to the fused graph index. This likely requires a higher-level fix, but for now
                // just bail early.
                THROW_HR(E_UNEXPECTED);
            }

            nameToFusedNodeInputIndex.emplace(graphInput->Name(), gsl::narrow_cast<uint32_t>(inputIndex));
        }

        StackAllocator<1024> allocator; // Used for converting abstract operator descs into DML_OPERATOR_DESC

        std::vector<NodeInfo> graphNodes;
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> graphInputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> graphIntermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> graphOutputEdges;

        // Get the topological sorting of Lotus nodes
        // paulm: breaking change from LOTUS that removed GetNodesInTopologicalOrder from Graph
        onnxruntime::GraphViewer viewer(graph);
        const std::vector<onnxruntime::NodeIndex>& orderedNodeIndices = viewer.GetNodesInTopologicalOrder();

        // Avoid using separate command lists for small graphs. This value can be reduced by tuning the 
        // flushing behavior of DmlCommandRecorder.  Its current behavior is to assume that graphs contain
        // enough GPU work to be worth flushing immediately.
        const uint32_t minNodeCountToReuseCommandList = 5;
        bool reuseCommandList = false;
        
        if (orderedNodeIndices.size() >= minNodeCountToReuseCommandList)
        {
            reuseCommandList = true;
        }

        auto constantCpuGraphInputGetter = [&fusedNodeInputDefs, &transferredInitializerMap](const std::string& argName)
        {
            ComPtr<OnnxTensorWrapper> tensorWrapper;

            auto iter = transferredInitializerMap.find(argName);
            if (iter != transferredInitializerMap.end())
            {
                tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(&iter->second);
            }

            return tensorWrapper;
        };

        // Iterate through each node and create a corresponding node in the new graph
        for (size_t sortedNodeIndex : orderedNodeIndices) 
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
                    const onnxruntime::NodeArg* arg = node.InputDefs()[inputIndex];
                    tensor = constantCpuGraphInputGetter(arg->Name());
                }

                return tensor;
            };

            DmlGraphNodeCreateInfo graphNodeInfo;
            graphNodeProps.internalRegInfo->graphNodeFactoryRegistration->factory(
                node,
                constantCpuNodeInputGetter,
                executionHandle,
                &graphNodeInfo
            );

            uint32_t nodeIndex = gsl::narrow_cast<uint32_t>(graphNodes.size());
            AbstractOperatorDesc opDesc = *graphNodeInfo.desc; // Make a copy

            // Retrieve lists of input and output tensor descs. These point into the opDesc, which allows us to modify
            // the tensor descs through these pointers.
            std::vector<DmlBufferTensorDesc*> inputTensorDescs = opDesc.GetInputTensors();
            std::vector<DmlBufferTensorDesc*> outputTensorDescs = opDesc.GetOutputTensors();

            // Set connections of the new node
            for (uint32_t inputIndex = 0; inputIndex < graphNodeInfo.kernelInputIndices.size(); ++inputIndex)
            {
                if (graphNodeInfo.kernelInputIndices[inputIndex] == std::numeric_limits<uint32_t>::max())
                {
                    continue;
                }

                uint32_t kernelInputIndex = graphNodeInfo.kernelInputIndices[inputIndex];

                const onnxruntime::NodeArg* arg = node.InputDefs()[kernelInputIndex];

                if (arg->Exists())
                {
                    auto iter = nameToFusedNodeInputIndex.find(arg->Name());

                    // The graph input could be missing the suffix, so try to match without it.
                    // This is part of a temporary workaround; see comments in GraphKernelHelper::GetFusedNodeArgNameMatchingGraph.
                    if (iter == nameToFusedNodeInputIndex.end())
                    {
                        iter = nameToFusedNodeInputIndex.find(GraphKernelHelper::GetFusedNodeArgNameMatchingGraph(arg->Name()));
                    }

                    if (iter != nameToFusedNodeInputIndex.end())
                    {
                        // This is a graph input

                        const uint32_t fusedNodeInputIndex = iter->second;

                        DML_INPUT_GRAPH_EDGE_DESC edge = {};
                        edge.GraphInputIndex = fusedNodeInputIndex;
                        edge.ToNodeIndex = nodeIndex;
                        edge.ToNodeInputIndex = inputIndex;
                        graphInputEdges.push_back(edge);

                        // If this is a constant input, set the appropriate flags on the desc
                        if (fusedNodeInputIndex < isConstGpuGraphInputCount && isConstGpuGraphInput[fusedNodeInputIndex])
                        {
                            DmlBufferTensorDesc* tensorDesc = inputTensorDescs[inputIndex];

                            tensorDesc->flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                        }
                    }
                    else
                    {
                        const auto& inputNodeAndIndex = nameToNodeAndIndexMap.at(arg->Name());

                        DML_INTERMEDIATE_GRAPH_EDGE_DESC edge = {};
                        edge.FromNodeIndex = inputNodeAndIndex.nodeIndex;
                        edge.FromNodeOutputIndex = inputNodeAndIndex.targetIndex;
                        edge.ToNodeIndex = nodeIndex;
                        edge.ToNodeInputIndex = inputIndex;
                        graphIntermediateEdges.push_back(edge);
                    }
                }
            }
            
            // Store the new node for lookup when downstream nodes consume it.

            for (uint32_t outputIndex = 0; outputIndex < graphNodeInfo.kernelOutputIndices.size(); ++outputIndex) 
            {
                if (graphNodeInfo.kernelOutputIndices[outputIndex] == std::numeric_limits<uint32_t>::max())
                {
                    continue;
                }

                uint32_t kernelOutputIndex = graphNodeInfo.kernelOutputIndices[outputIndex];
                const onnxruntime::NodeArg* arg = node.OutputDefs()[kernelOutputIndex];
                if (arg->Exists())
                {
                    nameToNodeAndIndexMap[arg->Name()] = NodeAndIndex{ nodeIndex, outputIndex };
                }
            }

            DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc(opDesc, &allocator);

            ComPtr<IDMLOperator> op;
            THROW_IF_FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op)));
            allocator.Reset();

            NodeInfo nodeInfo = {};
            nodeInfo.op = std::move(op);
            graphNodes.push_back(std::move(nodeInfo));
        }

        assert(graphNodes.size() == orderedNodeIndices.size());

        // Add graph output nodes, which might be in a different order from the encapsulating node
        for (size_t outputIndex = 0; outputIndex < fusedNodeOutputDefs.size(); ++outputIndex)
        {
            const onnxruntime::NodeArg* graphOutput = graph.GetNodeArg(
                GraphKernelHelper::GetFusedNodeArgNameMatchingGraph(fusedNodeOutputDefs[outputIndex]->Name()));

            THROW_HR_IF_NULL_MSG(E_POINTER, graphOutput, "FusedNode's nodeArgList does not contain one of the nodeArg");
            const auto& outputNodeAndIndex = nameToNodeAndIndexMap.at(graphOutput->Name());

            DML_OUTPUT_GRAPH_EDGE_DESC edge = {};
            edge.FromNodeIndex = outputNodeAndIndex.nodeIndex;
            edge.FromNodeOutputIndex = outputNodeAndIndex.targetIndex;
            edge.GraphOutputIndex = gsl::narrow_cast<uint32_t>(outputIndex);
            graphOutputEdges.push_back(edge);
        }
        
        GraphDesc graphDesc{};
        graphDesc.nodes = std::move(graphNodes);
        graphDesc.inputEdges = std::move(graphInputEdges);
        graphDesc.outputEdges = std::move(graphOutputEdges);
        graphDesc.intermediateEdges = std::move(graphIntermediateEdges);
        graphDesc.reuseCommandList = reuseCommandList;
        return graphDesc;
    }
}