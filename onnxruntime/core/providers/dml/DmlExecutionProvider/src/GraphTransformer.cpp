// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "GraphTransformer.h"
#include "Operators/OperatorRegistration.h"
#include "Operators/OperatorUtility.h"
#include "ExecutionProvider.h"
#include "GraphPartitioner.h"
#include "core/providers/dml/OperatorAuthorHelper/Attributes.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorRegistration.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_utils.h"

namespace Dml
{
    GraphTransformer::GraphTransformer(
        const std::string& name, 
        const onnxruntime::IExecutionProvider* provider
    )
        : onnxruntime::GraphTransformer(name),
          m_providerImpl(static_cast<const ExecutionProvider* >(provider)->GetImpl())
    {
    }

    onnxruntime::common::Status GraphTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level, const onnxruntime::logging::Logger&) const 
    {
      modified = false;

      // Perform fusion
      {
        bool transformModifiedGraph = false;
        PerformOperatorFusion(&graph, &transformModifiedGraph);
        modified |= transformModifiedGraph;

        transformModifiedGraph = false;
        PerformQuantizedOperatorDecomposition(&graph, &transformModifiedGraph);
        modified |= transformModifiedGraph;

        if (modified) 
        {
            ORT_RETURN_IF_ERROR(graph.Resolve());
        }
      }

      return onnxruntime::common::Status::OK();
    }

    static std::string GetUniqueNodeName(const onnxruntime::Node* node)
    {
        std::stringstream ss;
        ss << '#' << node->Index();
        if (!node->Name().empty())
        {
            ss << " \'" << node->Name() << '\'';
        }
        return ss.str();
    }
    
    void GraphTransformer::PerformOperatorFusion(onnxruntime::Graph* graph, bool* modified) const
    {
        onnxruntime::KernelRegistry* registry = m_providerImpl->GetKernelRegistry().get();

        struct NodeToAdd
        {
            std::string name;
            std::string description;
            std::string opType;
            std::string domain;
            onnxruntime::NodeAttributes attributes;
            std::string activationOpType;
            std::string activationOpDomain;
            int activationOpVersion;
            onnxruntime::NodeAttributes activationAttributes;
            std::vector<onnxruntime::NodeArg*> inputs;
            std::vector<onnxruntime::NodeArg*> outputs;
        };

        // Defer adding new nodes to the graph until after we're done iterating over it, because we can't mutate the
        // graph while iterating over it
        std::vector<NodeToAdd> nodesToAdd;

        for (auto& node : graph->Nodes())
        {
            // We need to predict whether the nodes will be assigned to the DML transformer by Lotus,
            // which occurs in IExecutionProvider::GetCapability.

            bool allow64BitInputThroughStrides = false;
            if (!IsNodeSupportedByDml(
                node,
                *registry,
                m_providerImpl->GetSupportedDeviceDataTypeMask(),
                *m_providerImpl->GetInternalRegistrationInfoMap().get(),
                allow64BitInputThroughStrides,
                nullptr))
            {
                // Can't fuse nodes that don't belong to this execution provider
                continue;
            }

            // The number of nodes which use the result of this convolution as input
            const auto outputNodeCount = std::distance(node.OutputEdgesBegin(), node.OutputEdgesEnd());

            if (outputNodeCount != 1)
            {
                // Can only fuse nodes whose only output feeds into a single activation - if multiple nodes use the
                // output of this one, we can't fuse it.
                continue;
            }

            const auto& outputNode = *node.OutputNodesBegin();

            // We need to predict whether the nodes will be assigned to the DML transformer by Lotus,
            // which occurs in IExecutionProvider::GetCapability.
            if (!onnxruntime::KernelRegistry::HasImplementationOf(*registry, outputNode, onnxruntime::kDmlExecutionProvider)) 
            {
                // Can't fuse nodes that don't belong to this execution provider
                continue;
            }

            if (outputNode.InputDefs().size() != 1)
            {
                // Can only fuse activation functions that take a single input
                continue;
            }

            auto fusedOpProperties = FusionHelpers::TryGetFusedOp(
                node.OpType(),
                node.Domain(),
                node.Op()->SinceVersion(),
                gsl::narrow_cast<uint32_t>(node.InputDefs().size()),
                outputNode.OpType(),
                outputNode.Domain(),
                outputNode.Op()->SinceVersion());

            if (!fusedOpProperties)
            {
                // These operators can't be fused
                continue;
            }

            const auto& fuseableNode = node;
            const auto& activationNode = outputNode;

            // Fusable nodes only produce one output
            assert(fuseableNode.OutputDefs().size() == 1);

            // Activation only produces one output
            assert(activationNode.OutputDefs().size() == 1);

            // Add a new node that represents the combination of the fuseable node and the activation node.
            NodeToAdd fusedNode;
            fusedNode.name = "fused op (" + GetUniqueNodeName(&fuseableNode) + ") + (" + GetUniqueNodeName(&activationNode) + ")";
            fusedNode.description = "";
            fusedNode.opType = fusedOpProperties->opType;
            fusedNode.activationOpType = activationNode.OpType();
            fusedNode.activationOpDomain = activationNode.Domain();
            fusedNode.activationOpVersion = activationNode.Op()->SinceVersion();
            fusedNode.domain = fusedOpProperties->domain;

            // Make a copy of the attributes of both nodes
            fusedNode.attributes = fuseableNode.GetAttributes();
            fusedNode.activationAttributes = activationNode.GetAttributes();

            // Inputs to the fused node are the inputs to the fuseable node
            for (const auto *input : fuseableNode.InputDefs()) 
            {
                fusedNode.inputs.push_back(graph->GetNodeArg(input->Name()));
            }

            // Outputs from the fused node are the outputs to the activation node
            for (const auto *output : activationNode.OutputDefs())
            {
                fusedNode.outputs.push_back(graph->GetNodeArg(output->Name()));
            }

            nodesToAdd.push_back(std::move(fusedNode));

            onnxruntime::graph_utils::RemoveNodeOutputEdges(*graph, const_cast<onnxruntime::Node&>(fuseableNode));
            onnxruntime::graph_utils::RemoveNodeOutputEdges(*graph, const_cast<onnxruntime::Node&>(activationNode));

            // Remove the fuseable and activation nodes - they're replaced by the fused node
            bool nodesRemoved = false;
            nodesRemoved = graph->RemoveNode(fuseableNode.Index());
            nodesRemoved &= graph->RemoveNode(activationNode.Index());
            THROW_HR_IF(E_UNEXPECTED, !nodesRemoved);

            *modified = true;
        }

        for (auto& nodeToAdd : nodesToAdd)
        {
            auto& node = graph->AddNode(
                nodeToAdd.name,
                nodeToAdd.opType,
                nodeToAdd.description,
                nodeToAdd.inputs,
                nodeToAdd.outputs,
                &nodeToAdd.attributes,
                nodeToAdd.domain);

            // Add a dynamic attribute to the fuseable operator to specify activation
            node.AddAttribute(AttrName::FusedActivation, nodeToAdd.activationOpType);
            node.AddAttribute(AttrName::FusedActivationDomain, nodeToAdd.activationOpDomain);
            node.AddAttribute(AttrName::FusedActivationSinceVersion, static_cast<int64_t>(nodeToAdd.activationOpVersion));

            // Copy all attributes from activation into the fuseable node (with new names)
            for (auto& attribute : nodeToAdd.activationAttributes)
            {
                // Change the name of the attribute to its fused node version
                std::string fusedAttributeName = Dml::FusionHelpers::GetFusedAttributeName(attribute.first);
                attribute.second.set_name(fusedAttributeName);
                node.AddAttribute(fusedAttributeName, attribute.second);
            }
        }
    }

    // Converts certain QLinear operations unsupported by the DML API into a sequence of DeQuantizeLinear, 32-bit operator, QuantizeLinear
    void GraphTransformer::PerformQuantizedOperatorDecomposition(onnxruntime::Graph* graph, bool* modified) const
    {
        struct NodeToAdd
        {
            std::string name;
            std::string description;
            std::string opType;
            std::string domain;
            onnxruntime::NodeAttributes attributes;
            std::vector<onnxruntime::NodeArg*> inputs;
            std::vector<onnxruntime::NodeArg*> outputs;
        };
        
        // Defer adding and removing nodes in the graph until after we're done iterating over it, because we can't mutate the
        // graph while iterating over it
        std::vector<NodeToAdd> nodesToAdd;
        std::vector<onnxruntime::NodeIndex> nodesToRemove;

        for (auto& node : graph->Nodes())
        {
            // For now, only QLinearSigmoid is handled
            if (node.Domain() == onnxruntime::kMSDomain &&
                node.OpType() == "QLinearSigmoid")
            {
                // Intermediate node arg type proto with floating point format
                onnx::TypeProto floatTensorProto;
                floatTensorProto.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);

                // Add intermediate graph edges for the input and output of the FP32 sigmoid operator
                auto* sigmoidInputArg = &graph->GetOrCreateNodeArg("decomposed_QLinearSigmoid_input_" + GetUniqueNodeName(&node), &floatTensorProto);
                auto* sigmoidOutputArg = &graph->GetOrCreateNodeArg("decomposed_QLinearSigmoid_output_" + GetUniqueNodeName(&node), &floatTensorProto);

                {
                    NodeToAdd dequantizeNode;
                    dequantizeNode.name = "decomposed_QLinearSigmoid_DequantizeLinear_" + GetUniqueNodeName(&node);
                    dequantizeNode.description = "";
                    dequantizeNode.opType = "DequantizeLinear";
                    dequantizeNode.domain = "";     

                    dequantizeNode.inputs.push_back(graph->GetNodeArg(node.InputDefs()[0]->Name()));
                    dequantizeNode.inputs.push_back(graph->GetNodeArg(node.InputDefs()[1]->Name()));
                    dequantizeNode.inputs.push_back(graph->GetNodeArg(node.InputDefs()[2]->Name()));
                    dequantizeNode.outputs.push_back(sigmoidInputArg);
                
                    nodesToAdd.push_back(std::move(dequantizeNode));
                }

                {
                    NodeToAdd sigmoidNode;
                    sigmoidNode.name = "decomposed_QLinearSigmoid_Sigmoid_" + GetUniqueNodeName(&node);
                    sigmoidNode.description = "";
                    sigmoidNode.opType = "Sigmoid";
                    sigmoidNode.domain = ""; 
                    sigmoidNode.inputs.push_back(sigmoidInputArg);
                    sigmoidNode.outputs.push_back(sigmoidOutputArg); 
                    nodesToAdd.push_back(std::move(sigmoidNode)); 
                }

                {
                    NodeToAdd quantizeNode;
                    quantizeNode.name = "decomposed_QLinearSigmoid_QuantizeLinear_" + GetUniqueNodeName(&node);
                    quantizeNode.description = "";
                    quantizeNode.opType = "QuantizeLinear";
                    quantizeNode.domain = "";
            
                    quantizeNode.inputs.push_back(sigmoidOutputArg);
                    quantizeNode.inputs.push_back(graph->GetNodeArg(node.InputDefs()[3]->Name()));
                    quantizeNode.inputs.push_back(graph->GetNodeArg(node.InputDefs()[4]->Name()));
                    quantizeNode.outputs.push_back(graph->GetNodeArg(node.OutputDefs()[0]->Name()));
 
                    nodesToAdd.push_back(std::move(quantizeNode)); 
                }

                nodesToRemove.push_back(node.Index());
                *modified = true;
            }
        }

        for (auto& nodeToAdd : nodesToAdd)
        {
            auto& node = graph->AddNode(
                nodeToAdd.name,
                nodeToAdd.opType,
                nodeToAdd.description,
                nodeToAdd.inputs,
                nodeToAdd.outputs,
                &nodeToAdd.attributes,
                nodeToAdd.domain);
        }

        for (const auto& nodeIndex : nodesToRemove) 
        {
            onnxruntime::Node* node = graph->GetNode(nodeIndex);
            onnxruntime::graph_utils::RemoveNodeOutputEdges(*graph, *node);
            graph->RemoveNode(node->Index());
        }
    }

} // namespace Dml
