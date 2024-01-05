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
#include "core/providers/dml/OperatorAuthorHelper/OperatorVersions.h"
#include "core/framework/kernel_lookup.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/graph/graph_utils.h"

namespace Dml
{
    GraphTransformer::GraphTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    ) : onnxruntime::GraphTransformer(name),
        m_providerImpl(static_cast<const ExecutionProvider*>(provider)->GetImpl())
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
        PerformOperatorFusion(&graph, m_providerImpl->IsMcdmDevice(), &transformModifiedGraph);
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

    void GraphTransformer::PerformOperatorFusion(onnxruntime::Graph* graph, bool isMcdmDevice, bool* modified) const
    {
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
            // Ignore the nodes which were not assigned to the DML by ORT during IExecutionProvider::GetCapability()
            if (!onnxruntime::graph_utils::IsSupportedProvider(node, {onnxruntime::kDmlExecutionProvider}))
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

            // Can't fuse if outputNode was not assigned to the DML by ORT during IExecutionProvider::GetCapability()
            if (!onnxruntime::graph_utils::IsSupportedProvider(outputNode, GetCompatibleExecutionProviders()))
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
                outputNode.Op()->SinceVersion(),
                isMcdmDevice);

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
            ORT_THROW_HR_IF(E_UNEXPECTED, !nodesRemoved);

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
            
            node.SetExecutionProviderType(onnxruntime::kDmlExecutionProvider);
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
                node.AddAttributeProto(attribute.second);
            }
        }
    }

} // namespace Dml
