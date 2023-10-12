#include "core/optimizer/matmul_bn_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"


namespace onnxruntime
{
    void AddNodesToRemove(
        Node::NodeConstIterator currItr,
        const NodeIndex& destNodeIndex,
        std::vector<NodeIndex>& nodesToRemove)
    {
      while (currItr->Index() != destNodeIndex) {
        nodesToRemove.push_back(currItr->Index());
        currItr = currItr->OutputNodesBegin();
      }
    }

    NodeIndex GetOtherParentOfNode(
        const Node& node,
        NodeIndex firstParentIndex)
    {
        NodeIndex otherParentIndex = std::numeric_limits<size_t>::max();
        if (node.GetInputEdgesCount() != 2)
        {
        return otherParentIndex;
        }

        auto parentNodeItr = node.InputNodesBegin();
        if (parentNodeItr->Index() != firstParentIndex)
        {
            otherParentIndex = parentNodeItr->Index();
        }
        ++parentNodeItr;
        if (parentNodeItr->Index() != firstParentIndex)
        {
            otherParentIndex = parentNodeItr->Index();
        }
        return otherParentIndex;
    }

    bool MatmulBNFusion::MatchPath(
        const Node& parentNode,
        const gsl::span<std::pair<std::string, std::initializer_list<int>>>& path,
        const Node& childNode) const
    {
        if (path.size() == 0)
        {
            return true;
        }

        if (!graph_utils::IsSupportedOptypeVersionAndDomain(childNode, path[0].first, path[0].second) ||
            childNode.GetExecutionProviderType() != parentNode.GetExecutionProviderType())
        {
            return false;
        }

        // last node in the path can have more than one output
        // because all those outputs will be preserved by the addition of new Gemm node
        if (path.size() > 1 && childNode.GetOutputEdgesCount() != 1)
        {
            return false;
        }

        return MatchPath(childNode, path.subspan(1), *childNode.OutputNodesBegin());
    }

    /*
    *   Given a MatMul node, it will verify the following pattern.
    *                      MatMul
    *                        |
    *                       / \
    *                      /   \
    *                     /     \
    *               Reshape     Shape
    *                  |          |
    *             Transpose      Cast
    *                  |          |
    *        BatchNormalization  Cast
    *                  |          |
    *              Transpose      |
    *                  |         /
    *                   \       /
    *                    \     /
    *                     \   /
    *                       | 
    *                    Reshape
    * As of writing this fusion, we are being conversative in the pattern because the customer
    * model we are targeting has this exact pattern. Above pattern will evolve in the future 
    * as we tend to add separate fusion to eliminate Transpose around the BatchNormalization, 
    * update the model optimizer script to eliminate adjacent Cast operator, etc.
    * 
    * We have to match the path (MatMul->Shape->Cast->Cast->Reshape) because sub-merging the 
    * BatchNormalization into the MatMul will change MatMul's output and thus we have to make 
    * sure that MatMul's output is not used by any operator to which MatMul's output matters.
    * Other Conditions:
    *   - B tensor of MatMul should be constant.
    *   - scale, B, mean, var tensors of BatchNormalization should be constant.
    *   - Every node in the path except first and last node, should have only 1 output edge.
    */
    bool MatmulBNFusion::SatisfyCondition(
        const Graph& graph,
        const Node& node,
        const logging::Logger&) const
    {
        if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", { 1, 9, 13 }) ||
            node.GetOutputEdgesCount() != 2)
        {
            return false;
        }

        auto childNodeIterator = node.OutputNodesBegin();
        const Node& firstChildNode = *childNodeIterator;
        ++childNodeIterator;
        const Node& secondChildNode = *childNodeIterator;

        std::vector<std::pair<std::string, std::initializer_list<int>>> firstPath = 
        {{"Reshape", {1, 5}},
         {"Transpose", {1}},
         {"BatchNormalization", {1, 6, 7}},
         {"Transpose", {1}},
         {"Reshape", {1, 5}}};

        std::vector<std::pair<std::string, std::initializer_list<int>>> secondPath =
        {{"Shape", {1}},
         {"Cast", {1, 6}},
         {"Cast", {1, 6}},
         {"Reshape", {1, 5}}};

        if (!(MatchPath(node, firstPath, firstChildNode) ^ MatchPath(node, secondPath, firstChildNode)))
        {
            return false;
        }

        if (!(MatchPath(node, firstPath, secondChildNode) ^ MatchPath(node, secondPath, secondChildNode))) {
            return false;
        }

        
        const auto& batchNormNode = firstChildNode.OpType() == "Reshape" ?
            *firstChildNode.OutputNodesBegin()->OutputNodesBegin() :
            *secondChildNode.OutputNodesBegin()->OutputNodesBegin();
        
        // Check that the appropriate inputs to the Matmul and BN nodes are constants.
        if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[1]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[2]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[3]) ||
            !graph_utils::NodeArgIsConstant(graph, *batchNormNode.InputDefs()[4]))
        {
            return false;
        }

        // First output from BN is required. Others are optional. If any optional outputs exist we can't fuse.
        const auto& output_defs = batchNormNode.OutputDefs();
        if (output_defs.size() > 1) {
            for (size_t i = 1, end = output_defs.size(); i < end; ++i) {
              if (output_defs[i] != nullptr && output_defs[i]->Exists())
                return false;
            }
        }

        if (graph.NodeProducesGraphOutput(node)) {
            return false;
        }

        return true;
    }

    Status MatmulBNFusion::Apply(
        Graph& graph,
        Node& matmulNode,
        RewriteRuleEffect& ruleEffect,
        const logging::Logger&) const
    {
        auto childNodeIterator = matmulNode.OutputNodesBegin();
        const Node& firstChildNode = *childNodeIterator;
        ++childNodeIterator;
        const Node& secondChildNode = *childNodeIterator;

        const Node& firstReshape = firstChildNode.OpType() == "Reshape" ? firstChildNode : secondChildNode;

        NodeIndex batchNormNodeIndex = firstReshape.OutputNodesBegin()->OutputNodesBegin()->Index();
        Node& batchNormNode = *graph.GetNode(batchNormNodeIndex);

        // only perform fusion if eplison is present and is of float_32 type
        auto epsilonAttr = batchNormNode.GetAttributes().find("epsilon");
        if (epsilonAttr == batchNormNode.GetAttributes().end() ||
            epsilonAttr->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT)
        {
            return Status::OK();
        }
        const float epsilon = epsilonAttr->second.f();

        const onnx::TensorProto* scaleTensor = graph_utils::GetConstantInitializer(graph, batchNormNode.InputDefs()[1]->Name());
        ORT_ENFORCE(scaleTensor);
        const onnx::TensorProto* biasTensor = graph_utils::GetConstantInitializer(graph, batchNormNode.InputDefs()[2]->Name());
        ORT_ENFORCE(biasTensor);
        const onnx::TensorProto* meanTensor = graph_utils::GetConstantInitializer(graph, batchNormNode.InputDefs()[3]->Name());
        ORT_ENFORCE(meanTensor);
        const onnx::TensorProto* varTensor = graph_utils::GetConstantInitializer(graph, batchNormNode.InputDefs()[4]->Name());
        ORT_ENFORCE(varTensor);
        const onnx::TensorProto* matmulBTensor = graph_utils::GetConstantInitializer(graph, matmulNode.InputDefs()[1]->Name());
        ORT_ENFORCE(matmulBTensor);

        if (!optimizer_utils::IsFloatingPointDataType(*matmulBTensor) ||
            !optimizer_utils::IsFloatingPointDataType(*scaleTensor) ||
            !optimizer_utils::IsFloatingPointDataType(*biasTensor) ||
            !optimizer_utils::IsFloatingPointDataType(*meanTensor) ||
            !optimizer_utils::IsFloatingPointDataType(*varTensor) ||
            scaleTensor->dims_size() != 1 ||
            biasTensor->dims_size() != 1 ||
            meanTensor->dims_size() != 1 ||
            varTensor->dims_size() != 1 ||
            scaleTensor->dims(0) != matmulBTensor->dims(1) ||
            biasTensor->dims(0) != matmulBTensor->dims(1) ||
            meanTensor->dims(0) != matmulBTensor->dims(1) ||
            varTensor->dims(0) != matmulBTensor->dims(1))
        {
            return Status::OK();
        }
        
        /*
        * temp = scale / sqrt(var + epsilon)
        * output = (temp * Input) - ((temp * mean) + bias)
        */
        Initializer scale(*scaleTensor, graph.ModelPath());
        Initializer bias(*biasTensor, graph.ModelPath());
        Initializer mean(*meanTensor, graph.ModelPath());
        Initializer var(*varTensor, graph.ModelPath());
        Initializer matmulB(*matmulBTensor, graph.ModelPath());

        var.add(epsilon);
        var.sqrt();
        scale.div(var); // this is the temp
        matmulB.scale_to_axis(scale, 1);

        mean.mul(scale);
        bias.sub(mean);
        
        // create B tensorProto for new Gemm node from <matmulB> initializer.
        ONNX_NAMESPACE::TensorProto newGemmBTensor(*matmulBTensor);
        matmulB.ToProto(newGemmBTensor);
        const std::string newGemmBName = graph.GenerateNodeArgName("MatMulBnFusion_GemmB_" + matmulBTensor->name());
        newGemmBTensor.set_name(newGemmBName);
        NodeArg& newGemmBNodeArg = graph_utils::AddInitializer(graph, newGemmBTensor);

        // create bias tensorProto for new Gemm node from <bias> initializer.
        ONNX_NAMESPACE::TensorProto newGemmBiasTensor(*biasTensor);
        bias.ToProto(newGemmBiasTensor);
        const std::string newGemmBiasName = graph.GenerateNodeArgName("MatMulBnFusion_GemmBias");
        newGemmBiasTensor.set_name(newGemmBiasName);
        NodeArg& newGemmBiasNodeArg = graph_utils::AddInitializer(graph, newGemmBiasTensor);

        NodeIndex lastReshapeNodeIndex = firstReshape.OutputNodesBegin()->OutputNodesBegin()->
                                         OutputNodesBegin()->OutputNodesBegin()->Index();
        graph.AddNode(
            graph.GenerateNodeArgName("MatMulBnFusion_Gemm"),
            "Gemm",
            "Generated from Matmul BatchNormalization fusion",
            {matmulNode.MutableInputDefs()[0], &newGemmBNodeArg, &newGemmBiasNodeArg},
            graph.GetNode(lastReshapeNodeIndex)->MutableOutputDefs(),
            nullptr,
            kOnnxDomain);
        
        std::vector<NodeIndex> nodesToRemove;
        nodesToRemove.push_back(matmulNode.Index());

        // Remove non-Matmul parent of Reshape if and only if
        // that parent has only 1 output.
        NodeIndex nonMatmulParentOfFirstReshape = GetOtherParentOfNode(firstReshape, matmulNode.Index());
        if (nonMatmulParentOfFirstReshape != std::numeric_limits<size_t>::max() &&
            graph.GetNode(nonMatmulParentOfFirstReshape)->GetOutputEdgesCount() == 1)
        {
            nodesToRemove.push_back(nonMatmulParentOfFirstReshape);
        }

        auto currItr = matmulNode.OutputNodesBegin();
        AddNodesToRemove(currItr, lastReshapeNodeIndex, nodesToRemove);
        ++currItr;
        AddNodesToRemove(currItr, lastReshapeNodeIndex, nodesToRemove);
        nodesToRemove.push_back(lastReshapeNodeIndex);

        for (const auto& nodeIndex : nodesToRemove) {
            Node* node = graph.GetNode(nodeIndex);
            graph_utils::RemoveNodeOutputEdges(graph, *node);
            graph.RemoveNode(nodeIndex);
        }

        ruleEffect = RewriteRuleEffect::kRemovedCurrentNode;
        return Status::OK();
    }
}