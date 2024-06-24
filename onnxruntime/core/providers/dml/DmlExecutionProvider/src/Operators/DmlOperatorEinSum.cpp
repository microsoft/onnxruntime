// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

// With a single equation, the Einstein summation operator can represent a variety of operators including: matmul,
// summation, transposition, diagonal slice, diagonal sum (trace), inner (dot) product, outer product...
//
// Parameters                   NumPy equivalent                Description
// -------------------------------------------------------------------------------------------------------------
// ('i', A1)                    A1                              returns a view of A1
// ('i->', A1)                  sum(A1)                         sums the values of A1
// ('i,i->i', A1, B1)           A1 * B1                         element-wise multiplication of A1 and B1
// ('i,i->', A1, B1)            inner(A1, B1) or dot(A1, B1)    inner product of A1 and B1
// ('i,i', A1, B1)              inner(A1, B1) or dot(A1, B1)    inner product of A1 and B1
// ('i,j->ij', A1, B1)          outer(A1, B1)                   outer product of A1 and B1
// ('ij->ij', A2)               A2                              returns a view of A2
// ('ij', A2)                   A2                              returns a view of A2
// ('ji', A2)                   A2.T                            view transpose of A2
// ('ji->ij', A2)               A2.T                            view transpose of A2
// ('ii->i', A2)                diag(A2)                        view main diagonal of A2
// ('ii->', A2)                 trace(A2)                       sums main diagonal of A2
// ('ij->', A2)                 sum(A2)                         sums the values of A2
// ('ij->j', A2)                sum(A2, axis=0)                 sum down the columns of A2 (across rows)
// ('ij->i', A2)                sum(A2, axis=1)                 sum horizontally along the rows of A2
// ('ij,ij->ij', A2, B2)        A2 * B2                         element-wise multiplication of A2 and B2
// ('ij,ji->ij', A2, B2)        A2 * B2.transpose()             element-wise multiplication of A2 and B2.T
// ('ij,jk', A2, B2)            matmul(A2, B2) or dot(A2, B2)   matrix multiplication of A2 and B2
// ('ij,jk->ik', A2, B2)        matmul(A2, B2) or dot(A2, B2)   matrix multiplication of A2 and B2
// ('bij,bjk->bik', A2, B2)     matmul(A3, B3)                  matrix multiplication of A3 and B3 (a stack of 2D matrices)
// ('bij,bkj->bik', A2, B2)     matmul(A3, transpose(B3))       matrix multiplication of A3 and B3 (a stack of 2D matrices)
// ('ij,kj->ik', A2, B2)        inner(A2, B2)                   inner product of A2 and B2
// ('ij,kj->ikj', A2, B2)       A2[:, None] * B2                each row of A2 multiplied by B2
// ('ij,kl->ijkl', A2, B2)      A2[:, :, None, None] * B2       each value of A2 multiplied by B2
// (',ij', 3, B2)                                               Scalar times array: array([[ 0, 3, 6], [ 9, 12, 15]])
// ("ij,j", A2, B1)             matvec(A2, B1)                  Matrix and vector.
// ("ii,ii->i", A2, B2)         A2.diag() * B2.diag()           diagonals multiplied by each other
// ("ii,ii->", A2, B2)          dot(A2.diag(), B2.diag())       dot product of diagonals
//
// Decomposition:
//
// Ultimately though EinSum is equivalent to an elementwise multiplication into an internal product tensor
// (given a helper function to reproject all inputs so they're shape-compatible) followed by sum reduction.
//
// 1. Determine the size of the internal product tensor by concatenating the dimensions of all inputs,
//    counting each unique label once. So "bij,bjk->bik" would yield an internal product of shape [b,i,j,k].
// 2. Project each input tensor as needed to the internal product shape (transposing and/or broadcasting).
//    So an input of shape [b,i] with product shape of [b,j,i,k] would insert broadcasted j and k dimensions.
//    An input of shape [a,b,c] with product shape of [b,c,a] would require a transpose.
//    The input shape [a,b,a] with product shape of [a,b] would collapse the first two input 'a' dimensions.
// 3. Multiply elementwise every input tensor to compute the internal product.
// 4. Sum reduce the product tensor to the final output shape, reducing along any missing dimensions.
//    So a product shape of [b,j,i,k] and output shape of [b,i,k] reduces along j.
//
//  ReduceSum(
//      Mul(
//          ExpandTransposeCollapseAsNeeded(A, aAxesToProductAxes),
//          ExpandTransposeCollapseAsNeeded(B, bAxesToProductAxes),
//      ),
//      reductionAxes,
//      keepdims=false
//  )
//
// Notes:
//
// - DirectML has no direct EinSum operator, but common cases map to existing operators.
// - EinSum can accept a variable number of input tensors, but the DML EP only supports a limited count
//   (falling back to CPU otherwise).

namespace Dml
{

class DmlOperatorEinSum : public DmlOperator, public EinSumHelper
{
public:
    DmlOperatorEinSum(const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t opsetVersion)
    :   DmlOperator(kernelCreationContext),
        EinSumHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription(), opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 1, "EinSum expects at least one input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "EinSum expects one output tensor.");
        ML_CHECK_VALID_ARGUMENT(
            static_cast<uint64_t>(kernelCreationContext.GetInputCount()) + 1 == m_components.size(),
            "EinSum input tensor count is inconsistent with the equation component count."
        );
        assert(m_recognizedOperatorType != RecognizedOperatorType::None && "Unrecognized EinSum operators should have fallen back to CPU");

        std::vector<std::optional<uint32_t>> inputIndices = {0,1,2};
        std::vector<std::optional<uint32_t>> outputIndices = {0};
        uint32_t bindableInputCount = kernelCreationContext.GetInputCount();
        if (IsMatMulOperatorType())
        {
            ++bindableInputCount;  // Account for the optional C tensor.
        }
        inputIndices.resize(bindableInputCount);

        uint32_t minimumDimensionCount = 1;
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, minimumDimensionCount);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        static_assert(RecognizedOperatorType::Total == static_cast<RecognizedOperatorType>(6), "Update this switch statement.");
        switch (m_recognizedOperatorType)
        {
        case RecognizedOperatorType::Multiply:
            {
                ReprojectTensorDescsToProductTensor();

                DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC operatorDesc = {};
                operatorDesc.ATensor = &inputDescs[0];
                operatorDesc.BTensor = &inputDescs[1];
                operatorDesc.OutputTensor = outputDescs.data();

                SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &operatorDesc}, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::MatMul:
            {
                assert(m_components.size() == 3 && "EinSum matmul expects 2 inputs and 1 output");
                assert(m_productDimensions.size() - 1 <= 4 && "DML Einsum matmul handles up to 4D");

                // Generate bitmasks for each of the active axes per tensor using their labels.
                const auto input0Labels = m_components[0].GetLabels(m_labelIndices);
                const auto input1Labels = m_components[1].GetLabels(m_labelIndices);
                const auto outputLabels = m_components[2].GetLabels(m_labelIndices);
                const uint32_t input0AxesMask = GetBitMaskFromIndices(input0Labels);
                const uint32_t input1AxesMask = GetBitMaskFromIndices(input1Labels);
                const uint32_t outputAxesMask = GetBitMaskFromIndices(outputLabels);

                // Find each of the interesting axes, including the one being reduced, height, width, batch, and channel.
                // - the reduced axis is the term missing from the output.
                // - height and width are the unique axes respectively found in only input A or input B.
                // - the batch (if present) is the first axis shared by both inputs, and the channel is the subsequent common one.
                // If any axis is not found (say it's a 2D GEMM), then the axis value will be beyond the rank, which is
                // safely handled correctly during projection as an inserted axis.

                auto findAndClearAxis = [](uint32_t& currentAxesMask, uint32_t contraintAxesMask) -> uint32_t
                {
                    uint32_t foundAxis = CountLeastSignificantZeros(currentAxesMask & ~contraintAxesMask);
                    currentAxesMask &= ~(1 << foundAxis);
                    return foundAxis;
                };

                uint32_t remainingAxesMask = ~0u;
                uint32_t reductionAxis     = findAndClearAxis(/*inout*/ remainingAxesMask, outputAxesMask);
                uint32_t heightAxis        = findAndClearAxis(/*inout*/ remainingAxesMask, input1AxesMask);
                uint32_t widthAxis         = findAndClearAxis(/*inout*/ remainingAxesMask, input0AxesMask);
                uint32_t batchAxis         = findAndClearAxis(/*inout*/ remainingAxesMask, 0);
                uint32_t channelAxis       = findAndClearAxis(/*inout*/ remainingAxesMask, 0);

                // Reproject all inputs and the output to the needed order pattern for DML compatibility,
                // which only accepts the rightmost axis as GEMM-reducible when TransB is true.
                ReprojectTensorDescToGivenAxes(/*inout*/ m_inputTensorDescs[0],  input0Labels, {{batchAxis, channelAxis, heightAxis, reductionAxis}});
                ReprojectTensorDescToGivenAxes(/*inout*/ m_inputTensorDescs[1],  input1Labels, {{batchAxis, channelAxis, widthAxis, reductionAxis}});
                ReprojectTensorDescToGivenAxes(/*inout*/ m_outputTensorDescs[0], outputLabels, {{batchAxis, channelAxis, heightAxis, widthAxis}});

                DML_GEMM_OPERATOR_DESC operatorDesc = {};
                operatorDesc.ATensor = &inputDescs[0];
                operatorDesc.BTensor = &inputDescs[1];
                // No operatorDesc.CTensor
                operatorDesc.OutputTensor = &outputDescs[0];
                operatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
                operatorDesc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
                operatorDesc.Alpha = 1.0;
                operatorDesc.Beta = 0.0;
                operatorDesc.FusedActivation = nullptr;

                SetDmlOperatorDesc({ DML_OPERATOR_GEMM, &operatorDesc }, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::ReduceSum:
            {
                ReprojectTensorDescsToProductTensor();

                DML_REDUCE_OPERATOR_DESC operatorDesc = {};
                std::vector<uint32_t> reducedAxes = GetReductionAxes();
                operatorDesc.InputTensor = inputDescs.data();
                operatorDesc.OutputTensor = outputDescs.data();
                operatorDesc.Function = DML_REDUCE_FUNCTION_SUM;
                operatorDesc.Axes = reducedAxes.data();
                operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(reducedAxes.size());

                SetDmlOperatorDesc({ DML_OPERATOR_REDUCE, &operatorDesc }, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::Transpose:
            {
                ReprojectTensorDescsToProductTensor();

                DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operatorDesc = {};
                operatorDesc.InputTensor = inputDescs.data();
                operatorDesc.OutputTensor = outputDescs.data();

                SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operatorDesc}, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::MultiplyReduceSum:
            {
                // DML has no generic DML_OPERATOR_DOT_PRODUCT. So construct one via a graph of mul+sumReduce.

                ReprojectTensorDescsToProductTensor();
                TensorDesc productTensorDesc(m_outputTensorDescs.front().GetDmlDataType(), m_productDimensions);
                auto dmlProductTensorDesc = productTensorDesc.GetDmlDesc();

                DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiplyOperatorDesc = {};
                multiplyOperatorDesc.ATensor = &inputDescs[0];
                multiplyOperatorDesc.BTensor = &inputDescs[1];
                multiplyOperatorDesc.OutputTensor = &dmlProductTensorDesc;
                DML_OPERATOR_DESC multiplyOperatorDescWithEnum = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &multiplyOperatorDesc };

                DML_REDUCE_OPERATOR_DESC reduceSumOperatorDesc = {};
                std::vector<uint32_t> reducedAxes = GetReductionAxes();
                reduceSumOperatorDesc.Function = DML_REDUCE_FUNCTION_SUM;
                reduceSumOperatorDesc.InputTensor = &dmlProductTensorDesc;
                reduceSumOperatorDesc.OutputTensor = &outputDescs[0];
                reduceSumOperatorDesc.Axes = reducedAxes.data();
                reduceSumOperatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(reducedAxes.size());
                DML_OPERATOR_DESC reduceSumOperatorDescWithEnum = { DML_OPERATOR_REDUCE, &reduceSumOperatorDesc };

                enum NodeIndex
                {
                    NodeIndexMultiply,
                    NodeIndexReduceSum,
                    NodeIndexTotal,
                };

                const DML_OPERATOR_DESC* operatorDescPointers[2] =
                {
                    &multiplyOperatorDescWithEnum,   // NodeIndexMultiply
                    &reduceSumOperatorDescWithEnum,  // NodeIndexReduceSum
                };

                DML_INPUT_GRAPH_EDGE_DESC inputEdges[2];
                DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdges[1];
                DML_OUTPUT_GRAPH_EDGE_DESC outputEdges[1];

                DML_INPUT_GRAPH_EDGE_DESC& input0ToMultiplyEdge = inputEdges[0];
                input0ToMultiplyEdge.GraphInputIndex = 0;
                input0ToMultiplyEdge.ToNodeIndex = NodeIndexMultiply;
                input0ToMultiplyEdge.ToNodeInputIndex = 0;

                DML_INPUT_GRAPH_EDGE_DESC& input1ToMultiplyEdge = inputEdges[1];
                input1ToMultiplyEdge.GraphInputIndex = 1;
                input1ToMultiplyEdge.ToNodeIndex = NodeIndexMultiply;
                input1ToMultiplyEdge.ToNodeInputIndex = 1;

                DML_INTERMEDIATE_GRAPH_EDGE_DESC& multiplyToReduceSumEdge = intermediateEdges[0];
                multiplyToReduceSumEdge.FromNodeIndex = NodeIndexMultiply;
                multiplyToReduceSumEdge.FromNodeOutputIndex = 0;
                multiplyToReduceSumEdge.ToNodeIndex = NodeIndexReduceSum;
                multiplyToReduceSumEdge.ToNodeInputIndex = 0;

                DML_OUTPUT_GRAPH_EDGE_DESC& reduceSumToOutputEdge = outputEdges[0];
                reduceSumToOutputEdge.FromNodeIndex = NodeIndexReduceSum;
                reduceSumToOutputEdge.FromNodeOutputIndex = 0;
                reduceSumToOutputEdge.GraphOutputIndex = 0;

                MLOperatorGraphDesc operatorGraphDesc = {};
                operatorGraphDesc.inputEdgeCount = uint32_t(std::size(inputEdges));
                operatorGraphDesc.inputEdges = std::data(inputEdges);
                operatorGraphDesc.intermediateEdgeCount = uint32_t(std::size(intermediateEdges));
                operatorGraphDesc.intermediateEdges = std::data(intermediateEdges);
                operatorGraphDesc.outputEdgeCount = uint32_t(std::size(outputEdges));
                operatorGraphDesc.outputEdges = std::data(outputEdges);
                operatorGraphDesc.nodeCount = uint32_t(std::size(operatorDescPointers));
                operatorGraphDesc.nodes = std::data(operatorDescPointers);
                SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
            }
            break;

        default:
            return;
        }
    }

    // Reproject all inputs and the output to the intermediate product tensor.
    // e.g.
    //
    //      Equation: i,j->ji
    //
    //      [1] [4,5,6,7]     [4, 8,12]
    //      [2]           ->  [5,10,15]
    //      [3]               [6,12,18]
    //                        [7,14,21]
    //
    //      Expand inputs 0 and 1 to 2D via strides to be directly broadcast-compatible.
    //
    //      [1,1,1,1] [4,5,6,7]    [4, 8,12]
    //      [2,2,2,2] [4,5,6,7] -> [5,10,15]
    //      [3,3,3,3] [4,5,6,7]    [6,12,18]
    //                             [7,14,21]
    //
    //      Transpose the output to be shape-compatible:
    //
    //      [1,1,1,1] [4,5,6,7]    [ 4, 5, 6, 7]
    //      [2,2,2,2] [4,5,6,7] -> [ 8,10,12,14]
    //      [3,3,3,3] [4,5,6,7]    [12,15,18,21]
    //
    void ReprojectTensorDescsToProductTensor()
    {
        assert(!m_components.empty() && "Equation components should have already been parsed.");
        assert(m_inputTensorDescs.size() + m_outputTensorDescs.size() == m_components.size());

        for (size_t i = 0, count = m_inputTensorDescs.size(); i < count; ++i)
        {
            auto inputLabels = m_components[i].GetLabels(m_labelIndices);
            ReprojectTensorDescToProductTensor(/*inout*/ m_inputTensorDescs[i], inputLabels, /*isReduced*/ false);
        }
        auto outputLabels = m_components.back().GetLabels(m_labelIndices);
        ReprojectTensorDescToProductTensor(/*inout*/ m_outputTensorDescs.front(), outputLabels, /*isReduced*/ true);
    }

    // Project the given tensor for shape compatibility to the internal product tensor, which may include broadcasting,
    // transposition, and collapsing repeated terms (e.g. iji,i->j with 2 i's in the first term with strides summed).
    //
    // e.g.
    //
    //      Axis labels:             3,0,2          // the 2 in the inputShape[0] corresponds to productDimensions[3].
    //      Original tensor shape:   [2,3,4]
    //      Original tensor strides: [12,4,1]       // packed strides right-to-left
    //      Product tensor shape:    [3,5,4,2]      // transposed relative to input, with 1 more axis not in input tensor
    //      Reprojected shape:       [3,5,4,2]      // identical to product shape
    //          (or when isReduced)  [3,1,4,2]      // inserted dimension is 1
    //      Reprojected strides:     [4,0,1,12]     // the newly inserted tensor has 0 stride for broadcasting
    //
    void ReprojectTensorDescToProductTensor(
        /*inout*/ TensorDesc& tensorDesc,
        gsl::span<const uint32_t> axisLabels,
        bool isReduced // Return 1's for any missing dimensions not in axisLabels.
    )
    {
        assert(m_productDimensions.size() == m_uniqueLabelCount && "Product dimensions were not computed yet");
        const size_t newRank = m_productDimensions.size();

        // Compute the default strides of the tensor (non-transposed).
        tensorDesc.EnsureStridesExist();
        const auto originalSizes = tensorDesc.GetSizes();
        const auto originalStrides = tensorDesc.GetStrides();
        assert(originalSizes.size() >= axisLabels.size());
        assert(originalStrides.size() >= axisLabels.size());

        // Set default sizes for shape compatibility with the product tensor, and
        // set strides to 0's initially to broadcast any missing dimensions.
        std::vector<uint32_t> newSizes;
        std::vector<uint32_t> newStrides(newRank, 0u);  // Default to 0 to broadcast missing entries.
        if (isReduced)
        {
            newSizes.resize(newRank, 1u);  // Fill with 1's initially for any missing (reduced) dimensions.
        }
        else
        {
            newSizes = m_productDimensions;  // Use the product tensor shape directly. Missing axes will be broadcasted.
        }

        // Scatter the original sizes and strides into the corresponding product tensor axis.
        for (size_t i = 0, count = axisLabels.size(); i < count; ++i)
        {
            uint32_t productAxis = axisLabels[i];
            if (productAxis < newRank)
            {
                newSizes[productAxis] = originalSizes[i];
                newStrides[productAxis] += originalStrides[i];  // Add to combine diagonal cases like i,j,i->i,j
            }
        }
        tensorDesc.SetDimensionsAndStrides(newSizes, newStrides);
        tensorDesc.EnsureDimensionCount(1, TensorAxis::RightAligned);
    }

    // Reproject a tensor to the given axis arrangement.
    // The new tensor will have rank == newAxes.size().
    // e.g.
    //
    //      product tensor shape = [2,3,4,5,6] // m_productDimensions
    //      newAxes              = [4,2,0,1]
    //      new tensor shape     = [6,4,2,3]
    //
    void ReprojectTensorDescToGivenAxes(
        /*inout*/ TensorDesc& tensorDesc,
        gsl::span<const uint32_t> axisLabels,
        gsl::span<const uint32_t> newAxes
    )
    {
        // First, reproject the original dimensions up to the product tensor.
        ReprojectTensorDescToProductTensor(/*inout*/ tensorDesc, axisLabels, /*isReduced*/ false);
        tensorDesc.PermuteDimensions(newAxes, TensorAxis::LeftAligned);
    }

    std::vector<uint32_t> GetReductionAxes() const
    {
        // Determine which axes are reduced by looking for any output dimensions of size 1.
        // Note this could include dimensions that are not actually being reduced and simply
        // already had size 1 from the input, but such cases harmless nops either way.

        auto outputSizes = m_outputTensorDescs.front().GetSizes();
        std::vector<uint32_t> reducedAxes;
        FindValueIndices<uint32_t>(outputSizes, 1u, /*out*/ reducedAxes);
        return reducedAxes;
    }
};

void CALLBACK QueryEinSum(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = false;

    MLOperatorAttributes attributes(context);
    EinSumHelper helper(attributes);
    auto recognizedOperatorType = helper.GetRecognizedOperatorType();

    static_assert(EinSumHelper::RecognizedOperatorType::Total == static_cast<EinSumHelper::RecognizedOperatorType>(6), "Verify if this function needs updating.");
    *isSupported = (recognizedOperatorType != EinSumHelper::RecognizedOperatorType::None);
}

DML_OP_DEFINE_CREATION_FUNCTION(Einsum12, VersionedKernel<DmlOperatorEinSum, 12>);

} // namespace Dml
