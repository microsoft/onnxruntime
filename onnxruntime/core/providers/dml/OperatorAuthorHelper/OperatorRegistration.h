// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace OperatorHelper
{
    // The since version of each operator in version 7 of the ONNX domain
    namespace OnnxOperatorSet7
    {
        static const int sc_sinceVer_Affine = 1;
        static const int sc_sinceVer_ArgMax = 1;
        static const int sc_sinceVer_ArgMin = 1;
        static const int sc_sinceVer_ATen = 1;
        static const int sc_sinceVer_Constant = 1;
        static const int sc_sinceVer_ConstantFill = 1;
        static const int sc_sinceVer_Conv = 1;
        static const int sc_sinceVer_ConvTranspose = 1;
        static const int sc_sinceVer_Crop = 1;
        static const int sc_sinceVer_DepthToSpace = 1;
        static const int sc_sinceVer_Flatten = 1;
        static const int sc_sinceVer_Gather = 1;
        static const int sc_sinceVer_GivenTensorFill = 1;
        static const int sc_sinceVer_GlobalAveragePool = 1;
        static const int sc_sinceVer_GlobalMaxPool = 1;
        static const int sc_sinceVer_GRUUnit = 1;
        static const int sc_sinceVer_Hardmax = 1;
        static const int sc_sinceVer_Identity = 1;
        static const int sc_sinceVer_If = 1;
        static const int sc_sinceVer_ImageScaler = 1;
        static const int sc_sinceVer_LogSoftmax = 1;
        static const int sc_sinceVer_Loop = 1;
        static const int sc_sinceVer_LoopIndexTensor = 1;
        static const int sc_sinceVer_LpNormalization = 1;
        static const int sc_sinceVer_LRN = 1;
        static const int sc_sinceVer_MatMul = 1;
        static const int sc_sinceVer_MaxPool = 1;
        static const int sc_sinceVer_MaxRoiPool = 1;
        static const int sc_sinceVer_MeanVarianceNormalization = 1;
        static const int sc_sinceVer_Not = 1;
        static const int sc_sinceVer_ParametricSoftplus = 1;
        static const int sc_sinceVer_RandomNormal = 1;
        static const int sc_sinceVer_RandomNormalLike = 1;
        static const int sc_sinceVer_RandomUniform = 1;
        static const int sc_sinceVer_RandomUniformLike = 1;
        static const int sc_sinceVer_ReduceL1 = 1;
        static const int sc_sinceVer_ReduceL2 = 1;
        static const int sc_sinceVer_ReduceLogSum = 1;
        static const int sc_sinceVer_ReduceLogSumExp = 1;
        static const int sc_sinceVer_ReduceMax = 1;
        static const int sc_sinceVer_ReduceMean = 1;
        static const int sc_sinceVer_ReduceMin = 1;
        static const int sc_sinceVer_ReduceProd = 1;
        static const int sc_sinceVer_ReduceSum = 1;
        static const int sc_sinceVer_ReduceSumSquare = 1;
        static const int sc_sinceVer_Scale = 1;
        static const int sc_sinceVer_ScaledTanh = 1;
        static const int sc_sinceVer_Shape = 1;
        static const int sc_sinceVer_Size = 1;
        static const int sc_sinceVer_Slice = 1;
        static const int sc_sinceVer_Softmax = 1;
        static const int sc_sinceVer_Softplus = 1;
        static const int sc_sinceVer_Softsign = 1;
        static const int sc_sinceVer_SpaceToDepth = 1;
        static const int sc_sinceVer_Squeeze = 1;
        static const int sc_sinceVer_ThresholdedRelu = 1;
        static const int sc_sinceVer_TopK = 1;
        static const int sc_sinceVer_Transpose = 1;
        static const int sc_sinceVer_Unsqueeze = 1;
        static const int sc_sinceVer_GlobalLpPool = 2;
        static const int sc_sinceVer_LpPool = 2;
        static const int sc_sinceVer_Pad = 2;
        static const int sc_sinceVer_Split = 2;
        static const int sc_sinceVer_Concat = 4;
        static const int sc_sinceVer_Reshape = 5;
        static const int sc_sinceVer_Abs = 6;
        static const int sc_sinceVer_Cast = 6;
        static const int sc_sinceVer_Ceil = 6;
        static const int sc_sinceVer_Clip = 6;
        static const int sc_sinceVer_Elu = 6;
        static const int sc_sinceVer_Exp = 6;
        static const int sc_sinceVer_Floor = 6;
        static const int sc_sinceVer_HardSigmoid = 6;
        static const int sc_sinceVer_InstanceNormalization = 6;
        static const int sc_sinceVer_LeakyRelu = 6;
        static const int sc_sinceVer_Log = 6;
        static const int sc_sinceVer_Max = 6;
        static const int sc_sinceVer_Mean = 6;
        static const int sc_sinceVer_Min = 6;
        static const int sc_sinceVer_Neg = 6;
        static const int sc_sinceVer_Reciprocal = 6;
        static const int sc_sinceVer_Relu = 6;
        static const int sc_sinceVer_Selu = 6;
        static const int sc_sinceVer_Sigmoid = 6;
        static const int sc_sinceVer_Sqrt = 6;
        static const int sc_sinceVer_Sum = 6;
        static const int sc_sinceVer_Tanh = 6;
        static const int sc_sinceVer_Tile = 6;
        static const int sc_sinceVer_Acos = 7;
        static const int sc_sinceVer_Add = 7;
        static const int sc_sinceVer_And = 7;
        static const int sc_sinceVer_Asin = 7;
        static const int sc_sinceVer_Atan = 7;
        static const int sc_sinceVer_AveragePool = 7;
        static const int sc_sinceVer_BatchNormalization = 7;
        static const int sc_sinceVer_Cos = 7;
        static const int sc_sinceVer_Div = 7;
        static const int sc_sinceVer_Dropout = 7;
        static const int sc_sinceVer_Equal = 7;
        static const int sc_sinceVer_Gemm = 7;
        static const int sc_sinceVer_Greater = 7;
        static const int sc_sinceVer_GRU = 7;
        static const int sc_sinceVer_Less = 7;
        static const int sc_sinceVer_LSTM = 7;
        static const int sc_sinceVer_Mul = 7;
        static const int sc_sinceVer_Multinomial = 7;
        static const int sc_sinceVer_Or = 7;
        static const int sc_sinceVer_Pow = 7;
        static const int sc_sinceVer_PRelu = 7;
        static const int sc_sinceVer_RNN = 7;
        static const int sc_sinceVer_Sin = 7;
        static const int sc_sinceVer_Sub = 7;
        static const int sc_sinceVer_Tan = 7;
        static const int sc_sinceVer_Upsample = 7;
        static const int sc_sinceVer_Xor = 7;
    
        // Special operators
        static const int sc_sinceVer_MemcpyToHost = 1;
        static const int sc_sinceVer_MemcpyFromHost = 1;
    } // namespace OnnxOperatorSet7

    namespace OnnxOperatorSet8
    {
        static const int sc_sinceVer_Expand = 8;
        static const int sc_sinceVer_Max = 8;
        static const int sc_sinceVer_Mean = 8;
        static const int sc_sinceVer_Min = 8;
        static const int sc_sinceVer_Sum = 8;
        static const int sc_sinceVer_MaxPool = 8;
    } // namespace OnnxOperatorSet8

    namespace OnnxOperatorSet9
    {
        static const int sc_sinceVer_Sign = 9;
        static const int sc_sinceVer_IsNan = 9;
        static const int sc_sinceVer_Sinh = 9;
        static const int sc_sinceVer_Cosh = 9;
        static const int sc_sinceVer_Asinh = 9;
        static const int sc_sinceVer_Acosh = 9;
        static const int sc_sinceVer_Atanh = 9;
        static const int sc_sinceVer_Erf = 9;
        static const int sc_sinceVer_Where = 9;
        static const int sc_sinceVer_ConstantOfShape = 9;
        static const int sc_sinceVer_OneHot = 9;
        static const int sc_sinceVer_MaxUnpool = 9;
        static const int sc_sinceVer_Compress = 9;
        static const int sc_sinceVer_EyeLike = 9;
        static const int sc_sinceVer_Scatter = 9;
        static const int sc_sinceVer_Nonzero = 9;
        static const int sc_sinceVer_Shrink = 9;
        static const int sc_sinceVer_Greater = 9;
        static const int sc_sinceVer_Less = 9;
        static const int sc_sinceVer_BatchNormalization = 9;
        static const int sc_sinceVer_Flatten = 9;
        static const int sc_sinceVer_Gemm = 9;
        static const int sc_sinceVer_PRelu = 9;
        static const int sc_sinceVer_MatMul = 9;
        static const int sc_sinceVer_Cast = 9;
        static const int sc_sinceVer_Upsample = 9;
        static const int sc_sinceVer_MeanVarianceNormalization = 9;
    } // namespace OnnxOperatorSet9

    namespace OnnxOperatorSet10
    {
        static const int sc_sinceVer_Crop = 10; // Removed in this version.
        static const int sc_sinceVer_Resize = 10;
        static const int sc_sinceVer_MaxPool = 10;
        static const int sc_sinceVer_QuantizeLinear = 10;
        static const int sc_sinceVer_DequantizeLinear = 10;
        static const int sc_sinceVer_Dropout = 10;
        static const int sc_sinceVer_ThresholdedRelu = 10;
        static const int sc_sinceVer_Upsample = 10;
        static const int sc_sinceVer_Slice = 10;
        static const int sc_sinceVer_IsInf = 10;
        static const int sc_sinceVer_Mod = 10;
        static const int sc_sinceVer_DropOut = 10;
        static const int sc_sinceVer_RoiAlign = 10;
        static const int sc_sinceVer_TopK = 10;
        static const int sc_sinceVer_ReverseSequence = 10;
        static const int sc_sinceVer_AveragePool = 10;
        static const int sc_sinceVer_ConvInteger = 10;
        static const int sc_sinceVer_MatMulInteger = 10;
        static const int sc_sinceVer_QLinearConv = 10;
        static const int sc_sinceVer_QLinearMatMul = 10;
    } // namespace OnnxOperatorSet10

    namespace OnnxOperatorSet11
    {
        static const int sc_sinceVer_ArgMax = 11;
        static const int sc_sinceVer_ArgMin = 11;
        static const int sc_sinceVer_AveragePool = 11;
        static const int sc_sinceVer_BitShift = 11;
        static const int sc_sinceVer_Clip = 11;
        static const int sc_sinceVer_Compress = 11;
        static const int sc_sinceVer_Concat = 11;
        static const int sc_sinceVer_Conv = 11;
        static const int sc_sinceVer_ConvTranspose = 11;
        static const int sc_sinceVer_CumSum = 11;
        static const int sc_sinceVer_DepthToSpace = 11;
        static const int sc_sinceVer_Equal = 11;
        static const int sc_sinceVer_Flatten = 11;
        static const int sc_sinceVer_Gather = 11;
        static const int sc_sinceVer_GatherElements = 11;
        static const int sc_sinceVer_GatherND = 11;
        static const int sc_sinceVer_Gemm = 11;
        static const int sc_sinceVer_Hardmax = 11;
        static const int sc_sinceVer_LogSoftmax = 11;
        static const int sc_sinceVer_LpPool = 11;
        static const int sc_sinceVer_MaxPool = 11;
        static const int sc_sinceVer_MaxUnpool = 11;
        static const int sc_sinceVer_OneHot = 11;
        static const int sc_sinceVer_Pad = 11;
        static const int sc_sinceVer_Range = 11;
        static const int sc_sinceVer_ReduceL1 = 11;
        static const int sc_sinceVer_ReduceL2 = 11;
        static const int sc_sinceVer_ReduceLogSum = 11;
        static const int sc_sinceVer_ReduceLogSumExp = 11;
        static const int sc_sinceVer_ReduceMax = 11;
        static const int sc_sinceVer_ReduceMean = 11;
        static const int sc_sinceVer_ReduceMin = 11;
        static const int sc_sinceVer_ReduceProd = 11;
        static const int sc_sinceVer_ReduceSum = 11;
        static const int sc_sinceVer_ReduceSumSquare = 11;
        static const int sc_sinceVer_Resize = 11;
        static const int sc_sinceVer_Round = 11;
        static const int sc_sinceVer_Scan = 11;
        static const int sc_sinceVer_Scatter = 11; // Deprecated alias
        static const int sc_sinceVer_ScatterElements = 11;
        static const int sc_sinceVer_ScatterND = 11;
        static const int sc_sinceVer_Slice = 11;
        static const int sc_sinceVer_Softmax = 11;
        static const int sc_sinceVer_Split = 11;
        static const int sc_sinceVer_Squeeze = 11;
        static const int sc_sinceVer_TopK = 11;
        static const int sc_sinceVer_Unsqueeze = 11;
    } // namespace OnnxOperatorSet11

    namespace MsftOperatorSet1
    {
        static const int sc_sinceVer_FusedConv = 1;
        static const int sc_sinceVer_FusedConvTranspose = 1;
        static const int sc_sinceVer_FusedInstanceNormalization = 1;
        static const int sc_sinceVer_FusedBatchNormalization = 1;
        static const int sc_sinceVer_FusedMeanVarianceNormalization = 1;
        static const int sc_sinceVer_FusedGemm = 1;
        static const int sc_sinceVer_FusedMatMul = 1;
        static const int sc_sinceVer_FusedAdd = 1;
        static const int sc_sinceVer_FusedSum = 1;
        static const int sc_sinceVer_QuantizeLinear = 1;
        static const int sc_sinceVer_DequantizeLinear = 1;
        static const int sc_sinceVer_ConvTransposeWithDynamicPads = 1;
    } // namespace MsftOperatorSet1

} // namespace OperatorHelper
