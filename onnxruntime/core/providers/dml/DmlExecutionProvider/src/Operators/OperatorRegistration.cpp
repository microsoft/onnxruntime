// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "OperatorRegistration.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorRegistration.h"
#include "core/providers/dml/DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "core/framework/customregistry.h"
#include "onnx/defs/operator_sets.h"
#include <wrl/client.h>
#include <wrl/implements.h>
#include <mutex>
using namespace Microsoft::WRL;

namespace Dml
{

enum class SupportedTensorDataTypes : uint32_t
{
    Undefined = 1<<0,
    Float32 = 1<<1,
    UInt8 = 1<<2,
    Int8 = 1<<3,
    UInt16 = 1<<4,
    Int16 = 1<<5,
    Int32 = 1<<6,
    Int64 = 1<<7,
    String = 1<<8,
    Bool = 1<<9,
    Float16 = 1<<10,
    Float64 = 1<<11,
    UInt32 = 1<<12,
    UInt64 = 1<<13,
    Complex64 = 1<<14,
    Complex128 = 1<<15,
    Int8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32,
    Int32to64 = UInt32|Int32|UInt64|Int64,
    Float16to32 = Float16|Float32, // Float64 is not supported by DirectML.
    NumericDefault = Int8to32|Float16to32,
    Scalars8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32|Float16to32|Bool, 
    AllScalars = UInt8|Int8|UInt16|Int16|UInt32|Int32|UInt64|Int64|Float16to32|Bool,
    All = static_cast<uint32_t>(-1),
};
DEFINE_ENUM_FLAG_OPERATORS(Dml::SupportedTensorDataTypes);

enum class DmGraphSupport
{
    Supported   = 0,
    NotSupported = 1,
};

struct OperatorRegistrationInformation
{
    const char* operatorName;
    int sinceVersion;
    const char* domain;
    MLOperatorKernelCreateFn creationFunction;
    MLOperatorShapeInferenceFunction shapeInferenceFunction;
    bool canAliasFirstInput;
    bool requiresFloatFormatsForGraph = false;

    gsl::span<char const* const> tensorTypeNames;
    gsl::span<const SupportedTensorDataTypes> supportedTensorDataTypes;
    DmGraphSupport DmGraphSupport;

    std::vector<uint32_t> requiresFloatFormatsExceptConstInputs;

    // For use by operators such as Sum, which may require multiple calls to DML, in which case they
    // can't be represented as nodes in an optimized graph yet.
    std::optional<uint32_t> requiredInputCountForDmlGraphSupport;

    MLOperatorSupportQueryFunction supportQueryFunction;
};

DML_OP_EXTERN_CREATION_FUNCTION(Copy);
DML_OP_EXTERN_CREATION_FUNCTION(FC);
DML_OP_EXTERN_CREATION_FUNCTION(Conv);
DML_OP_EXTERN_CREATION_FUNCTION(ConvTranspose);
DML_OP_EXTERN_CREATION_FUNCTION(ConvTransposeWithDynamicPads);
DML_OP_EXTERN_CREATION_FUNCTION(AveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalAveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalMaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(LpPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalLpPool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxRoiPool);
DML_OP_EXTERN_CREATION_FUNCTION(InstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BatchNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(LRN);
DML_OP_EXTERN_CREATION_FUNCTION(MeanVarianceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(LpNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(RNN);
DML_OP_EXTERN_CREATION_FUNCTION(GRU);
DML_OP_EXTERN_CREATION_FUNCTION(LSTM);
DML_OP_EXTERN_CREATION_FUNCTION(Gather);
DML_OP_EXTERN_CREATION_FUNCTION(Flatten);
DML_OP_EXTERN_CREATION_FUNCTION(Split);
DML_OP_EXTERN_CREATION_FUNCTION(Transpose);
DML_OP_EXTERN_CREATION_FUNCTION(Tile);
DML_OP_EXTERN_CREATION_FUNCTION(Concat);
DML_OP_EXTERN_CREATION_FUNCTION(Slice7);
DML_OP_EXTERN_CREATION_FUNCTION(Slice10);
DML_OP_EXTERN_CREATION_FUNCTION(Pad);
DML_OP_EXTERN_CREATION_FUNCTION(SpaceToDepth);
DML_OP_EXTERN_CREATION_FUNCTION(DepthToSpace);
DML_OP_EXTERN_CREATION_FUNCTION(Sqrt);
DML_OP_EXTERN_CREATION_FUNCTION(Reciprocal);
DML_OP_EXTERN_CREATION_FUNCTION(Pow);
DML_OP_EXTERN_CREATION_FUNCTION(Exp);
DML_OP_EXTERN_CREATION_FUNCTION(Log);
DML_OP_EXTERN_CREATION_FUNCTION(Abs);
DML_OP_EXTERN_CREATION_FUNCTION(Ceil);
DML_OP_EXTERN_CREATION_FUNCTION(Floor);
DML_OP_EXTERN_CREATION_FUNCTION(Clip);
DML_OP_EXTERN_CREATION_FUNCTION(Greater);
DML_OP_EXTERN_CREATION_FUNCTION(Less);
DML_OP_EXTERN_CREATION_FUNCTION(Equal);
DML_OP_EXTERN_CREATION_FUNCTION(Not);
DML_OP_EXTERN_CREATION_FUNCTION(And);
DML_OP_EXTERN_CREATION_FUNCTION(Or);
DML_OP_EXTERN_CREATION_FUNCTION(Xor);
DML_OP_EXTERN_CREATION_FUNCTION(Add);
DML_OP_EXTERN_CREATION_FUNCTION(Sub);
DML_OP_EXTERN_CREATION_FUNCTION(Mul);
DML_OP_EXTERN_CREATION_FUNCTION(Div);
DML_OP_EXTERN_CREATION_FUNCTION(Sum);
DML_OP_EXTERN_CREATION_FUNCTION(Mean);
DML_OP_EXTERN_CREATION_FUNCTION(Max);
DML_OP_EXTERN_CREATION_FUNCTION(Min);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSum);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMean);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceProd);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSum);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSumExp);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSumSquare);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL1);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL2);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMax);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMin);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMax);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMin);
DML_OP_EXTERN_CREATION_FUNCTION(Gemm);
DML_OP_EXTERN_CREATION_FUNCTION(Neg);
DML_OP_EXTERN_CREATION_FUNCTION(Crop);
DML_OP_EXTERN_CREATION_FUNCTION(ImageScaler);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample);
DML_OP_EXTERN_CREATION_FUNCTION(Sigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(HardSigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(ScaledTanh);
DML_OP_EXTERN_CREATION_FUNCTION(Relu);
DML_OP_EXTERN_CREATION_FUNCTION(LeakyRelu);
DML_OP_EXTERN_CREATION_FUNCTION(PRelu);
DML_OP_EXTERN_CREATION_FUNCTION(ThresholdedRelu);
DML_OP_EXTERN_CREATION_FUNCTION(Elu);
DML_OP_EXTERN_CREATION_FUNCTION(Selu);
DML_OP_EXTERN_CREATION_FUNCTION(Softmax);
DML_OP_EXTERN_CREATION_FUNCTION(LogSoftmax);
DML_OP_EXTERN_CREATION_FUNCTION(Hardmax);
DML_OP_EXTERN_CREATION_FUNCTION(Softsign);
DML_OP_EXTERN_CREATION_FUNCTION(Softplus);
DML_OP_EXTERN_CREATION_FUNCTION(ParametricSoftplus);
DML_OP_EXTERN_CREATION_FUNCTION(Affine);
DML_OP_EXTERN_CREATION_FUNCTION(Dropout);
DML_OP_EXTERN_CREATION_FUNCTION(MatMul);
DML_OP_EXTERN_CREATION_FUNCTION(Cast);
DML_OP_EXTERN_CREATION_FUNCTION(MemcpyFromHost);
DML_OP_EXTERN_CREATION_FUNCTION(MemcpyToHost);
DML_OP_EXTERN_CREATION_FUNCTION(TopK);
DML_OP_EXTERN_CREATION_FUNCTION(Expand);
DML_OP_EXTERN_CREATION_FUNCTION(Cos);
DML_OP_EXTERN_CREATION_FUNCTION(Sin);
DML_OP_EXTERN_CREATION_FUNCTION(Tan);
DML_OP_EXTERN_CREATION_FUNCTION(Acos);
DML_OP_EXTERN_CREATION_FUNCTION(Asin);
DML_OP_EXTERN_CREATION_FUNCTION(Atan);
DML_OP_EXTERN_CREATION_FUNCTION(FusedConv);
DML_OP_EXTERN_CREATION_FUNCTION(FusedConvTranspose);
DML_OP_EXTERN_CREATION_FUNCTION(FusedInstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(FusedBatchNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(FusedMeanVarianceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(FusedGemm);
DML_OP_EXTERN_CREATION_FUNCTION(FusedMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(FusedAdd);
DML_OP_EXTERN_CREATION_FUNCTION(FusedSum);
DML_OP_EXTERN_CREATION_FUNCTION(QuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(DequantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(Sign);
DML_OP_EXTERN_CREATION_FUNCTION(IsNan);
DML_OP_EXTERN_CREATION_FUNCTION(Sinh);
DML_OP_EXTERN_CREATION_FUNCTION(Cosh);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(Asinh);
DML_OP_EXTERN_CREATION_FUNCTION(Acosh);
DML_OP_EXTERN_CREATION_FUNCTION(Atanh);
DML_OP_EXTERN_CREATION_FUNCTION(Erf);
DML_OP_EXTERN_CREATION_FUNCTION(Where);
DML_OP_EXTERN_CREATION_FUNCTION(Shrink);
DML_OP_EXTERN_CREATION_FUNCTION(OneHot);
DML_OP_EXTERN_CREATION_FUNCTION(EyeLike);
DML_OP_EXTERN_CREATION_FUNCTION(MaxUnpool);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter);
DML_OP_EXTERN_CREATION_FUNCTION(Resize);
DML_OP_EXTERN_CREATION_FUNCTION(ConstantOfShape);

DML_OP_EXTERN_QUERY_FUNCTION(MaxPool);
DML_OP_EXTERN_QUERY_FUNCTION(Slice);

const static char* const typeNameListDefault[1] = {"T"};
const static char* const typeNameListTopK[2] = { "T", "I" };
const static char* const typeNameListLogicalComparison[2] = { "T", "T1" };
const static char* const typeNameListCast[2] = { "T1", "T2" };
const static char* const typeNameListIsNan[2] = { "T1", "T2" };
const static char* const typeNameListConstantOfShape[2] = { "T1", "T2" };
const static char* const typeNameListScatterGather[2] = { "T", "Tind" };
const static char* const typeNameListQuantize[2] = { "T1", "T2" };
const static char* const typeNameListWhere[2] = { "B", "T" };
const static char* const typeNameListOneHot[3] = { "T1", "T2", "T3" };
const static char* const typeNameListEyeLike[1] = { "T2" };
const static SupportedTensorDataTypes supportedTypeListAll[1] = {SupportedTensorDataTypes::All};
const static SupportedTensorDataTypes supportedTypeListFloat16to32[1] = {SupportedTensorDataTypes::Float16to32};
const static SupportedTensorDataTypes supportedTypeListInt32to64AndFloat16to32[1] = {SupportedTensorDataTypes::Int32to64|SupportedTensorDataTypes::Float16to32};
const static SupportedTensorDataTypes supportedTypeListNumericDefault[1] = { SupportedTensorDataTypes::NumericDefault };
const static SupportedTensorDataTypes supportedTypeListAllScalars[1] = { SupportedTensorDataTypes::AllScalars };
const static SupportedTensorDataTypes supportedTypeListBool[1] = {SupportedTensorDataTypes::Bool};
const static SupportedTensorDataTypes supportedTypeListTopK[2] = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int64};
const static SupportedTensorDataTypes supportedTypeListIndices[1] = { SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64 };
const static SupportedTensorDataTypes supportedTypeListCast[2] = { SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::Scalars8to32 };
const static SupportedTensorDataTypes supportedTypeListScatterGather[2] = { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int64 };
const static SupportedTensorDataTypes supportedTypeListQuantize[2] = { SupportedTensorDataTypes::Float32, SupportedTensorDataTypes::UInt8 };
const static SupportedTensorDataTypes supportedTypeListIsNan[2] = { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::UInt8 };
const static SupportedTensorDataTypes supportedTypeListConstantOfShape[2] = { SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64, SupportedTensorDataTypes::Float16to32 };
const static SupportedTensorDataTypes supportedTypeListWhere[2] = { SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Float16to32 };
const static SupportedTensorDataTypes supportedTypeListOneHot[3] = /* indices, depth, values */ { SupportedTensorDataTypes::Int32to64, SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::Float16to32 };
const static SupportedTensorDataTypes supportedTypeListLogicalComparison[2] = /* A&B,C */ { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Bool };

// Define a single row of registration information.
#define REG_INFO(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, false, ##__VA_ARGS__, 

// Versioned operator
#define REG_INFO_VER(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName##version, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName##version>, false, false, ##__VA_ARGS__, 

// Identity operators use Copy, alias their first input, and require floating point formats
// for usage in the graph, besides constant inputs.  This is because they currently use 
// element-wise identity operators  in the graph for striding support, but issue actual copies 
// outside the graph.  Element-wise identity currently only supports floating point types.  
#define REG_INFO_ID(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, CreateCopy, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, true, true, ##__VA_ARGS__, 

// MS-domain operators
#define REG_INFO_MS(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, false, ##__VA_ARGS__, 

// MS-domain operators
#define REG_INFO_MSDML(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDmlDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, false, ##__VA_ARGS__, 

const static OperatorRegistrationInformation operatorRegistrationInformationTable[] =
{
///  Domain/Type, Ver,  Name,                               TypeNames,                       Types,                              Graph Support,                  Required const CPU inputs, 
///                                                                                                                                                              Input count required for graph support,
///                                                                                                                                                              Support query function

    // Deep Learning Standard Layers
    {REG_INFO(      7,  Conv,                               typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ConvTranspose,                      typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  AveragePool,                        typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalAveragePool,                  typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  MaxPool,                            typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      8,  MaxPool,                            typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported, {}, std::nullopt, QueryMaxPool)},
    {REG_INFO(      10, MaxPool,                            typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported, {}, std::nullopt, QueryMaxPool)},
      
    {REG_INFO(      7,  GlobalMaxPool,                      typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  LpPool,                             typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalLpPool,                       typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  MaxRoiPool,                         typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  InstanceNormalization,              typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  BatchNormalization,                 typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  BatchNormalization,                 typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)}, // v9 just removes 'spatial' attribute.
    {REG_INFO(      7,  LRN,                                typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  MeanVarianceNormalization,          typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  LpNormalization,                    typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  RNN,                                typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::NotSupported)},
    {REG_INFO(      7,  GRU,                                typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::NotSupported)},
    {REG_INFO(      7,  LSTM,                               typeNameListDefault,             supportedTypeListFloat16to32,       DmGraphSupport::NotSupported)},
    {REG_INFO_MS(   1,  ConvTransposeWithDynamicPads,       typeNameListDefault,            supportedTypeListFloat16to32,        DmGraphSupport::Supported,      {2})},

    // Data Reorganization Layers
    {REG_INFO(      7,  Split,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Transpose,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Concat,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_VER(  7,  Slice,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_VER(  10, Slice,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {1, 2, 3}, std::nullopt, QuerySlice)},
    {REG_INFO(      7,  Pad,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  SpaceToDepth,                       typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  DepthToSpace,                       typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Tile,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {1})},  
    {REG_INFO(      8,  Expand,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {1})},
    {REG_INFO(      9,  ConstantOfShape,                    typeNameListConstantOfShape,    supportedTypeListConstantOfShape,   DmGraphSupport::NotSupported,   {0})},
    {REG_INFO(      7,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,     DmGraphSupport::Supported)},
    {REG_INFO(      9,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,     DmGraphSupport::Supported)},
    {REG_INFO(      9,  EyeLike,                            typeNameListEyeLike,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},

    // Data reorganization that merely changes the dimensions while keeping the data identical.
    {REG_INFO_ID(   7,  Identity,                           typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported)},
    {REG_INFO_ID(   7,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported)},
    {REG_INFO_ID(   9,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported)},
    {REG_INFO_ID(   7,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported)},
    {REG_INFO_ID(   7,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported)},
    {REG_INFO_ID(   7,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,        DmGraphSupport::Supported,      {1})},

    // Elementwise
    {REG_INFO(      7,  Sqrt,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Reciprocal,                         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Pow,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Exp,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Log,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Abs,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Ceil,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Floor,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Add,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Sub,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Mul,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Div,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      8,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      7,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      8,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      7,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      8,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      7,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      8,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)},
    {REG_INFO(      7,  Cos,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Sin,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Tan,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Acos,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Asin,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Atan,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Affine,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      10, QuantizeLinear,                     typeNameListQuantize,           supportedTypeListQuantize,          DmGraphSupport::Supported)},
    {REG_INFO(      10, DequantizeLinear,                   typeNameListQuantize,           supportedTypeListQuantize,          DmGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QuantizeLinear,                     typeNameListQuantize,           supportedTypeListQuantize,          DmGraphSupport::Supported)},
    {REG_INFO_MS(   1,  DequantizeLinear,                   typeNameListQuantize,           supportedTypeListQuantize,          DmGraphSupport::Supported)},
    {REG_INFO(      9,  Sign,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  IsNan,                              typeNameListIsNan,              supportedTypeListIsNan,             DmGraphSupport::Supported)},
    {REG_INFO(      9,  Sinh,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Cosh,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Asinh,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Acosh,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Atanh,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Erf,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Where,                              typeNameListWhere,              supportedTypeListWhere,             DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ArgMax,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ArgMin,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Neg,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison, DmGraphSupport::Supported)},
    {REG_INFO(      9,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison, DmGraphSupport::Supported)},
    {REG_INFO(      7,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison, DmGraphSupport::Supported)},
    {REG_INFO(      9,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison, DmGraphSupport::Supported)},
    {REG_INFO(      7,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison, DmGraphSupport::Supported)},
    {REG_INFO(      7,  Not,                                typeNameListDefault,            supportedTypeListBool,              DmGraphSupport::Supported)},
    {REG_INFO(      7,  And,                                typeNameListDefault,            supportedTypeListBool,              DmGraphSupport::Supported)},
    {REG_INFO(      7,  Or,                                 typeNameListDefault,            supportedTypeListBool,              DmGraphSupport::Supported)},
    {REG_INFO(      7,  Xor,                                typeNameListDefault,            supportedTypeListBool,              DmGraphSupport::Supported)},

    // Imaging Operators                                
    {REG_INFO(      7,  Crop,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ImageScaler,                        typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {1})},
    {REG_INFO(     10,  Resize,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {1})},
                                                        
    // Activation Functions                             
    {REG_INFO(      7,  Sigmoid,                            typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  HardSigmoid,                        typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Tanh,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ScaledTanh,                         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Relu,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  LeakyRelu,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(     10,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Elu,                                typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Selu,                               typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Softsign,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Softplus,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  ParametricSoftplus,                 typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Dropout,                            typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  Shrink,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported) },
                                                        
    // Uncategorized                                    
    {REG_INFO(      7,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      9,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO(      7,  Cast,                               typeNameListCast,               supportedTypeListCast,              DmGraphSupport::Supported)},
    {REG_INFO(      9,  Cast,                               typeNameListCast,               supportedTypeListCast,              DmGraphSupport::Supported)},
    {REG_INFO(      7,  MemcpyFromHost,                     typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO(      7,  MemcpyToHost,                       typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO(      7,  TopK,                               typeNameListTopK,               supportedTypeListTopK,              DmGraphSupport::Supported)},
    {REG_INFO(      9,  OneHot,                             typeNameListOneHot,             supportedTypeListOneHot,            DmGraphSupport::Supported,      {1})},
                                                        
    // Fused operators                                  
    {REG_INFO_MSDML(1,  FusedConv,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedConvTranspose,                 typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedInstanceNormalization,         typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedBatchNormalization,            typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedMeanVarianceNormalization,     typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedGemm,                          typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedMatMul,                        typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedAdd,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported)}, 
    {REG_INFO_MSDML(1,  FusedSum,                           typeNameListDefault,            supportedTypeListFloat16to32,       DmGraphSupport::Supported,      {}, 2)}, 
                
    // TODO: DwayneR implement MaxUnpool https://dev.azure.com/microsoft/OS/_workitems/edit/21267466
};
 
template<typename T> 
MLOperatorEdgeDescription EdgeDesc()
{
    return {MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLTypeTraits<T>::TensorType)};
}

void RegisterDmlOperators(IMLOperatorRegistry* registry)
{
    ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
    THROW_IF_FAILED(registry->QueryInterface(registryPrivate.GetAddressOf()));

    std::vector<MLOperatorEdgeTypeConstrant> typeConstraints;
    std::vector<MLOperatorEdgeDescription> edgeDescs;

    for (const OperatorRegistrationInformation& information : operatorRegistrationInformationTable)
    {
        assert(information.tensorTypeNames.size() == information.supportedTensorDataTypes.size());
            
        MLOperatorKernelDescription desc = {};
        desc.domain = information.domain;
        desc.name = information.operatorName;
        desc.executionType = MLOperatorExecutionType::D3D12; 

        // The graph must be configured with operators from only the legacy DML API, or only the new DML API
        bool kernelSupportsGraph = (information.DmGraphSupport == DmGraphSupport::Supported);

        desc.options = information.shapeInferenceFunction ? 
            MLOperatorKernelOptions::None : MLOperatorKernelOptions::AllowDynamicInputShapes;

        desc.minimumOperatorSetVersion = information.sinceVersion;
    
        typeConstraints.resize(information.tensorTypeNames.size());
        desc.typeConstraints = typeConstraints.data();
        desc.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        // edgeDescs will accumulate the edge descriptions across all type constraints.  
        // The values of allowedTypeCount will indicate how many elements of edgeDescs
        // belong to each type constraint.
        edgeDescs.clear();
        size_t lastEdgeDescSize = 0;

        // Append all supported tensor data types for each tensor type.
        for (gsl::index i = 0, ci = information.tensorTypeNames.size(); i < ci; ++i)
        {
            typeConstraints[i].typeLabel = information.tensorTypeNames[i];

            std::vector<MLOperatorTensorDataType> supportedTypeList;
            SupportedTensorDataTypes supportedTypes = information.supportedTensorDataTypes[i];

            if (bool(supportedTypes & SupportedTensorDataTypes::Float32)) edgeDescs.push_back(EdgeDesc<float>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt8  )) edgeDescs.push_back(EdgeDesc<uint8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int8   )) edgeDescs.push_back(EdgeDesc<int8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt16 )) edgeDescs.push_back(EdgeDesc<uint16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int16  )) edgeDescs.push_back(EdgeDesc<int16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int32  )) edgeDescs.push_back(EdgeDesc<int32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int64  )) edgeDescs.push_back(EdgeDesc<int64_t>());
            //if (bool(supportedTypes & SupportedTensorDataTypes::String )) edgeDescs.push_back(EdgeDesc<std::string>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Bool   )) edgeDescs.push_back(EdgeDesc<bool>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Float16)) edgeDescs.push_back(EdgeDesc<::MLFloat16>()); 
            if (bool(supportedTypes & SupportedTensorDataTypes::Float64)) edgeDescs.push_back(EdgeDesc<double>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt32 )) edgeDescs.push_back(EdgeDesc<uint32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt64 )) edgeDescs.push_back(EdgeDesc<uint64_t>());

            typeConstraints[i].allowedTypeCount = static_cast<uint32_t>(edgeDescs.size() - lastEdgeDescSize);
            lastEdgeDescSize = edgeDescs.size();
        }
        
        // Now that the edge descriptions list won't be re-allocated, assign pointers to its memory
        // into the type constraints entries
        size_t totalTypeCount = 0;
        for (gsl::index i = 0, ci = information.tensorTypeNames.size(); i < ci; ++i)
        {
            typeConstraints[i].allowedTypes = &edgeDescs.data()[totalTypeCount];
            totalTypeCount += typeConstraints[i].allowedTypeCount;
        }

        ComPtr<MLOperatorKernelFactory> factory =  wil::MakeOrThrow<MLOperatorKernelFactory>(information.creationFunction);
        ComPtr<MLOperatorShapeInferrer> shapeInferrer;
        
        if (information.shapeInferenceFunction)
        {
            shapeInferrer = wil::MakeOrThrow<MLOperatorShapeInferrer>(information.shapeInferenceFunction);
        }

        ComPtr<IMLOperatorSupportQueryPrivate> supportQuery;
        if (information.supportQueryFunction)
        {
            supportQuery = wil::MakeOrThrow<MLOperatorSupportQuery>(information.supportQueryFunction);
        }

        THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &desc, 
            factory.Get(), 
            shapeInferrer.Get(),
            supportQuery.Get(),
            true, // isInternalOperator
            information.canAliasFirstInput, // alias
            kernelSupportsGraph, // supportsGraph
            information.requiredInputCountForDmlGraphSupport ? &(*information.requiredInputCountForDmlGraphSupport) : nullptr,
            information.requiresFloatFormatsForGraph,
            information.requiresFloatFormatsExceptConstInputs.empty() ? nullptr : information.requiresFloatFormatsExceptConstInputs.data(),
            static_cast<uint32_t>(information.requiresFloatFormatsExceptConstInputs.size())
        ));
    }
}

} // namespace Dml
