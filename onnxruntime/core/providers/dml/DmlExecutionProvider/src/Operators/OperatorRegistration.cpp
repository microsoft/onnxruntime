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
    Ints8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32,
    Int32to64 = UInt32|Int32|UInt64|Int64,
    Float16to32 = Float16|Float32,
    Float16to64 = Float16|Float32|Float64,
    NumericDefault = Ints8to32|Float16to32, // Only simple numbers, not bool, complex, or string.
    Scalars8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32|Float16to32|Bool,
    AllScalars = UInt8|Int8|UInt16|Int16|UInt32|Int32|UInt64|Int64|Float16|Float32|Float64|Bool,
    AllScalarsButFloat64 = UInt8|Int8|UInt16|Int16|UInt32|Int32|UInt64|Int64|Float16to32|Bool,
    Ints8Bit = UInt8|Int8,
    Ints16Bit = UInt16|Int16,
    Ints32Bit = UInt32|Int32,
    All = static_cast<uint32_t>(-1),
};
DEFINE_ENUM_FLAG_OPERATORS(Dml::SupportedTensorDataTypes);

enum class DmlGraphSupport : uint32_t
{
    Supported                                          = 0,
    NotSupported                                       = 1,
    SupportedWith64BitTensorsVia32BitStrides           = 2, // Supports them via 32-bit tensors and doubled strides.
    SupportedWith64BitTensorsVia32BitStridesFromAnyEp  = 4, // Supports input from any execution provider (otherwise only inputs from other DML nodes)
    Prefer64BitTensorsDirectly                         = 8, // Natively supports 64-bit tensors. So avoid strided 32-bit unless the device lacks support.
};
DEFINE_ENUM_FLAG_OPERATORS(DmlGraphSupport);

struct OperatorRegistrationInformation
{
    const char* operatorName;
    int sinceVersion;
    const char* domain;
    MLOperatorKernelCreateFn creationFunction;
    MLOperatorShapeInferenceFunction shapeInferenceFunction;
    bool canAliasFirstInput;

    gsl::span<char const* const> tensorTypeNames;
    gsl::span<const SupportedTensorDataTypes> supportedTensorDataTypes;
    DmlGraphSupport dmlGraphSupport;

    std::pair<std::array<const uint32_t, 4>, int> requiredConstantCpuInputs = {{}, 0};

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
DML_OP_EXTERN_CREATION_FUNCTION(RoiAlign10);
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
DML_OP_EXTERN_CREATION_FUNCTION(Slice11);
DML_OP_EXTERN_CREATION_FUNCTION(Pad7);
DML_OP_EXTERN_CREATION_FUNCTION(Pad11);
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
DML_OP_EXTERN_CREATION_FUNCTION(Clip7);
DML_OP_EXTERN_CREATION_FUNCTION(Clip11);
DML_OP_EXTERN_CREATION_FUNCTION(Clip12);
DML_OP_EXTERN_CREATION_FUNCTION(Greater);
DML_OP_EXTERN_CREATION_FUNCTION(Less);
DML_OP_EXTERN_CREATION_FUNCTION(GreaterOrEqual);
DML_OP_EXTERN_CREATION_FUNCTION(LessOrEqual);
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
DML_OP_EXTERN_CREATION_FUNCTION(Einsum12);
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
DML_OP_EXTERN_CREATION_FUNCTION(Upsample7);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample9);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample10);
DML_OP_EXTERN_CREATION_FUNCTION(Sigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(HardSigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(ScaledTanh);
DML_OP_EXTERN_CREATION_FUNCTION(Relu);
DML_OP_EXTERN_CREATION_FUNCTION(LeakyRelu);
DML_OP_EXTERN_CREATION_FUNCTION(PRelu);
DML_OP_EXTERN_CREATION_FUNCTION(ThresholdedRelu);
DML_OP_EXTERN_CREATION_FUNCTION(Elu);
DML_OP_EXTERN_CREATION_FUNCTION(Celu);
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
DML_OP_EXTERN_CREATION_FUNCTION(TopK7);
DML_OP_EXTERN_CREATION_FUNCTION(TopK10);
DML_OP_EXTERN_CREATION_FUNCTION(TopK11);
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
DML_OP_EXTERN_CREATION_FUNCTION(Scatter9);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter11);
DML_OP_EXTERN_CREATION_FUNCTION(Resize10);
DML_OP_EXTERN_CREATION_FUNCTION(Resize11);
DML_OP_EXTERN_CREATION_FUNCTION(ConstantOfShape);
DML_OP_EXTERN_CREATION_FUNCTION(IsInf);
DML_OP_EXTERN_CREATION_FUNCTION(Mod);
DML_OP_EXTERN_CREATION_FUNCTION(BitShift);
DML_OP_EXTERN_CREATION_FUNCTION(CumSum);
DML_OP_EXTERN_CREATION_FUNCTION(GatherElements);
DML_OP_EXTERN_CREATION_FUNCTION(GatherND);
DML_OP_EXTERN_CREATION_FUNCTION(Range);
DML_OP_EXTERN_CREATION_FUNCTION(ReverseSequence);
DML_OP_EXTERN_CREATION_FUNCTION(Round);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterElements);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterND);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearAdd);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearConv);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(DynamicQuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(MatMulInteger);
DML_OP_EXTERN_CREATION_FUNCTION(ConvInteger);

DML_OP_EXTERN_QUERY_FUNCTION(MaxPool);
DML_OP_EXTERN_QUERY_FUNCTION(Slice);
DML_OP_EXTERN_QUERY_FUNCTION(Resize);
DML_OP_EXTERN_QUERY_FUNCTION(EinSum);

constexpr static std::array<const char*, 1> typeNameListDefault = {"T"};
constexpr static std::array<const char*, 2> typeNameListTwo = { "T1", "T2" };
constexpr static std::array<const char*, 3> typeNameListThree = { "T1", "T2", "T3" };
constexpr static std::array<const char*, 4> typeNameListFour = { "T1", "T2", "T3", "T4" };
constexpr static std::array<const char*, 2> typeNameListTopK = { "T", "I" };
constexpr static std::array<const char*, 2> typeNameListLogicalComparison = { "T", "T1" };
constexpr static std::array<const char*, 2> typeNameListPow12 = {"T", "T1"};
constexpr static std::array<const char*, 2> typeNameListConstantOfShape = { "T1", "T2" };
constexpr static std::array<const char*, 2> typeNameListScatterGather = { "T", "Tind" };
constexpr static std::array<const char*, 1> typeNameListScatterGatherND = { "T" }; // Tind is curiously missing, only allowing 64-bit.
constexpr static std::array<const char*, 2> typeNameListSlice10 = { "T", "Tind" };
constexpr static std::array<const char*, 2> typeNameListWhere = { "B", "T" };
constexpr static std::array<const char*, 1> typeNameListEyeLike = { "T2" };

constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAll = {SupportedTensorDataTypes::All};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat32 = {SupportedTensorDataTypes::Float32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32 = {SupportedTensorDataTypes::Float16to32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Int8 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Int32 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::UInt32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Int8to32 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit | SupportedTensorDataTypes::Ints16Bit | SupportedTensorDataTypes::Ints32Bit};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListInt8to32 = {SupportedTensorDataTypes::Ints8to32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListInt32to64AndFloat16to32 = {SupportedTensorDataTypes::Int32to64|SupportedTensorDataTypes::Float16to32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListNumericDefault = { SupportedTensorDataTypes::NumericDefault };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAllScalarsButFloat64 = { SupportedTensorDataTypes::AllScalarsButFloat64 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListBool = {SupportedTensorDataTypes::Bool};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListPow12 = {SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::NumericDefault};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListTopK = {SupportedTensorDataTypes::NumericDefault, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListIndices = { SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListCast = { SupportedTensorDataTypes::AllScalarsButFloat64, SupportedTensorDataTypes::AllScalarsButFloat64 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListScalars8to32 = { SupportedTensorDataTypes::Scalars8to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListScatterGather = { SupportedTensorDataTypes::Scalars8to32, SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListScatterGatherND = { SupportedTensorDataTypes::Scalars8to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListSlice10 = { SupportedTensorDataTypes::NumericDefault, SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListQuantizeLinear = { SupportedTensorDataTypes::Float32 | SupportedTensorDataTypes::Int32, SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDequantizeLinear = { SupportedTensorDataTypes::Float32, SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 | SupportedTensorDataTypes::Int32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListQuantize = { SupportedTensorDataTypes::Float32, SupportedTensorDataTypes::UInt8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListIsNan = { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListIsInf = { SupportedTensorDataTypes::Float32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListConstantOfShape = { SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64, SupportedTensorDataTypes::Float16to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListWhere = { SupportedTensorDataTypes::Bool, SupportedTensorDataTypes::Scalars8to32 };
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListOneHot = /* indices, depth, values */ { SupportedTensorDataTypes::Int32to64, SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::Scalars8to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListLogicalComparison7 = /* A&B,C */ { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListLogicalComparison9 = /* A&B,C */ { SupportedTensorDataTypes::NumericDefault, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListSigned = { SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int16 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListRange = {SupportedTensorDataTypes::Int16|SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Float32};
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListInteger = {SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Int32 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListInteger8 = {SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListRoiAlign = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListQLinearMatMul = {
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8,
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8,
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8
};
constexpr static std::array<SupportedTensorDataTypes, 4> supportedTypeListQLinearConv = {
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8,
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8,
    SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8,
    SupportedTensorDataTypes::Int32 
};


constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDynamicQuantizeLinear = {
    SupportedTensorDataTypes::Float32, 
    SupportedTensorDataTypes::UInt8,
};

template<typename... Args>
constexpr auto requiredConstantCpuInputs(Args... args)
{
    std::array<const uint32_t, 4> inputs = {static_cast<uint32_t>(args)...};
    return std::make_pair(inputs, static_cast<int>(sizeof...(args)));
}

// Define a single row of registration information.
#define REG_INFO(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, ##__VA_ARGS__, 

// Versioned operator
#define REG_INFO_VER(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName##version, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName##version>, false, ##__VA_ARGS__, 

// Identity operators use Copy, alias their first input, and require floating point formats
// for usage in the graph, besides constant inputs.  This is because they currently use 
// element-wise identity operators in the graph for striding support, but issue actual copies 
// outside the graph.  Element-wise identity currently only supports floating point types.  
#define REG_INFO_ID(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, CreateCopy, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, true, ##__VA_ARGS__, 

// MS-domain operators
#define REG_INFO_MS(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, ##__VA_ARGS__, 

// MS-domain operators
#define REG_INFO_MSDML(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDmlDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, false, ##__VA_ARGS__, 

constexpr static OperatorRegistrationInformation operatorRegistrationInformationTable[] =
{
///  Domain/Type, Ver,  Name,                               TypeNames,                       Types,                                 Graph Support,                  Required const CPU inputs, 
///                                                                                                                                                                 Input count required for graph support,
///                                                                                                                                                                 Support query function

    // Deep Learning Standard Layers
    {REG_INFO(      7,  Conv,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Conv,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ConvTranspose,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ConvTranspose,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalAveragePool,                  typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      8,  MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      10, MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      11, MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      12, MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32Int8,       DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},

    {REG_INFO(      7,  GlobalMaxPool,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LpPool,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  LpPool,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalLpPool,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MaxRoiPool,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 10,  RoiAlign,                           typeNameListTwo,                supportedTypeListRoiAlign,              DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(      7,  InstanceNormalization,              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)}, // v9 just removes 'spatial' attribute.
    {REG_INFO(      7,  LRN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MeanVarianceNormalization,          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  MeanVarianceNormalization,          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LpNormalization,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  RNN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO(      7,  GRU,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO(      7,  LSTM,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO_MS(   1,  ConvTransposeWithDynamicPads,       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(2))},

    // Data Reorganization Layers
    {REG_INFO(      7,  Split,                              typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     11,  Split,                              typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)}, // Adds negative axis.
    {REG_INFO(      7,  Transpose,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      7,  Concat,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     11,  Concat,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)}, // Adds negative axis.
    {REG_INFO_VER( 7,   Slice,                              typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_VER( 10,  Slice,                              typeNameListSlice10,            supportedTypeListSlice10,               DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1, 2, 3, 4), std::nullopt, QuerySlice)}, // Adds negative axes.
    {REG_INFO_VER( 11,  Slice,                              typeNameListSlice10,            supportedTypeListSlice10,               DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1, 2, 3, 4), std::nullopt, QuerySlice)},
    {REG_INFO_VER(  7,  Pad,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_VER( 11,  Pad,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1, 2) /*pads, value*/)}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
    {REG_INFO(      7,  SpaceToDepth,                       typeNameListDefault,            supportedTypeListScalars8to32,          DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      7,  DepthToSpace,                       typeNameListDefault,            supportedTypeListScalars8to32,          DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     11,  DepthToSpace,                       typeNameListDefault,            supportedTypeListScalars8to32,          DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      7,  Tile,                               typeNameListDefault,            supportedTypeListScalars8to32,          DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1))},
    {REG_INFO(      8,  Expand,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1))},
    {REG_INFO(      9,  ConstantOfShape,                    typeNameListConstantOfShape,    supportedTypeListConstantOfShape,       DmlGraphSupport::NotSupported,   requiredConstantCpuInputs(0))},
    {REG_INFO(      7,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  GatherElements,                     typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  GatherND,                           typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     12,  GatherND,                           typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO_VER(  9,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO_VER( 11,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  ScatterElements,                    typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  ScatterND,                          typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(      9,  EyeLike,                            typeNameListEyeLike,            supportedTypeListScalars8to32,          DmlGraphSupport::Supported)},

    // Data reorganization that merely changes the dimensions while keeping the data identical.
    {REG_INFO_ID(   7,  Identity,                           typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(   7,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(   9,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(  11,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(   7,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(  11,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(   7,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(  11,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_ID(   7,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalarsButFloat64,  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1))},

    // Elementwise
    {REG_INFO(      7,  Sqrt,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Reciprocal,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Pow,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      12, Pow,                                typeNameListPow12,              supportedTypeListPow12,                 DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Exp,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Log,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Abs,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Ceil,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Floor,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  7,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 11,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1,2))},
    {REG_INFO_VER( 12,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32Int8to32,   DmlGraphSupport::Supported,     requiredConstantCpuInputs(1,2))},
    {REG_INFO(      7,  Add,                                typeNameListDefault,            supportedTypeListFloat16to32Int32,      DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sub,                                typeNameListDefault,            supportedTypeListFloat16to32Int32,      DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Mul,                                typeNameListDefault,            supportedTypeListFloat16to32Int32,      DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Div,                                typeNameListDefault,            supportedTypeListFloat16to32Int32,      DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      12, Max,                                typeNameListDefault,            supportedTypeListFloat16to32Int8to32,   DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      12, Min,                                typeNameListDefault,            supportedTypeListFloat16to32Int8to32,   DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Cos,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sin,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Tan,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Acos,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Asin,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Atan,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Affine,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      10, QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear,        DmlGraphSupport::Supported)},
    {REG_INFO(      10, DequantizeLinear,                   typeNameListTwo,                supportedTypeListDequantizeLinear,      DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantize,              DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  DequantizeLinear,                   typeNameListTwo,                supportedTypeListQuantize,              DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Sign,                               typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(      9,  IsNan,                              typeNameListTwo,                supportedTypeListIsNan,                 DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Sinh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Cosh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Asinh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Acosh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Atanh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Erf,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Where,                              typeNameListWhere,              supportedTypeListWhere,                 DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 12,  Einsum,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), std::nullopt, QueryEinSum )},
    {REG_INFO(      7,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32Int8,       DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32Int8,       DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ArgMax,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     11,  ArgMax,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     12,  ArgMax,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      7,  ArgMin,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     11,  ArgMin,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(     12,  ArgMin,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO(      7,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Neg,                                typeNameListDefault,            supportedTypeListSigned,                DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      12, GreaterOrEqual,                     typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      12, LessOrEqual,                        typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Not,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  And,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Or,                                 typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Xor,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},

    // Imaging Operators
    {REG_INFO(      7,  Crop,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ImageScaler,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  7,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  9,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 10,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 10,  Resize,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 11,  Resize,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*roi, scales, sizes*/, std::nullopt, QueryResize)},

    // Activation Functions
    {REG_INFO(      7,  Sigmoid,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  HardSigmoid,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Tanh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ScaledTanh,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Relu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LeakyRelu,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Elu,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      12, Celu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Selu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softsign,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softplus,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ParametricSoftplus,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Dropout,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Shrink,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},

    // Uncategorized
    {REG_INFO(      7,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MemcpyFromHost,                     typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO(      7,  MemcpyToHost,                       typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO_VER(  7,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides)},
    {REG_INFO_VER( 10,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1))},
    {REG_INFO_VER( 11,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides,      requiredConstantCpuInputs(1))},
    {REG_INFO(      9,  OneHot,                             typeNameListThree,              supportedTypeListOneHot,                DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp,      requiredConstantCpuInputs(1))},
    {REG_INFO(     11,  OneHot,                             typeNameListThree,              supportedTypeListOneHot,                DmlGraphSupport::Supported|DmlGraphSupport::Prefer64BitTensorsDirectly|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp,      requiredConstantCpuInputs(1))},

    // Fused operators
    {REG_INFO_MSDML(1,  FusedConv,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedConvTranspose,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedInstanceNormalization,         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedBatchNormalization,            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedMeanVarianceNormalization,     typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedGemm,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedMatMul,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedAdd,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  FusedSum,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(), 2)},
    
    {REG_INFO(     10,  IsInf,                              typeNameListTwo,                supportedTypeListIsInf,                 DmlGraphSupport::Supported)},
    {REG_INFO(     10,  Mod,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(     11,  BitShift,                           typeNameListDefault,            supportedTypeListInt8to32,              DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Round,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ReverseSequence,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp)},
    {REG_INFO(     11,  CumSum,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(     11,  Range,                              typeNameListDefault,            supportedTypeListRange,                 DmlGraphSupport::Supported,      requiredConstantCpuInputs(0,1,2))},

    {REG_INFO(      9,  MaxUnpool,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp,      requiredConstantCpuInputs(2))},
    {REG_INFO(     11,  MaxUnpool,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported|DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp,      requiredConstantCpuInputs(2))}, // 11 is identical to 9.
        
    {REG_INFO_MS(  1,   QLinearAdd,                         typeNameListDefault,            supportedTypeListInteger8,              DmlGraphSupport::Supported)},
    {REG_INFO(     10,  QLinearConv,                        typeNameListFour,               supportedTypeListQLinearConv,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  QLinearMatMul,                      typeNameListThree,              supportedTypeListQLinearMatMul,         DmlGraphSupport::Supported)},
    {REG_INFO(     10,  MatMulInteger,                      typeNameListThree,              supportedTypeListInteger,               DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ConvInteger,                        typeNameListThree,              supportedTypeListInteger,               DmlGraphSupport::Supported)},
    {REG_INFO(     11,  DynamicQuantizeLinear,              typeNameListTwo,                supportedTypeListDynamicQuantizeLinear, DmlGraphSupport::Supported)},
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
        bool kernelSupportsGraph = !bool(information.dmlGraphSupport & DmlGraphSupport::NotSupported);
        bool prefer64BitTensorsDirectly = bool(information.dmlGraphSupport & DmlGraphSupport::Prefer64BitTensorsDirectly);
        bool supportedWith64BitTensorsVia32BitStrides = bool(information.dmlGraphSupport & DmlGraphSupport::SupportedWith64BitTensorsVia32BitStrides);
        bool supportedWith64BitTensorsVia32BitStridesFromAnyEp = bool(information.dmlGraphSupport & DmlGraphSupport::SupportedWith64BitTensorsVia32BitStridesFromAnyEp);

        desc.options = information.shapeInferenceFunction ? 
            MLOperatorKernelOptions::None : MLOperatorKernelOptions::AllowDynamicInputShapes;

        desc.minimumOperatorSetVersion = information.sinceVersion;
    
        typeConstraints.resize(information.tensorTypeNames.size());
        desc.typeConstraints = typeConstraints.data();
        desc.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

#if _DEBUG
        // If some version of the operator is supported for fusion, check that each registered version is also supported.
        // This ensures that table of operators and versions supporting fusion does not become stale as operator sets are added.
        FusionHelpers::AssertFusableOperatorSupportsVersionIfExists(desc.name, desc.domain, desc.minimumOperatorSetVersion);
#endif

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
            supportedWith64BitTensorsVia32BitStrides,
            supportedWith64BitTensorsVia32BitStridesFromAnyEp,
            prefer64BitTensorsDirectly,
            information.requiredConstantCpuInputs.first.data(),
            static_cast<uint32_t>(information.requiredConstantCpuInputs.second)
        ));
    }
}

} // namespace Dml
