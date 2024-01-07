// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

interface IDMLOperation;
interface IDMLOperator;
struct DML_OPERATOR_DESC;
struct DML_INPUT_GRAPH_EDGE_DESC;
struct DML_OUTPUT_GRAPH_EDGE_DESC;
struct DML_INTERMEDIATE_GRAPH_EDGE_DESC;

// Either nodesAsOpDesc or nodesAsIDMLOperator is present.
//  1) Operator kernels which implement operators using only a single DML operator will pass a DML_OPERATOR_DESC.
//     These kernels pass DML_OPERATOR_DESC, because while building Dml graph (inside FusedGraphKernel.cpp) we can change the
//     the flag of constant inputs to DML_TENSOR_FLAG_OWNED_BY_DML.
//  2) Operator kernels which implement operators using DMLX graph, they will pass IDMLOperator and won't be able
//     to use DML_TENSOR_FLAG_OWNED_BY_DML.
struct MLOperatorGraphDesc
{
    uint32_t nodeCount;
    _Field_size_opt_(nodeCount) const DML_OPERATOR_DESC** nodesAsOpDesc;
    _Field_size_opt_(nodeCount) IDMLOperator** nodesAsIDMLOperator;

    uint32_t inputEdgeCount;
    _Field_size_(inputEdgeCount) const DML_INPUT_GRAPH_EDGE_DESC* inputEdges;

    uint32_t intermediateEdgeCount;
    _Field_size_(intermediateEdgeCount) const DML_INTERMEDIATE_GRAPH_EDGE_DESC* intermediateEdges;

    uint32_t outputEdgeCount;
    _Field_size_(outputEdgeCount) const DML_OUTPUT_GRAPH_EDGE_DESC* outputEdges;
};


interface __declspec(uuid("aa2173bb-6684-4de8-abf2-9acbdf88b426"))
IMLOperatorShapeInferenceContextPrivate : public IMLOperatorShapeInferenceContext
{
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex,
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    STDMETHOD(TryGetConstantInputTensor)(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    //! Gets the number of dimensions of a tensor output of the operator.
    STDMETHOD(GetSequenceInputInfo)(
        uint32_t inputIndex,
        _Out_ uint32_t* inputCount,
        MLOperatorTensorDataType* dataType
        ) const noexcept PURE;

    //! Gets the number of dimensions of a tensor output of the operator.
    STDMETHOD(GetSequenceInputTensorDimensionCount)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        _Out_ uint32_t* dimensionCount
        ) const noexcept PURE;

    //! Gets the sizes of dimensions of an input tensor of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetSequenceInputTensorShape)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;
};

interface __declspec(uuid("63bff199-0203-43c7-86c4-f442a599df4c"))
IMLOperatorKernelCreationContextPrivate : public IMLOperatorKernelCreationContext
{
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex,
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    STDMETHOD(TryGetConstantInputTensor)(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    STDMETHOD_(bool, IsDmlGraphNode)() const noexcept PURE;

    STDMETHOD(SetDmlOperator)(
        _In_ const MLOperatorGraphDesc* operatorGraphDesc
    ) const noexcept PURE;
};

interface __declspec(uuid("1d2e1226-a918-4236-8775-175cf1f52c9a"))
IMLOperatorKernelCreationContextNodeWrapperPrivate : public IMLOperatorKernelCreationContextPrivate
{
    //! Gets the minimum size of a char buffer to store the node name (including null terminator).
    //! Returns 1 if the node has no name (calling GetUtf8Name will write a single null terminator).
    STDMETHOD_(uint32_t, GetUtf8NameBufferSizeInBytes)() const noexcept PURE;

    //! Writes the node name and null terminator into a char buffer.
    STDMETHOD(GetUtf8Name)(
        uint32_t bufferSizeInBytes,
        _Out_writes_bytes_(bufferSizeInBytes) char* name
        ) const noexcept PURE;

    //! Gets the minimum size of a wchar buffer to store the node name (including null terminator).
    //! Returns sizeof(wchar_t) if the node has no name (calling GetWideName will write a null terminator).
    STDMETHOD_(uint32_t, GetWideNameBufferSizeInBytes)() const noexcept PURE;

    //! Writes the node name and null terminator into a wchar buffer.
    STDMETHOD(GetWideName)(
        uint32_t bufferSizeInBytes,
        _Out_writes_bytes_(bufferSizeInBytes) wchar_t* name
        ) const noexcept PURE;

    STDMETHOD(GetExecutionProvider)(
        _Outptr_result_maybenull_ IUnknown** executionProvider
        ) const noexcept PURE;
};

//! \interface IMLOperatorAttributes1
//! \brief Represents the values of an operator's attributes, as determined by a model using the operator.
//! This interface is called by implementations of custom operator kernels, and by implementations
//! of shape and type inferrers.
interface DECLSPEC_UUID("3a798815-dfe3-4bcd-b6a6-f70650d5f80b") DECLSPEC_NOVTABLE
IMLOperatorAttributes1 : public IMLOperatorAttributes
{
    //! Gets an interface pointer for the constant tensor.
    //! Note the tensor is CPU side (IsCpuData is true).
    STDMETHOD(GetTensorAttribute)(
        _In_z_ const char* name,
        _COM_Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;
};

interface __declspec(uuid("897bb586-6cee-4106-8513-dda33151c109")) DECLSPEC_NOVTABLE
IMLOperatorSupportQueryContextPrivate : public IMLOperatorAttributes1
{
    //! Gets the number of inputs to the operator.
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept PURE;

    //! Gets the number of outputs to the operator.
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept PURE;

    //! Returns true if an input to the operator is valid.
    //! This always returns true except for optional inputs and invalid indices.
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept PURE;

    //! Returns true if an output to the operator is valid.
    //! This always returns true if within GetOutputCount except for optional outputs.
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept PURE;

    //! Gets the description of the specified input edge of the operator.
    STDMETHOD(GetInputEdgeDescription)(
        uint32_t inputIndex,
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;

    //! Gets the description of the specified output edge of the operator.
    STDMETHOD(GetOutputEdgeDescription)(
        uint32_t outputIndex,
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;
};

interface __declspec(uuid("023954b3-aed2-4b03-b7c7-f0838053a9a1")) DECLSPEC_NOVTABLE
IMLOperatorSupportQueryPrivate : public IUnknown
{
    STDMETHOD(QuerySupport)(
        IMLOperatorSupportQueryContextPrivate* context,
        BOOL* isSupported
        ) noexcept PURE;
};

interface DECLSPEC_UUID("3de1dc1e-13e9-4099-ae88-7b4100083415") DECLSPEC_NOVTABLE
IMLOperatorRegistryPrivate : public IUnknown
{
    STDMETHOD(RegisterOperatorKernel)(
        const MLOperatorKernelDescription* operatorKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
        _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
        bool isInternalOperator,
        bool canAliasFirstInput,
        bool supportsGraph,
        const uint32_t* requiredInputCountForGraph = nullptr,
        _In_reads_(constantCpuInputCount) const uint32_t* constantCpuInputs = nullptr,
        uint32_t constantCpuInputCount = 0
        ) const noexcept PURE;
};

//! \interface IMLOperatorTensorShapeDescription1
//! \brief Represents the set of input and output tensor shapes of an operator.
//! This interface is called by the factory objects registered to create kernels.
//! It is available to these factory objects unless corresponding kernels are
//! registered using the MLOperatorKernelOptions::AllowDynamicInputShapes flag.
interface DECLSPEC_UUID("440DA47C-018B-41F6-80A4-13FCF0544F37") DECLSPEC_NOVTABLE
IMLOperatorTensorShapeDescriptionPrivate : IUnknown
{
    //! Gets the number of dimensions of a tensor output of the operator.
    STDMETHOD(GetSequenceInputInfo)(
        uint32_t inputIndex,
        _Out_ uint32_t* inputCount,
        MLOperatorTensorDataType* dataType
        ) const noexcept PURE;

    //! Gets the number of dimensions of a tensor input of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetSequenceInputTensorDimensionCount)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        _Out_ uint32_t* dimensionCount
        ) const noexcept PURE;

    //! Gets the sizes of dimensions of an input tensor of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetSequenceInputTensorShape)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;

};

//! \interface IMLOperatorKernelContext
//! \brief Provides information about an operator's usage while kernels are being computed.
interface DECLSPEC_UUID("AFEED22E-B1B4-4DCE-BE09-27B95B7AD5AF") DECLSPEC_NOVTABLE
IMLOperatorKernelContextPrivate : IUnknown
{
    //! Gets the input tensor of the operator at the specified index.
    //! This sets tensor to nullptr for optional inputs which do not exist.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetSequenceInputTensor)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        _COM_Outptr_result_maybenull_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    //! Prepare the output tensor of the operator at the specified index.
    STDMETHOD(PrepareSequenceOutput)(
        uint32_t outputIndex,
        MLOperatorTensorDataType dataType) const noexcept PURE;

    //! Gets the output tensor of the operator at the specified index.
    //! This sets tensor to nullptr for optional outputs which do not exist.
    //! Returns an error if the output at the specified index is not a tensor.
    STDMETHOD(GetSequenceOutputTensor)(
        uint32_t outputIndex,
        uint32_t sequenceIndex,
        MLOperatorTensorDataType dataType,
        uint32_t dimensions,
        const uint32_t* dimensionSizes,
        bool gpuOutput,
        _COM_Outptr_result_maybenull_ IMLOperatorTensor** tensor
        ) const noexcept PURE;

    //! Gets the input tensor of the operator at the specified index.
    //! This sets tensor to nullptr for optional inputs which do not exist.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetSequenceInputInfo)(
        uint32_t inputIndex,
        _Out_ uint32_t* inputCount,
        MLOperatorTensorDataType* dataType
        ) const noexcept PURE;

    //! Returns whether the tensor at inputIndex is a sequence tensor or not
    STDMETHOD_(bool, IsSequenceInputTensor)(uint32_t inputIndex) const = 0;
};

// Declare private enum MLOperatorAttributeType::Tensor.
//
//      enum class MLOperatorAttributeType : uint32_t
//          ...
//          //! Tensor
//          Tensor = 5,
//          ...
constexpr enum MLOperatorAttributeType MLOperatorAttributeTypeTensor = static_cast<MLOperatorAttributeType>(5);
