// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

interface IDMLOperation;
interface IDMLOperator;
enum DML_TENSOR_DATA_TYPE;
struct DML_OPERATOR_DESC;

struct MLOperatorKernelDmlProperties
{
    uint32_t dmlInputCount;
    _Field_size_opt_(dmlInputCount) const uint32_t* kernelInputIndices;

    uint32_t dmlOutputCount;
    _Field_size_opt_(dmlOutputCount) const uint32_t* kernelOutputIndices;

    bool allowHalfPrecisionComputation = false;
};


interface __declspec(uuid("aa2173bb-6684-4de8-abf2-9acbdf88b426"))
IMLOperatorShapeInferenceContextPrivate : public IMLOperatorShapeInferenceContext 
{
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;
};

interface __declspec(uuid("63bff199-0203-43c7-86c4-f442a599df4c"))
IMLOperatorKernelCreationContextPrivate : public IMLOperatorKernelCreationContext 
{
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;
    
    STDMETHOD_(bool, IsDmlGraphNode)() const noexcept PURE;

    STDMETHOD(SetDmlOperator)(
        IDMLOperator* op,
        _In_ const DML_OPERATOR_DESC* desc,
        _In_opt_ const MLOperatorKernelDmlProperties* dmlProperties
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
        bool supportedWith64BitTensorsVia32BitStrides = false,
        bool supportedWith64BitTensorsVia32BitStridesFromAnyEp = false,
        bool prefer64BitTensorsDirectly = false,
        _In_reads_(constantCpuInputCount) const uint32_t* constantCpuInputs = nullptr,
        uint32_t constantCpuInputCount = 0
        ) const noexcept PURE;
};

// Declare private enum MLOperatorAttributeType::Tensor.
//
//      enum class MLOperatorAttributeType : uint32_t
//          ...
//          //! Tensor
//          Tensor = 5,
//          ...
constexpr enum MLOperatorAttributeType MLOperatorAttributeTypeTensor = static_cast<MLOperatorAttributeType>(5);
