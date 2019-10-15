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

interface DECLSPEC_UUID("3de1dc1e-13e9-4099-ae88-7b4100083415") DECLSPEC_NOVTABLE
IMLOperatorRegistryPrivate : public IUnknown
{
    STDMETHOD(RegisterOperatorKernel)(
        const MLOperatorKernelDescription* operatorKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
        bool isInternalOperator,
        bool canAliasFirstInput,
        bool supportsGraph,
        const uint32_t* requiredInputCountForGraph = nullptr,
        bool requiresFloatFormatsForGraph = false,
        _In_reads_(constantCpuInputCount) const uint32_t* constantCpuInputs = nullptr,
        uint32_t constantCpuInputCount = 0
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

// Declare private enum MLOperatorAttributeType::Tensor.
//
//      enum class MLOperatorAttributeType : uint32_t
//          ...
//          //! Tensor
//          Tensor = 5,
//          ...
constexpr enum MLOperatorAttributeType MLOperatorAttributeTypeTensor = static_cast<MLOperatorAttributeType>(5);
