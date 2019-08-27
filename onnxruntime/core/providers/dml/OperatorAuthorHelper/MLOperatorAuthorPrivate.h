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


interface __declspec(uuid("95ED8242-8B98-4A7B-A697-3D9D57CC085C"))
IMLOperatorShapeInferenceContextPrivate : public IMLOperatorShapeInferenceContext 
{
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex, 
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept PURE;
};

interface __declspec(uuid("F5197886-9D15-4939-87BE-B682CE3CA8FA"))
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

interface DECLSPEC_UUID("2508589D-D59F-4308-887A-9AE82FE4C2D4") DECLSPEC_NOVTABLE
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
interface DECLSPEC_UUID("6B510C7A-8EA5-4105-B2A5-81E8D651376B") DECLSPEC_NOVTABLE
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
