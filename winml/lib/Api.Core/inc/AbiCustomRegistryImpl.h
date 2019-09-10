//-----------------------------------------------------------------------------
//
//    Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include "core/providers/dml/DmlExecutionProvider/src/AbiCustomRegistry.h"

namespace winrt::Windows::AI::MachineLearning::implementation
{ 

// An implementation of AbiCustomRegistry that emits telemetry events when operator kernels or schemas are registered.
class AbiCustomRegistryImpl : public AbiCustomRegistry
{
 public:
    HRESULT STDMETHODCALLTYPE RegisterOperatorSetSchema(
        const MLOperatorSetId* opSetId,
        int baseline_version,
        const MLOperatorSchemaDescription* const* schema,
        uint32_t schemaCount,
        _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept override;

    HRESULT STDMETHODCALLTYPE RegisterOperatorKernel(
        const MLOperatorKernelDescription* operatorKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
        bool isInternalOperator,
        bool canAliasFirstInput,
        bool supportsGraph,
        const uint32_t* requiredInputCountForGraph = nullptr,
        bool requiresFloatFormatsForGraph = false,
        _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs = nullptr,
        uint32_t constantCpuInputCount = 0) const noexcept override;

    HRESULT STDMETHODCALLTYPE RegisterOperatorKernel(
        const MLOperatorKernelDescription* opKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept override;
};

}    // namespace winrt::Windows::AI::MachineLearning::implementation
