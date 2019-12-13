// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "inc/AbiCustomRegistryImpl.h"

namespace winrt::Windows::AI::MachineLearning::implementation {

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorSetSchema(
    const MLOperatorSetId* opSetId,
    int baseline_version,
    const MLOperatorSchemaDescription* const* schema,
    uint32_t schemaCount,
    _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept try {
  for (uint32_t i = 0; i < schemaCount; ++i) {
    telemetry_helper.RegisterOperatorSetSchema(
        schema[i]->name,
        schema[i]->inputCount,
        schema[i]->outputCount,
        schema[i]->typeConstraintCount,
        schema[i]->attributeCount,
        schema[i]->defaultAttributeCount);
  }

  // Delegate to base class
  return AbiCustomRegistry::RegisterOperatorSetSchema(
      opSetId,
      baseline_version,
      schema,
      schemaCount,
      typeInferrer,
      shapeInferrer);
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel,
    IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept {
  return RegisterOperatorKernel(opKernel, operatorKernelFactory, shapeInferrer, nullptr, false, false, false);
}

HRESULT STDMETHODCALLTYPE AbiCustomRegistryImpl::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel,
    IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
    _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
    bool isInternalOperator,
    bool canAliasFirstInput,
    bool supportsGraph,
    const uint32_t* requiredInputCountForGraph,
    bool requiresFloatFormatsForGraph,
    _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs,
    uint32_t constantCpuInputCount) const noexcept try {
  // Log a custom op telemetry if the operator is not a built-in DML operator
  if (!isInternalOperator) {
    telemetry_helper.LogRegisterOperatorKernel(
        opKernel->name,
        opKernel->domain,
        static_cast<int>(opKernel->executionType));
  }

  // Delegate to base class
  return AbiCustomRegistry::RegisterOperatorKernel(
      opKernel,
      operatorKernelFactory,
      shapeInferrer,
      supportQuery,
      isInternalOperator,
      canAliasFirstInput,
      supportsGraph,
      requiredInputCountForGraph,
      requiresFloatFormatsForGraph,
      requiredConstantCpuInputs,
      constantCpuInputCount);
}
CATCH_RETURN();

}  // namespace winrt::Windows::AI::MachineLearning::implementation