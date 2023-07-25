// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_DML
#include "core/providers/dml/DmlExecutionProvider/src/AbiCustomRegistry.h"

namespace Windows::AI::MachineLearning::Adapter {

// An implementation of AbiCustomRegistry that emits telemetry events when operator kernels or schemas are registered.
class AbiCustomRegistryImpl : public AbiCustomRegistry {
 public:
  HRESULT STDMETHODCALLTYPE RegisterOperatorSetSchema(
    const MLOperatorSetId* op_set_id,
    int baseline_version,
    const MLOperatorSchemaDescription* const* schema,
    uint32_t schema_count,
    _In_opt_ IMLOperatorTypeInferrer* type_inferrer,
    _In_opt_ IMLOperatorShapeInferrer* shape_inferrer
  ) const noexcept override;

  HRESULT STDMETHODCALLTYPE RegisterOperatorKernel(
    const MLOperatorKernelDescription* operator_kernel,
    IMLOperatorKernelFactory* operator_kernel_factory,
    _In_opt_ IMLOperatorShapeInferrer* shape_inferrer,
    _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
    bool is_internal_operator,
    bool can_alias_first_input,
    bool supports_graph,
    const uint32_t* required_input_count_for_graph = nullptr,
    _In_reads_(constant_cpu_input_count) const uint32_t* required_constant_cpu_inputs = nullptr,
    uint32_t constant_cpu_input_count = 0
  ) const noexcept override;

  HRESULT STDMETHODCALLTYPE RegisterOperatorKernel(
    const MLOperatorKernelDescription* op_kernel,
    IMLOperatorKernelFactory* operator_kernel_factory,
    _In_opt_ IMLOperatorShapeInferrer* shape_inferrer
  ) const noexcept override;
};

}  // namespace Windows::AI::MachineLearning::Adapter
#endif USE_DML
