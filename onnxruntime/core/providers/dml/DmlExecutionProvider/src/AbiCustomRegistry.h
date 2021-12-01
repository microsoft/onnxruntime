// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"

namespace WRL
{
    template <typename... TInterfaces>
    using Base = Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
        TInterfaces...
        >;
}

namespace Windows::AI::MachineLearning::Adapter
{ 

using namespace Microsoft::WRL;

class AbiCustomRegistry : public WRL::Base<IMLOperatorRegistry, IMLOperatorRegistryPrivate>
{
 public:
    AbiCustomRegistry();

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
        _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
        bool isInternalOperator,
        bool canAliasFirstInput,
        bool supportsGraph,
        const uint32_t* requiredInputCountForGraph = nullptr,
        bool supportedWith64BitTensorsVia32BitStrides = false,
        bool supportedWith64BitTensorsVia32BitStridesFromAnyEp = false,
        bool prefer64BitTensorsDirectly = false,
        bool support64BitTensorsViaEmulation = false,
        _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs = nullptr,
        uint32_t constantCpuInputCount = 0) const noexcept override;

    HRESULT STDMETHODCALLTYPE RegisterOperatorKernel(
        const MLOperatorKernelDescription* opKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept override;

    std::list<std::shared_ptr<onnxruntime::CustomRegistry>> GetRegistries()
    {
        std::list<std::shared_ptr<onnxruntime::CustomRegistry>> registries;
        for (auto& registry : m_customRegistryOpsetVerMap)
        {
            registries.push_back(registry.second);
        }
        
        registries.push_back(m_kernelRegistry);

        return registries;
    }

    std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> GetSchemaRegistries()
    {
        std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> registries;
        for (auto& registry : m_customRegistryOpsetVerMap)
        {
            registries.push_back(registry.second->GetOpschemaRegistry());
        }

        return registries;
    }


    std::shared_ptr<onnxruntime::CustomRegistry> GetLotusKernelRegistry()
    {
        return m_kernelRegistry;
    }

    std::shared_ptr<InternalRegistrationInfoMap> GetInternalRegInfoMap() const
    {
        return m_internalRegInfoMap;
    }

 private:
    static onnx::OpSchema ConvertOpSchema(
        _In_z_ const char* domain, 
        const MLOperatorSchemaDescription& abiSchema,
        IMLOperatorTypeInferrer* typeInferrer,
        IMLOperatorShapeInferrer* shapeInferrer);

    static std::string ConvertFormalParameterType(const MLOperatorSchemaEdgeDescription& formalParameter);
    static onnx::OpSchema::FormalParameterOption ConvertFormalParameterOption(MLOperatorParameterOptions options);
    static void SetAttributesAndDefaults(onnx::OpSchema& schema, const MLOperatorSchemaDescription& abiSchema);
    
    static AttributeMap GetDefaultAttributes(const MLOperatorKernelDescription* opKernel);

    std::shared_ptr<onnxruntime::CustomRegistry> m_kernelRegistry;

    // Map between (baseline version, opset version) and registries.  This ensures that no registry has multiple
    // versions of the same domain within it.  This works around limitations in Lotus op-set version arbitration
    // (see LotusOpSchemaRegistry::GetSchemaAndHistory).
    mutable std::map<std::pair<int, int>, std::shared_ptr<onnxruntime::CustomRegistry>> m_customRegistryOpsetVerMap;

    // Map between Lotus KernelDefs and extended data used during partitioning
    mutable std::shared_ptr<InternalRegistrationInfoMap> m_internalRegInfoMap;

};

}    // namespace Windows::AI::MachineLearning::Adapter
