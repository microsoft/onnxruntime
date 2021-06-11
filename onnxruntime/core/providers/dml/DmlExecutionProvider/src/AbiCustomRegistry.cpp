// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "AbiCustomRegistry.h"

namespace Windows::AI::MachineLearning::Adapter
{

AbiCustomRegistry::AbiCustomRegistry() : 
    m_kernelRegistry(std::make_shared<onnxruntime::CustomRegistry>()),
    m_internalRegInfoMap(std::make_shared<InternalRegistrationInfoMap>())
{
}

onnx::OpSchema::FormalParameterOption AbiCustomRegistry::ConvertFormalParameterOption(MLOperatorParameterOptions options)
{
    switch (options)
    {
        case MLOperatorParameterOptions::Single:
            return onnx::OpSchema::FormalParameterOption::Single;

        case MLOperatorParameterOptions::Optional:
            return onnx::OpSchema::FormalParameterOption::Optional;

        case MLOperatorParameterOptions::Variadic:
            return onnx::OpSchema::FormalParameterOption::Variadic;

        default:
            THROW_HR(E_NOTIMPL);
    }
}

// Convert edge types from the ABI types to ONNX strings
std::string AbiCustomRegistry::ConvertFormalParameterType(const MLOperatorSchemaEdgeDescription& formalParameter)
{
    ML_CHECK_BOOL(formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::Label ||
                                formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::EdgeDescription);

    if (formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::Label)
    {
        return formalParameter.typeLabel;
    } else
    {
        return ToTypeString(formalParameter.edgeDescription);
    }
}

// Convert type constraints from the ABI types to ONNX strings
std::vector<std::string> ConvertTypeConstraintTypes(const MLOperatorEdgeTypeConstrant& constraint)
{
    std::vector<std::string> ret;
    ret.reserve(constraint.allowedTypeCount);

    for (uint32_t i = 0; i < constraint.allowedTypeCount; ++i)
    {
        ret.emplace_back(ToTypeString(constraint.allowedTypes[i]));
    }

    return ret;
}

// Convert attributes and defaults from the ABI to ONNX schema
void AbiCustomRegistry::SetAttributesAndDefaults(onnx::OpSchema& schema, const MLOperatorSchemaDescription& abiSchema)
{
    // Create a map with default attributes
    std::map<std::string, const MLOperatorAttributeNameValue*> defaultAttributes;
    for (uint32_t attributeIndex = 0; attributeIndex < abiSchema.defaultAttributeCount; ++attributeIndex)
    {
        const MLOperatorAttributeNameValue& defaultAttribute = abiSchema.defaultAttributes[attributeIndex];
        defaultAttributes[defaultAttribute.name] = &defaultAttribute;
    }

    // Set each attribute along with default values, looked up by name, if available
    for (uint32_t attributeIndex = 0; attributeIndex < abiSchema.attributeCount; ++attributeIndex)
    {
        const MLOperatorAttribute& attribute = abiSchema.attributes[attributeIndex];
        auto defaultVal = defaultAttributes.find(attribute.name);
        if (defaultVal == defaultAttributes.end())
        {
            schema.Attr(attribute.name, "", ToProto(attribute.type), attribute.required);
        } 
        else
        {
            ML_CHECK_BOOL(!attribute.required);
            ML_CHECK_BOOL(attribute.type == defaultVal->second->type);
            uint32_t defaultCount = defaultVal->second->valueCount;

            switch (attribute.type)
            {
                case MLOperatorAttributeType::Float:
                    ML_CHECK_BOOL(defaultCount == 1);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVal->second->floats[0]);
                    break;

                case MLOperatorAttributeType::Int:
                    ML_CHECK_BOOL(defaultCount == 1);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVal->second->ints[0]);
                    break;

                case MLOperatorAttributeType::String:
                    ML_CHECK_BOOL(defaultCount == 1);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), std::string(defaultVal->second->strings[0]));
                    break;

                case MLOperatorAttributeType::FloatArray:
                {
                    std::vector<float> defaultVals(defaultVal->second->floats, defaultVal->second->floats + defaultCount);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                    break;
                }

                case MLOperatorAttributeType::IntArray:
                {
                    std::vector<int64_t> defaultVals(defaultVal->second->ints, defaultVal->second->ints + defaultCount);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                    break;
                }

                case MLOperatorAttributeType::StringArray:
                {
                    std::vector<std::string> defaultVals(defaultVal->second->strings, defaultVal->second->strings + defaultCount);
                    schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                    break;
                }

                case MLOperatorAttributeTypeTensor:
                    // Tensor is too complex to express a default value. Default checking is done by the operator code.
                    __fallthrough;

                default:
                    ML_CHECK_BOOL(false);
                    break;
            }

            // Remove the default attribute from the map, to later ensure defaults matched attributes
            defaultAttributes.erase(attribute.name);
        }
    }

    ML_CHECK_BOOL(defaultAttributes.empty());
}

// Convert a schema from the ABI to ONNX type
onnx::OpSchema AbiCustomRegistry::ConvertOpSchema(
    _In_z_ const char* domain,
    const MLOperatorSchemaDescription& abiSchema,
    IMLOperatorTypeInferrer* typeInferrer,
    IMLOperatorShapeInferrer* shapeInferrer
    )
{
    // Set the op schema name, domain, and version
    onnx::OpSchema schema(abiSchema.name, "", 0);
    schema.SetDomain(domain);
    schema.SinceVersion(abiSchema.operatorSetVersionAtLastChange);

    // ONNX fails if using an empty string for edge names, although their names don't
    // matter for us.
    const char* emptyName = " ";

    // Populate inputs
    for (uint32_t inputIndex = 0; inputIndex < abiSchema.inputCount; ++inputIndex)
    {
        schema.Input(
                inputIndex,
                emptyName,
                "",
                ConvertFormalParameterType(abiSchema.inputs[inputIndex]),
                ConvertFormalParameterOption(abiSchema.inputs[inputIndex].options));
    }

    // Populate outputs
    for (uint32_t outputIndex = 0; outputIndex < abiSchema.outputCount; ++outputIndex)
    {
        schema.Output(
                outputIndex,
                emptyName,
                "",
                ConvertFormalParameterType(abiSchema.outputs[outputIndex]),
                ConvertFormalParameterOption(abiSchema.outputs[outputIndex].options));
    }

    // Populate type constraints
    for (uint32_t constraintIndex = 0; constraintIndex < abiSchema.typeConstraintCount; ++constraintIndex)
    {
        schema.TypeConstraint(
                abiSchema.typeConstraints[constraintIndex].typeLabel,
                ConvertTypeConstraintTypes(abiSchema.typeConstraints[constraintIndex]),
                "");
    }

    // Set attribute defaults
    SetAttributesAndDefaults(schema, abiSchema);

    // Set an inferencing method
    if (shapeInferrer || typeInferrer)
    {
        ComPtr<IMLOperatorShapeInferrer> shapeInferrerCapture = shapeInferrer;
        ComPtr<IMLOperatorTypeInferrer> typeInferrerCapture = typeInferrer;

        schema.TypeAndShapeInferenceFunction([=](onnx::InferenceContext& ctx)
        {
            // Constant CPU inputs cannot currently be specified through the public ABI for schema registration.
            gsl::span<const uint32_t> requiredConstantCpuInputs;

            onnxruntime::OpNodeProtoHelper<onnx::InferenceContext> nodeInfo(&ctx);
            ComPtr<MLSchemaInferenceContext> abiContext = wil::MakeOrThrow<MLSchemaInferenceContext>(&nodeInfo, &ctx, requiredConstantCpuInputs);

            // Do type inference
            if (typeInferrerCapture)
            {
                THROW_IF_FAILED(typeInferrerCapture->InferOutputTypes(abiContext.Get()));
            }

            // Do shape inference if all input tensor shapes are known
            if (shapeInferrerCapture && InputTensorShapesDefinedOnNode(nodeInfo))
            {
                THROW_IF_FAILED(shapeInferrerCapture->InferOutputShapes(abiContext.Get()));
            }

            abiContext->Close();
        });
    }

    return schema;
}

HRESULT STDMETHODCALLTYPE AbiCustomRegistry::RegisterOperatorSetSchema(
    const MLOperatorSetId* opSetId,
    int baseline_version,
    const MLOperatorSchemaDescription* const* schema,
    uint32_t schemaCount,
    _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept try
{
    std::vector<onnx::OpSchema> schemaVector;
    schemaVector.reserve(schemaCount);

    // Convert schema to ONNX types and accumulate them in a vector
    for (uint32_t i = 0; i < schemaCount; ++i)
    {
        schemaVector.emplace_back(ConvertOpSchema(opSetId->domain, *schema[i], typeInferrer, shapeInferrer));
    }

    // Multiple registries are used to avoid having different versions of the same domain in a single
    // registry, which Lotus doesn't support.
    auto registryKey = std::pair<int, int>(baseline_version, opSetId->version);
    auto registryIter = m_customRegistryOpsetVerMap.find(registryKey);
    if (registryIter == m_customRegistryOpsetVerMap.end())
    {
        m_customRegistryOpsetVerMap[registryKey] = std::make_shared<onnxruntime::CustomRegistry>();
    }

    // Register the operator set with Lotus
    // TODO - Split apart multiple op-sets with a common domain into multiple registries, as required by Lotus
    // for correct lookup (Bug 4662).
    THROW_IF_NOT_OK(m_customRegistryOpsetVerMap[registryKey]->RegisterOpSet(
        schemaVector, 
        opSetId->domain, 
        baseline_version, 
        opSetId->version));

    return S_OK;
}
CATCH_RETURN();

// Convert the list of attribute defaults in a kernel registration into a 
// map of AttributeValue entries, which own their own memory
AttributeMap AbiCustomRegistry::GetDefaultAttributes(
    const MLOperatorKernelDescription* opKernel
    )
{
    AttributeMap ret;

    for (uint32_t i = 0; i < opKernel->defaultAttributeCount; ++i)
    {
        const MLOperatorAttributeNameValue &apiAttr = opKernel->defaultAttributes[i];
        AttributeValue attr;

        attr.type = apiAttr.type;
        switch(apiAttr.type)
        {
        case MLOperatorAttributeType::Float:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::FloatArray:
            attr.floats.assign(&apiAttr.floats[0], apiAttr.floats + apiAttr.valueCount);
            attr.floats.assign(&apiAttr.floats[0], apiAttr.floats + apiAttr.valueCount);
            break;

        case MLOperatorAttributeType::String:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::StringArray:
            attr.strings.assign(&apiAttr.strings[0], &apiAttr.strings[apiAttr.valueCount]);
            break;

        case MLOperatorAttributeType::Int:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::IntArray:
            attr.ints.assign(&apiAttr.ints[0], &apiAttr.ints[apiAttr.valueCount]);
            break;

        case MLOperatorAttributeTypeTensor:
            // Tensor is too complex to express a default value. Default checking is done by the operator code.
            __fallthrough;

        default:
            THROW_HR(E_INVALIDARG);
        }

        ret[apiAttr.name] = attr;
    }

    return ret;
}

HRESULT STDMETHODCALLTYPE AbiCustomRegistry::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel,
    IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept 
{
    return RegisterOperatorKernel(opKernel, operatorKernelFactory, shapeInferrer, nullptr, false, false, false);
}

HRESULT STDMETHODCALLTYPE AbiCustomRegistry::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel,
    IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer,
    _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
    bool isInternalOperator,
    bool canAliasFirstInput,
    bool supportsGraph,
    const uint32_t* requiredInputCountForGraph,
    bool supportedWith64BitTensorsVia32BitStrides,
    bool supportedWith64BitTensorsVia32BitStridesFromAnyEp,
    bool prefer64BitTensorsDirectly,
    bool support64BitTensorsViaEmulation,
    _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs,
    uint32_t constantCpuInputCount) const noexcept try
{

    // Verify that invalid flags are not passed
    if ((opKernel->options & ~MLOperatorKernelOptions::AllowDynamicInputShapes) !=
            MLOperatorKernelOptions::None)
    {
        return E_INVALIDARG;
    }

    // Translate flags
    bool requiresInputShapesAtCreation = (opKernel->options & MLOperatorKernelOptions::AllowDynamicInputShapes) == MLOperatorKernelOptions::None;
    bool requiresOutputShapesAtCreation = !!shapeInferrer;

    // Verify allowed combinations of flags are used
    if (!requiresInputShapesAtCreation && requiresOutputShapesAtCreation)
    {
        return E_INVALIDARG;
    }
    
    const char* providerType = nullptr;
    if (opKernel->executionOptions != 0)
    {
        return E_INVALIDARG;        
    }    
    
    if (opKernel->executionType == MLOperatorExecutionType::Cpu)
    {
        providerType = onnxruntime::kCpuExecutionProvider;
    }
    else if (opKernel->executionType == MLOperatorExecutionType::D3D12)
    {
        providerType = onnxruntime::kDmlExecutionProvider;
    }
    else
    {
        return E_INVALIDARG;
    }

    // Set the name, domain, version, and provider
    onnxruntime::KernelDefBuilder builder;
    builder.SetName(opKernel->name);
    builder.SetDomain(opKernel->domain)
            .SinceVersion(opKernel->minimumOperatorSetVersion)
            .Provider(providerType);

    std::string_view name(opKernel->name);
    if (name == "MemcpyToHost")
    {
        builder.OutputMemoryType(::OrtMemType::OrtMemTypeCPUOutput, 0);
    }
    else if (name == "MemcpyFromHost")
    {
        builder.InputMemoryType(::OrtMemType::OrtMemTypeCPUInput, 0);
    }
        
    std::vector<uint32_t> constantCpuInputCapture;
    constantCpuInputCapture.assign(requiredConstantCpuInputs, requiredConstantCpuInputs + constantCpuInputCount);

    for (uint32_t inputIndex : constantCpuInputCapture)
    {
        builder.InputMemoryType(::OrtMemType::OrtMemTypeCPUInput, inputIndex);
    }

    if (canAliasFirstInput)
    {
        builder.Alias(0, 0);
    }

    // Set type constraints
    for (uint32_t i = 0; i < opKernel->typeConstraintCount; ++i)
    {
        std::vector<onnxruntime::MLDataType> types;
        types.reserve(opKernel->typeConstraints[i].allowedTypeCount);

        for (uint32_t j = 0; j < opKernel->typeConstraints[i].allowedTypeCount; ++j)
        {
            // TODO - handle non-tensor types
            if (opKernel->typeConstraints[i].allowedTypes[j].edgeType != MLOperatorEdgeType::Tensor)
            {
                THROW_IF_FAILED(E_NOTIMPL);
            }

            types.push_back(ToTensorDataType(opKernel->typeConstraints[i].allowedTypes[j].tensorDataType));
        }

        builder.TypeConstraint(opKernel->typeConstraints[i].typeLabel, types);
    }

    ComPtr<IMLOperatorKernelFactory> kernelFactoryCapture = operatorKernelFactory;
    ComPtr<IMLOperatorShapeInferrer> shapeInferrerCapture = shapeInferrer;
    AttributeMap defaultAttributesCapture = GetDefaultAttributes(opKernel);

    auto lotusKernelCreateFn = [
        kernelFactoryCapture,
        requiresInputShapesAtCreation,
        requiresOutputShapesAtCreation,
        isInternalOperator,
        constantCpuInputCapture,
        shapeInferrerCapture,
        defaultAttributesCapture
        ](const onnxruntime::OpKernelInfo& info) -> onnxruntime::OpKernel*
        {
            return new AbiOpKernel(
                    kernelFactoryCapture.Get(),
                    info,
                    requiresInputShapesAtCreation,
                    requiresOutputShapesAtCreation,
                    isInternalOperator,
                    constantCpuInputCapture,
                    shapeInferrerCapture.Get(),
                    &defaultAttributesCapture);
        };

    onnxruntime::KernelCreateInfo create_info(builder.Build(), lotusKernelCreateFn);
    onnxruntime::KernelDef* kernelDef = create_info.kernel_def.get();

    if (isInternalOperator)
    {
        auto regInfo = std::make_shared<InternalRegistrationInfo>();
        regInfo->requiredConstantCpuInputs = constantCpuInputCapture;
        regInfo->supportedWith64BitTensorsVia32BitStrides = supportedWith64BitTensorsVia32BitStrides;
        regInfo->supportedWith64BitTensorsVia32BitStridesFromAnyEp = supportedWith64BitTensorsVia32BitStridesFromAnyEp;
        regInfo->prefer64BitTensorsDirectly = prefer64BitTensorsDirectly;
        regInfo->support64BitTensorsViaEmulation = support64BitTensorsViaEmulation;

        // Only internal operators support usage in DML graphs
        if (supportsGraph)
        {
            GraphNodeFactoryRegistration graphReg;
            graphReg.factory = 
                [kernelFactoryCapture,
                requiresInputShapesAtCreation,
                requiresOutputShapesAtCreation,
                shapeInferrerCapture,
                defaultAttributesCapture,
                constantCpuInputCapture](const onnxruntime::Node& node, MLOperatorTensorGetter& constantInputGetter, const void* executionHandle, DmlGraphNodeCreateInfo* graphNodeCreateInfo)
                {
                    onnxruntime::ProtoHelperNodeContext nodeContext(node);
                    onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> protoHelper(&nodeContext);
                
                    // Use the same list of required constant inputs for the shape inferrer and the kernel.
                    EdgeShapes outputShapes;
                    InferAndVerifyOutputSizes(node, &defaultAttributesCapture, shapeInferrerCapture.Get(), constantCpuInputCapture, constantInputGetter, nullptr, outputShapes);

                    // Create the kernel while allowing input shape and output shape queries according to options
                    ComPtr<DmlGraphOpKernelInfoWrapper> kernelInfoWrapper = wil::MakeOrThrow<DmlGraphOpKernelInfoWrapper>(
                            &protoHelper,
                            executionHandle,
                            true,
                            &outputShapes,
                            &defaultAttributesCapture,
                            graphNodeCreateInfo,
                            constantCpuInputCapture,
                            constantInputGetter);

                    Microsoft::WRL::ComPtr<IMLOperatorKernel> kernel;
                    THROW_IF_FAILED(kernelFactoryCapture->CreateKernel(kernelInfoWrapper.Get(), kernel.GetAddressOf()));
                    kernelInfoWrapper->Close();
                };

            if (requiredInputCountForGraph)
            {
                graphReg.requiredInputCount = *requiredInputCountForGraph;
            }

            regInfo->graphNodeFactoryRegistration = graphReg;
        }

        if (supportQuery)
        {
            ComPtr<IMLOperatorSupportQueryPrivate> supportQueryCapture = supportQuery;

            regInfo->supportQuery = [supportQueryCapture, defaultAttributesCapture](const onnxruntime::Node& node)
            {
                onnxruntime::ProtoHelperNodeContext nodeContext(node);
                onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> protoHelper(&nodeContext);
                              
                // Create the kernel while allowing input shape and output shape queries according to options
                ComPtr<MLSupportQueryContext> supportContext = wil::MakeOrThrow<MLSupportQueryContext>(
                        &protoHelper,
                        &defaultAttributesCapture);

                BOOL bSupported = FALSE;
                THROW_IF_FAILED(supportQueryCapture->QuerySupport(supportContext.Get(), &bSupported));
                return !!bSupported;
            };
        }

        THROW_IF_NOT_OK(m_kernelRegistry->RegisterCustomKernel(create_info));
        (*m_internalRegInfoMap)[kernelDef] = regInfo;
    }
    else
    {
        // Currently unsupported for external operators
        if (canAliasFirstInput ||
            supportsGraph ||
            requiredInputCountForGraph ||
            requiredConstantCpuInputs ||
            supportedWith64BitTensorsVia32BitStrides ||
            supportedWith64BitTensorsVia32BitStridesFromAnyEp ||
            prefer64BitTensorsDirectly ||
            support64BitTensorsViaEmulation)
        {
            THROW_HR(E_INVALIDARG);
        }

        //
        // For backward compatibility, this does not propagate errors for external operators
        m_kernelRegistry->RegisterCustomKernel(create_info);
    }

    return S_OK;
}
CATCH_RETURN();

}