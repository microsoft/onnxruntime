// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "precomp.h"



void ConvertToUnorderedMap(
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* list,
    /*out*/ std::unordered_map<std::string_view, uint32_t>& nameToIndexMap)
{
    for (uint32_t index = 0; index < list->size(); index++)
    {
        const flatbuffers::String* name = list->GetAsString(index);
        if (name->size() == 0)
        {
            continue;
        }
        nameToIndexMap[name->string_view()] = index;
    }
}

template <typename EdgeType> void PopulateEdges(
    const uint32_t nodeIndex,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* edgeNames,
    const std::unordered_map<std::string_view, uint32_t>& edgeNameToIndexMap,
    /*out*/ std::vector<EdgeType>& edges,
    /*out*/ std::vector<DmlIntermediateSerializedGraphEdge>& intermediateEdges,
    /*out*/ std::unordered_map<std::string_view, NodeIndex>& intermediateEdgeToNodeIndexMap)
{
    for (flatbuffers::uoffset_t edgeIndex = 0; edgeIndex < edgeNames->size(); edgeIndex++)
    {
        const flatbuffers::String* edgeName = edgeNames->Get(edgeIndex);
        if (edgeName->size() == 0)
        {
            continue;
        }
        // edge can be graphInput or graphOutput
        if (edgeNameToIndexMap.find(edgeName->string_view()) != edgeNameToIndexMap.end())
        {
            EdgeType edge = {};
            edge.Name = edgeName->str();

            if constexpr (std::is_same_v<EdgeType, DmlInputSerializedGraphEdge>)
            {
                edge.GraphInputIndex = edgeNameToIndexMap.at(edgeName->string_view());
                edge.ToNodeIndex = nodeIndex;
                edge.ToNodeInputIndex = edgeIndex;
            }
            else if constexpr (std::is_same_v<EdgeType, DmlOutputSerializedGraphEdge>)
            {
                edge.GraphOutputIndex = edgeNameToIndexMap.at(edgeName->string_view());
                edge.FromNodeIndex = nodeIndex;
                edge.FromNodeOutputIndex = edgeIndex;
            }

            edges.push_back(edge);
        }
        // edge is intermediate edge
        else 
        {
            if constexpr (std::is_same_v<EdgeType, DmlInputSerializedGraphEdge>)
            {
                auto& intermediateEdgeNodeIndex = intermediateEdgeToNodeIndexMap[edgeName->string_view()];
                DmlIntermediateSerializedGraphEdge intermediateEdge = {};
                intermediateEdge.Name = edgeName->str();
                intermediateEdge.FromNodeIndex = intermediateEdgeNodeIndex.nodeIndex;
                intermediateEdge.FromNodeOutputIndex = intermediateEdgeNodeIndex.nodeOutputIndex;
                intermediateEdge.ToNodeIndex = nodeIndex;
                intermediateEdge.ToNodeInputIndex = edgeIndex;
                intermediateEdges.push_back(intermediateEdge);
            }
            else if constexpr (std::is_same_v<EdgeType, DmlOutputSerializedGraphEdge>)
            {
                intermediateEdgeToNodeIndexMap[edgeName->string_view()] = {nodeIndex, edgeIndex};
            }
        }
    }
}

OperatorFieldTypes::TensorDesc CreateBufferTensorDesc(
    const dml::ir::DmlBufferTensorDesc* tensorDesc,
    const bool isConstantTensor = false)
{
    DmlBufferTensorDesc bufferTensorDesc = {};
    bufferTensorDesc.dataType = ApiTraits::StringifyHelpers::FromString<DML_TENSOR_DATA_TYPE>(tensorDesc->dataType()->c_str());
    if (isConstantTensor)
    {
        bufferTensorDesc.flags = DML_TENSOR_FLAG_OWNED_BY_DML;
    }
    bufferTensorDesc.sizes = std::vector<uint32_t>(tensorDesc->sizes()->begin(), tensorDesc->sizes()->end());
    if (flatbuffers::IsFieldPresent(tensorDesc, dml::ir::DmlBufferTensorDesc::VT_STRIDES))
    {
        bufferTensorDesc.strides = std::vector<uint32_t>(tensorDesc->strides()->begin(), tensorDesc->strides()->end());
    }
    bufferTensorDesc.totalTensorSizeInBytes = tensorDesc->totalTensorSizeInBytes();
    return bufferTensorDesc;
}

OperatorFieldVariant CreateActivation(
    const dml::ir::operatorFieldTypes::Activation* activationDesc)
{
    DML_OPERATOR_TYPE activationOperatorType = ApiTraits::StringifyHelpers::FromString<DML_OPERATOR_TYPE>(activationDesc->type()->c_str());
    const DML_OPERATOR_SCHEMA& activationSchema = SchemaHelpers::GetSchema(activationOperatorType);
    std::vector<OperatorField> activationOperatorFields(activationSchema.FieldCount);
    uint32_t attributeIndex = 0;

    for (uint32_t fieldIndex = 0; fieldIndex < activationSchema.FieldCount; fieldIndex++)
    {
        const DML_SCHEMA_FIELD* schemaField = &activationSchema.Fields[fieldIndex];
        OperatorFieldVariant field;
        switch (schemaField->Kind)
        {
            case DML_SCHEMA_FIELD_KIND_INPUT_TENSOR:
            case DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR:
            {
                if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
                {
                    field = OperatorFieldTypes::TensorDesc();
                }
                else if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
                {
                    field = OperatorFieldTypes::TensorDescArray();
                }
                break;
            }
            case DML_SCHEMA_FIELD_KIND_ATTRIBUTE:
            {
                const dml::ir::operatorFieldTypes::AttributeDesc* attributeDesc = 
                    attributeIndex >= activationDesc->attributes()->size() ?
                    nullptr : 
                    activationDesc->attributes()->Get(attributeIndex++);
                field = CreateAttribute(schemaField, attributeDesc);
                break;
            }
        }

        activationOperatorFields[fieldIndex] = OperatorField(schemaField, std::move(field));
    }

    return AbstractOperatorDesc(&activationSchema, std::move(activationOperatorFields));
}

OperatorFieldVariant CreateActivations(
    const dml::ir::operatorFieldTypes::ActivationArray* activationDescs)
{
    std::vector<AbstractOperatorDesc> activations;
    for (uint32_t index = 0; index < static_cast<uint32_t>(activationDescs->data()->size()); index++)
    {
        OperatorFieldVariant activation = CreateActivation(activationDescs->data()->Get(index));
        activations.push_back(std::get<OperatorFieldTypes::FusedActivationOperatorDesc>(activation).value());
    }
    return activations;
}

OperatorFieldVariant CreateAttribute(
    const DML_SCHEMA_FIELD* schemaField,
    const dml::ir::operatorFieldTypes::AttributeDesc* attributeDesc)
{
    switch (schemaField->Type)
    {
        case DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC:
        {
            return attributeDesc != nullptr && attributeDesc->val_as_Activation() != nullptr ?  
                CreateActivation(attributeDesc->val_as_Activation()) : 
                OperatorFieldTypes::FusedActivationOperatorDesc();
        }
        case DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC_ARRAY:
        {
            return attributeDesc != nullptr && attributeDesc->val_as_ActivationArray() != nullptr ?  
                CreateActivations(attributeDesc->val_as_ActivationArray()) : 
                OperatorFieldTypes::FusedActivationOperatorDescArray();
        }
        case DML_SCHEMA_FIELD_TYPE_UINT:
        {
            OperatorFieldTypes::UInt data;
            if (attributeDesc != nullptr)
            {
                data = attributeDesc->val_as_UInt32()->data();
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_UINT64:
        {
            OperatorFieldTypes::UInt64 data;
            if (attributeDesc != nullptr)
            {
                data = attributeDesc->val_as_UInt64()->data();
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_INT:
        {
            OperatorFieldTypes::Int data;
            if (attributeDesc != nullptr)
            {
                data = attributeDesc->val_as_Int32()->data();
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_FLOAT:
        {
            OperatorFieldTypes::Float data;
            if (attributeDesc != nullptr)
            {
                data = attributeDesc->val_as_Float32()->data();
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_UINT_ARRAY:
        {
            OperatorFieldTypes::UIntArray data;
            if (attributeDesc != nullptr)
            {
                data = std::vector<uint32_t>(attributeDesc->val_as_UIntArray()->data()->begin(), attributeDesc->val_as_UIntArray()->data()->end());
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_INT_ARRAY:
        {
            OperatorFieldTypes::IntArray data;
            if (attributeDesc != nullptr)
            {
                data = std::vector<int32_t>(attributeDesc->val_as_IntArray()->data()->begin(), attributeDesc->val_as_IntArray()->data()->end());
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_FLOAT_ARRAY:
        {
            OperatorFieldTypes::FloatArray data;
            if (attributeDesc != nullptr)
            {
                data = std::vector<float>(attributeDesc->val_as_FloatArray()->data()->begin(), attributeDesc->val_as_FloatArray()->data()->end());
            }
            return data;
        }	
        case DML_SCHEMA_FIELD_TYPE_SCALE_BIAS:
        {
            OperatorFieldTypes::ScaleBias scaleBias;
            const dml::ir::operatorFieldTypes::ScaleBias* scaleBiasAttribute = attributeDesc->val_as_ScaleBias();
            if (scaleBiasAttribute != nullptr)
            {
                scaleBias = {scaleBiasAttribute->scale(), scaleBiasAttribute->bias()};
            }
            return scaleBias;
        }
        case DML_SCHEMA_FIELD_TYPE_SIZE_2D:
        {
            OperatorFieldTypes::Size2D size2d = {};
            if (attributeDesc != nullptr)
            {
                size2d.Height = attributeDesc->val_as_Size2D()->height();
                size2d.Width = attributeDesc->val_as_Size2D()->width();
            }
            return size2d;
        }
        case DML_SCHEMA_FIELD_TYPE_SCALAR_UNION:
        {
            DML_SCALAR_UNION scalarUnion;
            if (attributeDesc != nullptr)
            {
                const dml::ir::operatorFieldTypes::ByteArray* byteArr = attributeDesc->val_as_ScalarUnionData()->data_as_ByteArray();
                std::copy(byteArr->data()->begin(), byteArr->data()->end(), scalarUnion.Bytes);
            }
            return scalarUnion;
        }
        case DML_SCHEMA_FIELD_TYPE_BOOL:
        {
            OperatorFieldTypes::Bool data;
            if (attributeDesc != nullptr)
            {
                data = attributeDesc->val_as_Bool()->data();
            }
            return data;
        }
        default:
        {
            THROW_HR(E_INVALIDARG);
        }
    }
}

AbstractOperatorDesc CreateAbstractOperatorDesc(
    const dml::ir::OperatorNodeDesc* flatbufferOperatorNodeDesc,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* nodeInputNames,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* nodeOutputNames,
    const std::unordered_set<std::string_view>& constantInputs)
{
    DML_OPERATOR_TYPE type = ApiTraits::StringifyHelpers::FromString<DML_OPERATOR_TYPE>(flatbufferOperatorNodeDesc->type()->c_str());
    const DML_OPERATOR_SCHEMA& schema = SchemaHelpers::GetSchema(type);
    std::vector<OperatorField> operatorFields(schema.FieldCount);
    
    auto inputNameItr = nodeInputNames->begin();
    uint32_t inputTensorDescIndex = 0;
    
    uint32_t outputTensorDescIndex = 0;
    auto outputNameItr = nodeOutputNames->begin();

    uint32_t attributeIndex = 0;
    

    for (uint32_t fieldIndex = 0; fieldIndex < schema.FieldCount; fieldIndex++)
    {
        const DML_SCHEMA_FIELD* schemaField = &schema.Fields[fieldIndex];
        
        OperatorFieldVariant field;
        switch (schemaField->Kind)
        {
            case DML_SCHEMA_FIELD_KIND_INPUT_TENSOR:
            {
                if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
                {
                    const flatbuffers::String* inputName = *inputNameItr;
                    inputNameItr++;
                    if (inputName->size() == 0)
                    {
                        field = OperatorFieldTypes::TensorDesc();
                        break;
                    }
                    bool isConstantTensor = !constantInputs.empty() && constantInputs.find(inputName->c_str()) != constantInputs.end();

                    const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->inputs()->Get(inputTensorDescIndex++);
                    field = CreateBufferTensorDesc(tensorDesc, isConstantTensor);
                }
                else if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
                {
                    std::vector<DmlBufferTensorDesc> tensors;
                    while (inputTensorDescIndex < static_cast<uint32_t>(flatbufferOperatorNodeDesc->inputs()->size()))
                    {
                        const flatbuffers::String* inputName = *inputNameItr;
                        inputNameItr++;
                        bool isConstantTensor = !constantInputs.empty() && constantInputs.find(inputName->c_str()) != constantInputs.end();
                        
                        const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->inputs()->Get(inputTensorDescIndex++);
                        tensors.push_back(CreateBufferTensorDesc(tensorDesc, isConstantTensor).value());
                    }
                    field = tensors;
                }
                break;
            }
            case DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR:
            {
                if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
                {
                    const flatbuffers::String* outputName = *outputNameItr;
                    outputNameItr++;

                    if (outputName->size() == 0)
                    {
                        field = OperatorFieldTypes::TensorDesc();
                        break;
                    }

                    const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->outputs()->Get(outputTensorDescIndex++);
                    field = CreateBufferTensorDesc(tensorDesc);
                }
                else if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
                {
                    std::vector<DmlBufferTensorDesc> tensors;
                    while (outputTensorDescIndex < static_cast<uint32_t>(flatbufferOperatorNodeDesc->outputs()->size()))
                    {
                        const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->outputs()->Get(outputTensorDescIndex++);
                        tensors.push_back(CreateBufferTensorDesc(tensorDesc).value());
                    }
                    field = tensors;
                }
                break;
            }
            case DML_SCHEMA_FIELD_KIND_ATTRIBUTE:
            {
                const dml::ir::operatorFieldTypes::AttributeDesc* attributeDesc = 
                    attributeIndex >= flatbufferOperatorNodeDesc->attributes()->size() ?
                    nullptr : 
                    flatbufferOperatorNodeDesc->attributes()->Get(attributeIndex++);
                field = CreateAttribute(schemaField, attributeDesc);
                break;
            }
        }

        operatorFields[fieldIndex] = OperatorField(schemaField, std::move(field));
    }

    return AbstractOperatorDesc(&schema, std::move(operatorFields));
}

DmlSerializedGraphDesc DeserializeDmlGraph(
    const uint8_t* flatbufferGraphDescBlob,
    /*out*/ std::vector<std::unique_ptr<std::byte[]>>& rawData)
{
    const dml::ir::DmlGraphDesc* flatbufferGraphDesc = dml::ir::GetDmlGraphDesc(flatbufferGraphDescBlob);
    
    std::unordered_map<std::string_view, uint32_t> graphInputEdgeToIndexMap;
    std::unordered_map<std::string_view, uint32_t> graphOutputEdgeToIndexMap;
    ConvertToUnorderedMap(flatbufferGraphDesc->graphInputNames(), graphInputEdgeToIndexMap);
    ConvertToUnorderedMap(flatbufferGraphDesc->graphOutputNames(), graphOutputEdgeToIndexMap);
    
    std::unordered_map<std::string_view, NodeIndex> intermediateEdgeToNodeIndexMap;
    std::unordered_set<std::string_view> constantInputs;

    std::vector<DmlSerializedGraphNode> nodes(flatbufferGraphDesc->nodes()->size());
    std::vector<DmlInputSerializedGraphEdge> inputEdges;
    std::vector<DmlOutputSerializedGraphEdge> outputEdges;
    std::vector<DmlIntermediateSerializedGraphEdge> intermediateEdges;

    for (uint32_t nodeIndex = 0; nodeIndex < flatbufferGraphDesc->nodes()->size(); nodeIndex++)
    {
        const dml::ir::DmlGraphNode* flatbufferNode = flatbufferGraphDesc->nodes()->Get(nodeIndex);

        PopulateEdges<DmlInputSerializedGraphEdge>(
            nodeIndex,
            flatbufferNode->inputNames(),
            graphInputEdgeToIndexMap,
            inputEdges,
            intermediateEdges,
            intermediateEdgeToNodeIndexMap);
        PopulateEdges<DmlOutputSerializedGraphEdge>(
            nodeIndex,
            flatbufferNode->outputNames(),
            graphOutputEdgeToIndexMap,
            outputEdges,
            intermediateEdges,
            intermediateEdgeToNodeIndexMap);

        DmlSerializedGraphNode node = {};
        node.Name = flatbufferNode->name()->c_str();
        if (flatbufferNode->desc_type() == dml::ir::NodeDesc_ConstantNodeDesc)
        {
            const dml::ir::ConstantNodeDesc* flatbufferConstantNode = flatbufferNode->desc_as_ConstantNodeDesc();
            if (flatbufferConstantNode->data_type() == dml::ir::ConstantNodeDescDetail_ConstantName)
            {
                ConstantName constantNode = {flatbufferConstantNode->data_as_ConstantName()->name()->c_str()};
                node.Desc = constantNode;
            }
            else if (flatbufferConstantNode->data_type() == dml::ir::ConstantNodeDescDetail_ConstantRawData)
            {
                
                uint32_t rawDataSize = flatbufferConstantNode->data_as_ConstantRawData()->data()->size();
                rawData.push_back(std::make_unique<std::byte[]>(rawDataSize));
                std::transform(
                    flatbufferConstantNode->data_as_ConstantRawData()->data()->begin(),
                    flatbufferConstantNode->data_as_ConstantRawData()->data()->end(),
                    rawData.back().get(),
                    [](uint8_t b) {return static_cast<std::byte>(b);});

                ConstantData constantData = {};
                constantData.dataSize = rawDataSize;
                constantData.data = rawData.back().get();
                node.Desc = constantData;
            }

            // output of this node will part of constantInputs list
            for (uint32_t outputIndex = 0; outputIndex < flatbufferNode->outputNames()->size(); outputIndex++)
            {
                constantInputs.insert(flatbufferNode->outputNames()->Get(outputIndex)->c_str());
            }

        }
        else if (flatbufferNode->desc_type() == dml::ir::NodeDesc::NodeDesc_OperatorNodeDesc)
        {
            // convert dml::ir::OperatorNodeDesc to AbstractOperatorDesc
            const dml::ir::OperatorNodeDesc* flatbufferOperatorNodeDesc = flatbufferNode->desc_as_OperatorNodeDesc();
            node.Desc = CreateAbstractOperatorDesc(
                flatbufferOperatorNodeDesc,
                flatbufferNode->inputNames(),
                flatbufferNode->outputNames(),
                constantInputs);
        }

        nodes[nodeIndex] = node;
    }

    DmlSerializedGraphDesc graphDesc;
    graphDesc.InputCount = flatbufferGraphDesc->graphInputNames()->size();
    graphDesc.OutputCount = flatbufferGraphDesc->graphOutputNames()->size();
    graphDesc.InputEdges = std::move(inputEdges);
    graphDesc.IntermediateEdges = std::move(intermediateEdges);
    graphDesc.OutputEdges = std::move(outputEdges);
    graphDesc.Nodes = std::move(nodes);
    return graphDesc;	
}
