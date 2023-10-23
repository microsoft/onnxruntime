// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "precomp.h"


OperatorFieldVariant CreateAttribute(
    const DML_SCHEMA_FIELD* schemaField,
    const dml::ir::operatorFieldTypes::AttributeDesc* attributeDesc);

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
                data.assign(attributeDesc->val_as_UIntArray()->data()->begin(), attributeDesc->val_as_UIntArray()->data()->end());
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_INT_ARRAY:
        {
            OperatorFieldTypes::IntArray data;
            if (attributeDesc != nullptr)
            {
                data.assign(attributeDesc->val_as_IntArray()->data()->begin(), attributeDesc->val_as_IntArray()->data()->end());
            }
            return data;
        }
        case DML_SCHEMA_FIELD_TYPE_FLOAT_ARRAY:
        {
            OperatorFieldTypes::FloatArray data;
            if (attributeDesc != nullptr)
            {
                data.assign(attributeDesc->val_as_FloatArray()->data()->begin(), attributeDesc->val_as_FloatArray()->data()->end());
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
            throw std::invalid_argument("Invalid attribute type.");
        }
    }
}

OperatorFieldTypes::TensorDesc CreateBufferTensorDesc(
    const dml::ir::DmlBufferTensorDesc* tensorDesc,
    const bool isLargeConstantTensor = false)
{
    DmlBufferTensorDesc bufferTensorDesc = {};
    bufferTensorDesc.dataType = ApiTraits::StringifyHelpers::FromString<DML_TENSOR_DATA_TYPE>(tensorDesc->dataType()->c_str());
    if (isLargeConstantTensor)
    {
        bufferTensorDesc.flags = DML_TENSOR_FLAG_OWNED_BY_DML;
    }
    bufferTensorDesc.sizes.assign(tensorDesc->sizes()->begin(), tensorDesc->sizes()->end());
    if (flatbuffers::IsFieldPresent(tensorDesc, dml::ir::DmlBufferTensorDesc::VT_STRIDES))
    {
        bufferTensorDesc.strides.emplace(tensorDesc->strides()->begin(), tensorDesc->strides()->end());
    }
    bufferTensorDesc.totalTensorSizeInBytes = tensorDesc->totalTensorSizeInBytes();
    return bufferTensorDesc;
}

AbstractOperatorDesc CreateAbstractOperatorDesc(
    uint32_t nodeIndex,
    const dml::ir::OperatorNodeDesc* flatbufferOperatorNodeDesc,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* nodeInputNames,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* nodeOutputNames,
    const std::unordered_set<std::string_view>& largeConstantInputs)
{
    DML_OPERATOR_TYPE type = ApiTraits::StringifyHelpers::FromString<DML_OPERATOR_TYPE>(flatbufferOperatorNodeDesc->type()->c_str());
    if (type == DML_OPERATOR_INVALID)
    {
        throw std::invalid_argument("Graph operator node at index: " + std::to_string(nodeIndex) +
                                    " either has empty or invalid operator type.");
    }
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
                if (inputNameItr == nodeInputNames->end())
                {
                    throw std::invalid_argument("Missing input names for node at index: " + std::to_string(nodeIndex));
                }

                if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
                {
                    const flatbuffers::String* inputName = *inputNameItr;
                    inputNameItr++;
                    if (inputName->size() == 0)
                    {
                        field = OperatorFieldTypes::TensorDesc();
                        break;
                    }
                    bool isLargeConstantTensor = !largeConstantInputs.empty() && largeConstantInputs.find(inputName->c_str()) != largeConstantInputs.end();

                    if (flatbufferOperatorNodeDesc->inputs()->size() <= inputTensorDescIndex)
                    {
                        throw std::invalid_argument("Expecting at least " + std::to_string(inputTensorDescIndex + 1) + 
                                                    " input tensor desc for graph operator node at index: " + std::to_string(nodeIndex));
                    }
                    const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->inputs()->Get(inputTensorDescIndex++);
                    field = CreateBufferTensorDesc(tensorDesc, isLargeConstantTensor);
                }
                else if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
                {
                    std::vector<DmlBufferTensorDesc> tensors;
                    while (inputTensorDescIndex < static_cast<uint32_t>(flatbufferOperatorNodeDesc->inputs()->size()))
                    {
                        const flatbuffers::String* inputName = *inputNameItr;
                        inputNameItr++;
                        bool isLargeConstantTensor = !largeConstantInputs.empty() && largeConstantInputs.find(inputName->c_str()) != largeConstantInputs.end();
                        
                        if (flatbufferOperatorNodeDesc->inputs()->size() <= inputTensorDescIndex)
                        {
                            throw std::invalid_argument("Expecting at least " + std::to_string(inputTensorDescIndex + 1) + 
                                                        " input tensor desc for graph operator node at index: " + std::to_string(nodeIndex));
                        }
                        const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->inputs()->Get(inputTensorDescIndex++);
                        tensors.push_back(CreateBufferTensorDesc(tensorDesc, isLargeConstantTensor).value());
                    }
                    field = tensors;
                }
                break;
            }
            case DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR:
            {
                if (outputNameItr == nodeOutputNames->end())
                {
                    throw std::invalid_argument("Missing output names for node at index: " + std::to_string(nodeIndex));
                }

                if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
                {
                    const flatbuffers::String* outputName = *outputNameItr;
                    outputNameItr++;

                    if (outputName->size() == 0)
                    {
                        field = OperatorFieldTypes::TensorDesc();
                        break;
                    }

                    if (flatbufferOperatorNodeDesc->outputs()->size() <= outputTensorDescIndex)
                    {
                        throw std::invalid_argument("Expecting at least " + std::to_string(outputTensorDescIndex + 1) + 
                                                    " output tensor desc for graph operator node at index: " + std::to_string(nodeIndex));
                    }
                    const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->outputs()->Get(outputTensorDescIndex++);
                    field = CreateBufferTensorDesc(tensorDesc);
                }
                else if (schemaField->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
                {
                    std::vector<DmlBufferTensorDesc> tensors;
                    while (outputTensorDescIndex < static_cast<uint32_t>(flatbufferOperatorNodeDesc->outputs()->size()))
                    {
                        if (flatbufferOperatorNodeDesc->outputs()->size() <= outputTensorDescIndex)
                        {
                            throw std::invalid_argument("Expecting at least " + std::to_string(outputTensorDescIndex + 1) + 
                                                        " output tensor desc for graph operator node at index: " + std::to_string(nodeIndex));
                        }
                        const dml::ir::DmlBufferTensorDesc* tensorDesc = flatbufferOperatorNodeDesc->outputs()->Get(outputTensorDescIndex++);
                        tensors.push_back(CreateBufferTensorDesc(tensorDesc).value());
                    }
                    field = tensors;
                }
                break;
            }
            case DML_SCHEMA_FIELD_KIND_ATTRIBUTE:
            {
                if (flatbufferOperatorNodeDesc->attributes()->size() <= attributeIndex)
                {
                    throw std::invalid_argument("Expecting at least " + std::to_string(attributeIndex + 1) + 
                                                " attributes for graph operator node at index: " + std::to_string(nodeIndex));
                }
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

std::unordered_map<std::string_view, uint32_t> ConvertToEdgeNameToIndexMap(
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* list)
{
    std::unordered_map<std::string_view, uint32_t> nameToIndexMap;
    for (uint32_t index = 0; index < list->size(); index++)
    {
        const flatbuffers::String* name = list->GetAsString(index);
        if (name->size() == 0)
        {
            continue;
        }
        nameToIndexMap[name->string_view()] = index;
    }
    return nameToIndexMap; // NRVO will automatically move it. no need to use std::move
}

template <typename EdgeType> void PopulateEdges(
    const uint32_t nodeIndex,
    const ::flatbuffers::Vector<::flatbuffers::Offset<::flatbuffers::String>>* edgeNames,
    const std::unordered_map<std::string_view, uint32_t>& edgeNameToIndexMap,
    /*out*/ std::vector<EdgeType>& edges,
    /*out*/ std::vector<DmlIntermediateSerializedGraphEdge>& intermediateEdges,
    /*out*/ std::unordered_map<std::string_view, NodeIndex>& edgeToOutgoingNodeIndexMap)
{
    for (flatbuffers::uoffset_t edgeIndex = 0; edgeIndex < edgeNames->size(); edgeIndex++)
    {
        const flatbuffers::String* edgeName = edgeNames->Get(edgeIndex);
        if (edgeName->size() == 0)
        {
            // This must be optional input/output
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
                edgeToOutgoingNodeIndexMap[edgeName->string_view()] = {nodeIndex, edgeIndex};
            }

            edges.push_back(edge);
        }
        // edge is intermediate edge
        else 
        {
            if constexpr (std::is_same_v<EdgeType, DmlInputSerializedGraphEdge>)
            {
                if (edgeToOutgoingNodeIndexMap.find(edgeName->string_view()) == edgeToOutgoingNodeIndexMap.end())
                {
                    throw std::range_error("Neither there is any graph input with name " + edgeName->str() + 
                                           " nor there is any node which has " + edgeName->str() + " as one of the output.");
                }
                auto& intermediateEdgeNodeIndex = edgeToOutgoingNodeIndexMap[edgeName->string_view()];
                DmlIntermediateSerializedGraphEdge intermediateEdge = {};
                intermediateEdge.Name = edgeName->str();
                intermediateEdge.FromNodeIndex = intermediateEdgeNodeIndex.nodeIndex;
                intermediateEdge.FromNodeOutputIndex = intermediateEdgeNodeIndex.nodeOutputIndex;
                intermediateEdge.ToNodeIndex = nodeIndex;
                intermediateEdge.ToNodeInputIndex = edgeIndex;
                intermediateEdges.push_back(std::move(intermediateEdge));
            }
            else if constexpr (std::is_same_v<EdgeType, DmlOutputSerializedGraphEdge>)
            {
                std::string strr = edgeName->str();
                size_t strrSize = strr.size();
                edgeToOutgoingNodeIndexMap[edgeName->string_view()] = {nodeIndex, edgeIndex};
            }
        }
    }
}

/*
* - Handling of empty optional input/output/attibute for non-constant node:
*   input/output
*   - <DmlGraphNode.inputNames> and <DmlGraphNode.outputNames> will have an null entry
*      but the actual OperatorNodeDesc variant's <OperatorNodeDesc.inputs> 
*      and <OperatorNodeDesc.outputs> will not have any entry.
*   attribute
*   - <OperatorNodeDesc.attributes> will have null entry
*/
DmlSerializedGraphDesc DeserializeDmlGraph(
    const uint8_t* flatbufferGraphDescBlob,
    /*out*/ std::vector<std::unique_ptr<std::byte[]>>& rawData)
{
    if (flatbufferGraphDescBlob == nullptr)
    {
        throw std::invalid_argument("Given pointer to flatbuffer blob is null");
    }
    const dml::ir::DmlGraphDesc* flatbufferGraphDesc = dml::ir::GetDmlGraphDesc(flatbufferGraphDescBlob);
    
    std::unordered_map<std::string_view, uint32_t> graphInputEdgeToIndexMap = ConvertToEdgeNameToIndexMap(flatbufferGraphDesc->graphInputNames());
    std::unordered_map<std::string_view, uint32_t> graphOutputEdgeToIndexMap = ConvertToEdgeNameToIndexMap(flatbufferGraphDesc->graphOutputNames());
    
    std::unordered_map<std::string_view, NodeIndex> edgeToOutgoingNodeIndexMap;
    std::unordered_set<std::string_view> largeConstantInputs;

    std::vector<DmlSerializedGraphNode> nodes(flatbufferGraphDesc->nodes()->size());
    std::vector<DmlInputSerializedGraphEdge> inputEdges;
    std::vector<DmlOutputSerializedGraphEdge> outputEdges;
    std::vector<DmlIntermediateSerializedGraphEdge> intermediateEdges;

    // Iterator on output edges of all nodes first because
    // <edgeToOutgoingNodeIndexMap> needs to be filled first.
    for (uint32_t nodeIndex = 0; nodeIndex < flatbufferGraphDesc->nodes()->size(); nodeIndex++)
    {
        const dml::ir::DmlGraphNode* flatbufferNode = flatbufferGraphDesc->nodes()->Get(nodeIndex);

        PopulateEdges<DmlOutputSerializedGraphEdge>(
            nodeIndex,
            flatbufferNode->outputNames(),
            graphOutputEdgeToIndexMap,
            outputEdges,
            intermediateEdges,
            edgeToOutgoingNodeIndexMap);
    }

    for (uint32_t nodeIndex = 0; nodeIndex < flatbufferGraphDesc->nodes()->size(); nodeIndex++)
    {
        const dml::ir::DmlGraphNode* flatbufferNode = flatbufferGraphDesc->nodes()->Get(nodeIndex);

        PopulateEdges<DmlInputSerializedGraphEdge>(
            nodeIndex,
            flatbufferNode->inputNames(),
            graphInputEdgeToIndexMap,
            inputEdges,
            intermediateEdges,
            edgeToOutgoingNodeIndexMap);

        DmlSerializedGraphNode node = {};
        if (flatbufferNode->name()->size() == 0)
        {
            throw std::invalid_argument("Graph node at index: " + std::to_string(nodeIndex) + " doesn't have any name");
        }
        node.Name = flatbufferNode->name()->c_str();

        if (flatbufferNode->desc_type() == dml::ir::NodeDesc_ConstantNodeDesc)
        {
            const dml::ir::ConstantNodeDesc* flatbufferConstantNode = flatbufferNode->desc_as_ConstantNodeDesc();
            if (flatbufferConstantNode->data_type() == dml::ir::ConstantNodeDescDetail_ConstantName)
            {
                if (flatbufferConstantNode->data_as_ConstantName()->name()->size() == 0)
                {
                    throw std::invalid_argument("Constant node at index: " + std::to_string(nodeIndex) + 
                                                " doesn't have constant data name.");
                }

                ConstantName constantNode = {flatbufferConstantNode->data_as_ConstantName()->name()->c_str()};
                node.Desc = constantNode;
                
                // output of this node will part of constantInputs list
                for (uint32_t outputIndex = 0; outputIndex < flatbufferNode->outputNames()->size(); outputIndex++)
                {
                    largeConstantInputs.insert(flatbufferNode->outputNames()->Get(outputIndex)->c_str());
                }
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


        }
        else if (flatbufferNode->desc_type() == dml::ir::NodeDesc::NodeDesc_OperatorNodeDesc)
        {
            // convert dml::ir::OperatorNodeDesc to AbstractOperatorDesc
            const dml::ir::OperatorNodeDesc* flatbufferOperatorNodeDesc = flatbufferNode->desc_as_OperatorNodeDesc();
            node.Desc = CreateAbstractOperatorDesc(
                nodeIndex,
                flatbufferOperatorNodeDesc,
                flatbufferNode->inputNames(),
                flatbufferNode->outputNames(),
                largeConstantInputs);
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
