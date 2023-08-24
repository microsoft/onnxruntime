// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "precomp.h"

flatbuffers::Offset<dml::ir::operatorFieldTypes::Activation> serializeActivation(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& activationOperatorDesc)
{
    std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> attributeDescs;
    SerializeAttributeDescs(builder, activationOperatorDesc, attributeDescs);
    
    flatbuffers::Offset<dml::ir::operatorFieldTypes::Activation> offset = dml::ir::operatorFieldTypes::CreateActivationDirect(
        builder,
        activationOperatorDesc.schema->OperatorName,
        &attributeDescs);
    return offset;
}

void SerializeAttributeDescs(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& operatorDesc,
    /*out*/ std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>& attributeDescs)
{
    for (const OperatorField& field : operatorDesc.fields)
    {
        if (field.GetSchema()->Kind == DML_SCHEMA_FIELD_KIND_INPUT_TENSOR || field.GetSchema()->Kind == DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR)
        {
            continue;
        }

        flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc> offset;

        if (std::holds_alternative<OperatorFieldTypes::FusedActivationOperatorDesc>(field.GetData()))
        {
            const OperatorFieldTypes::FusedActivationOperatorDesc& fusedActivation = field.AsFusedActivationOperatorDesc();
            if (!fusedActivation.has_value())
            {
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    nullptr,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_Activation);
            }
            else
            {
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    field.GetSchema()->Name,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_Activation,
                    serializeActivation(builder, fusedActivation.value()).Union());
            }
        }
        else if (std::holds_alternative<OperatorFieldTypes::FusedActivationOperatorDescArray>(field.GetData()))
        {
            const OperatorFieldTypes::FusedActivationOperatorDescArray& fusedActivations = 
                field.AsFusedActivationOperatorDescArray();
            if (!fusedActivations.has_value())
            {
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    nullptr,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_ActivationArray);
            }
            else
            {
                std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::Activation>> fbActivations;

                for (AbstractOperatorDesc activationOpDesc : fusedActivations.value())
                {
                    flatbuffers::Offset<dml::ir::operatorFieldTypes::Activation> fbActivation = 
                        serializeActivation(builder, activationOpDesc);
                    fbActivations.push_back(fbActivation);
                }

                flatbuffers::Offset<dml::ir::operatorFieldTypes::ActivationArray> activationOffset = 
                    dml::ir::operatorFieldTypes::CreateActivationArrayDirect(builder, &fbActivations);
                
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    field.GetSchema()->Name,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_ActivationArray,
                    activationOffset.Union());
            }
        }
        else if (std::holds_alternative<OperatorFieldTypes::UInt>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_UInt32,
                builder.CreateStruct(dml::ir::operatorFieldTypes::UInt32(field.AsUInt())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::UInt64>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_UInt64,
                builder.CreateStruct(dml::ir::operatorFieldTypes::UInt64(field.AsUInt64())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::Int>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_Int32,
                builder.CreateStruct(dml::ir::operatorFieldTypes::Int32(field.AsInt())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::Float>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_Float32,
                builder.CreateStruct(dml::ir::operatorFieldTypes::Float32(field.AsFloat())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::UIntArray>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_UIntArray,
                dml::ir::operatorFieldTypes::CreateUIntArray(builder, builder.CreateVector(field.AsUIntArray())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::IntArray>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_IntArray,
                dml::ir::operatorFieldTypes::CreateIntArray(builder, builder.CreateVector(field.AsIntArray())).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::FloatArray>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_FloatArray,
                dml::ir::operatorFieldTypes::CreateFloatArray(builder, builder.CreateVector(field.AsFloatArray())).Union());
        }	
        else if (std::holds_alternative<OperatorFieldTypes::ScaleBias>(field.GetData()))
        {
            const OperatorFieldTypes::ScaleBias& scaleBias = field.AsScaleBias();
            if (!scaleBias.has_value())
            {
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    nullptr,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_ScaleBias);
            }
            else
            {
                dml::ir::operatorFieldTypes::ScaleBias fbScaleBias(scaleBias.value().Scale, scaleBias.value().Bias);
                offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                    builder,
                    field.GetSchema()->Name,
                    dml::ir::operatorFieldTypes::AttributeFieldVariant_ScaleBias,
                    builder.CreateStruct(fbScaleBias).Union());
            }
        }
        else if (std::holds_alternative<OperatorFieldTypes::Size2D>(field.GetData()))
        {
            const DML_SIZE_2D size2d = field.AsSize2D();
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_Size2D,
                builder.CreateStruct(dml::ir::operatorFieldTypes::Size2D(size2d.Width, size2d.Height)).Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::ScalarUnion>(field.GetData()))
        {
            OperatorFieldTypes::ScalarUnion scalarUnion = field.AsScalarUnion();
            dml::ir::operatorFieldTypes::ByteArray byteArr;
            for (uint32_t index = 0; index < static_cast<uint32_t>(sizeof(scalarUnion.Bytes)); index++)
            {
                byteArr.mutable_data()->Mutate(index, scalarUnion.Bytes[index]);
            }

            flatbuffers::Offset<dml::ir::operatorFieldTypes::ScalarUnionData> scalarUnionOffset = 
                dml::ir::operatorFieldTypes::CreateScalarUnionData(
                    builder,
                    dml::ir::operatorFieldTypes::ScalarVariant_ByteArray,
                    builder.CreateStruct(byteArr).Union());

            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_ScalarUnionData,
                scalarUnionOffset.Union());
        }
        else if (std::holds_alternative<OperatorFieldTypes::Bool>(field.GetData()))
        {
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_Bool,
                builder.CreateStruct(dml::ir::operatorFieldTypes::Bool(field.AsBool())).Union());
        }
        else
        {
            continue;
        }
        
        attributeDescs.push_back(offset);
    }
}

flatbuffers::Offset<dml::ir::DmlBufferTensorDesc> SerializeDmlTensorDesc(
    flatbuffers::FlatBufferBuilder& builder,
    const DmlBufferTensorDesc* tensorDesc)
{
    const std::vector<uint32_t> *strides = nullptr;
    if (tensorDesc->strides.has_value())
    {
        strides = &tensorDesc->strides.value();
    }
    
    flatbuffers::Offset<dml::ir::DmlBufferTensorDesc> offset = dml::ir::CreateDmlBufferTensorDescDirect(
        builder,
        ApiTraits::StringifyHelpers::ToString(tensorDesc->dataType),
        &tensorDesc->sizes,
        strides,
        tensorDesc->totalTensorSizeInBytes);
    return offset;
}

flatbuffers::Offset<void> SerializeOperatorNodeDesc(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& operatorDesc)
{
    const DML_OPERATOR_SCHEMA* operatorSchema = operatorDesc.schema;

    std::vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> inputTensorDescs;
    std::vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> outputTensorDescs;
    
    for (const DmlBufferTensorDesc* tensorDesc : operatorDesc.GetInputTensors())
    {
        if (tensorDesc == nullptr)
        {
            continue;
        }
        flatbuffers::Offset<dml::ir::DmlBufferTensorDesc> serializedDmlTensorDesc = SerializeDmlTensorDesc(builder, tensorDesc);
        inputTensorDescs.push_back(serializedDmlTensorDesc);
    }
    
    for (const DmlBufferTensorDesc* tensorDesc : operatorDesc.GetOutputTensors())
    {
        if (tensorDesc == nullptr)
        {
            continue;
        }
        flatbuffers::Offset<dml::ir::DmlBufferTensorDesc> serializedDmlTensorDesc = SerializeDmlTensorDesc(builder, tensorDesc);
        outputTensorDescs.push_back(serializedDmlTensorDesc);
    }
    
    std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> attributeDescs;
    SerializeAttributeDescs(builder, operatorDesc, attributeDescs);
    
    flatbuffers::Offset<dml::ir::OperatorNodeDesc> offset = dml::ir::CreateOperatorNodeDesc(
        builder,
        builder.CreateString(operatorSchema->OperatorName),
        builder.CreateVector(inputTensorDescs),
        builder.CreateVector(outputTensorDescs),
        builder.CreateVector(attributeDescs));
    return offset.Union();
}

flatbuffers::Offset<void> SerializeConstantNodeDesc(
    flatbuffers::FlatBufferBuilder& builder,
    const DmlSerializedGraphNodeConstantVariant& constantNodeDesc)
{
    flatbuffers::Offset<dml::ir::ConstantNodeDesc> offset;
    
    if (std::holds_alternative<ConstantName>(constantNodeDesc))
    {
        auto& constantName = std::get<ConstantName>(constantNodeDesc);
        flatbuffers::Offset<dml::ir::ConstantName> constantNameOffset = dml::ir::CreateConstantName(
            builder, 
            builder.CreateString(constantName.name));

        offset = dml::ir::CreateConstantNodeDesc(
            builder,
            dml::ir::ConstantNodeDescDetail_ConstantName,
            constantNameOffset.Union());
    }
    else
    {
        auto& constantData = std::get<ConstantData>(constantNodeDesc);
        std::vector<uint8_t> rawBytes;
        std::transform(constantData.data, constantData.data + constantData.dataSize, std::back_inserter(rawBytes), [](std::byte b) {return static_cast<uint8_t>(b); });
        flatbuffers::Offset<dml::ir::ConstantRawData> constantDataOffset = dml::ir::CreateConstantRawDataDirect(
            builder,
            &rawBytes);

        offset = dml::ir::CreateConstantNodeDesc(
            builder,
            dml::ir::ConstantNodeDescDetail_ConstantRawData,
            constantDataOffset.Union());
    }
    
    return offset.Union();
}

flatbuffers::Offset<dml::ir::DmlGraphNode> SerializeNode(
    flatbuffers::FlatBufferBuilder& builder,
    const DmlSerializedGraphNode& graphNode,
    const std::vector<flatbuffers::Offset<flatbuffers::String>>& nodeInputNames,
    const std::vector<flatbuffers::Offset<flatbuffers::String>>& nodeOutputNames)
{
    flatbuffers::Offset<dml::ir::DmlGraphNode> offset;
    if (std::holds_alternative<AbstractOperatorDesc>(graphNode.Desc))
    {
        auto& operatorNode = std::get<AbstractOperatorDesc>(graphNode.Desc);
        offset = dml::ir::CreateDmlGraphNode(
            builder,
            dml::ir::NodeDesc_OperatorNodeDesc,
            SerializeOperatorNodeDesc(builder, operatorNode),
            builder.CreateString(graphNode.Name),
            builder.CreateVector(nodeInputNames),
            builder.CreateVector(nodeOutputNames));
    }
    else
    {
        auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(graphNode.Desc);
        offset = dml::ir::CreateDmlGraphNode(
            builder,
            dml::ir::NodeDesc_ConstantNodeDesc,
            SerializeConstantNodeDesc(builder, constantNodeVariant),
            builder.CreateString(graphNode.Name),
            builder.CreateVector(nodeInputNames),
            builder.CreateVector(nodeOutputNames));
    }
    return offset;
}

template <typename Edge>
void PopulateEdgeIndexToNameMap(
    const std::vector<Edge>& edges,
    flatbuffers::FlatBufferBuilder& builder,
    /*out*/ std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>>& edgeIndexToNameMap,
    /*out*/ uint32_t& maxIndex)
{
    // If an edge has a name, then use that. Otherwise assign default 
    // name to all unassigned inputs.
    for (auto& edge : edges)
    {
        uint32_t index;
        if constexpr (std::is_same_v<Edge, DmlInputSerializedGraphEdge>)
        {
            index = edge.GraphInputIndex;
        }
        else if constexpr (std::is_same_v<Edge, DmlOutputSerializedGraphEdge>)
        {
            index = edge.GraphOutputIndex;
        }
        maxIndex = std::max(index, maxIndex);
        if (edge.Name.empty())
        {
            THROW_IF_FAILED(E_INVALIDARG);
        }
        if (edgeIndexToNameMap.find(index) == edgeIndexToNameMap.end())
        {
            edgeIndexToNameMap[index] = builder.CreateString(edge.Name);
        }
    }
}

void VerifyNodeInputOutputMaxIndex(
    const std::vector<DmlSerializedGraphNode>& nodes,
    std::vector<uint32_t>& nodeInputCounts,
    std::vector<uint32_t>& nodeOutputCounts)
{
    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(nodes.size()); nodeIndex++)
    {
        auto& node = nodes[nodeIndex];
        if (std::holds_alternative<AbstractOperatorDesc>(node.Desc))
        {
            auto& operatorNode = std::get<AbstractOperatorDesc>(node.Desc);
            nodeInputCounts[nodeIndex] = std::max(
                nodeInputCounts[nodeIndex], 
                static_cast<uint32_t>(operatorNode.GetInputTensors().size()));

            nodeOutputCounts[nodeIndex] = std::max(
                nodeOutputCounts[nodeIndex], 
                static_cast<uint32_t>(operatorNode.GetOutputTensors().size()));
        }
    }
}

template <typename Edge>
void PopulateNodeInputOutputMaxIndex(
    const std::vector<Edge>& edges,
    /*out*/std::vector<uint32_t>& maxInputIndexForNodes,
    /*out*/std::vector<uint32_t>& maxOutputIndexForNodes)
{
    for (auto& edge : edges)
    {
        if constexpr (std::is_same<Edge, DmlInputSerializedGraphEdge>::value)
        {
            maxInputIndexForNodes[edge.ToNodeIndex] = std::max(maxInputIndexForNodes[edge.ToNodeIndex], edge.ToNodeInputIndex + 1);
        }
        else if constexpr (std::is_same<Edge, DmlOutputSerializedGraphEdge>::value)
        {
            maxOutputIndexForNodes[edge.FromNodeIndex] = std::max(maxInputIndexForNodes[edge.FromNodeIndex], edge.FromNodeOutputIndex + 1);
        }
        else if constexpr (std::is_same<Edge, DmlIntermediateSerializedGraphEdge>::value)
        {
            maxInputIndexForNodes[edge.ToNodeIndex] = std::max(maxInputIndexForNodes[edge.ToNodeIndex], edge.ToNodeInputIndex + 1);
            maxOutputIndexForNodes[edge.FromNodeIndex] = std::max(maxInputIndexForNodes[edge.FromNodeIndex], edge.FromNodeOutputIndex + 1);
        }
    }
}

void PopulateNodeInputOutputNames(
    flatbuffers::FlatBufferBuilder& builder,
    const DmlSerializedGraphDesc& graphDesc,
    const std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>>& graphInputIndexToNameMap,
    const std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>>& graphOutputIndexToNameMap,
    /*out*/std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>>& nodeToInputNames, 
    /*out*/std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>>& nodeToOutputNames)
{
    for (auto& edge : graphDesc.InputEdges)
    {
        nodeToInputNames[edge.ToNodeIndex][edge.ToNodeInputIndex] = graphInputIndexToNameMap.at(edge.GraphInputIndex);
    }

    for (auto& edge : graphDesc.OutputEdges)
    {
        nodeToOutputNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = graphOutputIndexToNameMap.at(edge.GraphOutputIndex);
    }

    std::unordered_map<uint32_t, std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>>> intermediateEdgeNames;
    for (uint32_t edgeIndex = 0; edgeIndex < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); edgeIndex++)
    {
        auto& edge = graphDesc.IntermediateEdges[edgeIndex];
        flatbuffers::Offset<flatbuffers::String> edgeName;
        
        if (intermediateEdgeNames.find(edge.FromNodeIndex) != intermediateEdgeNames.end() &&
            intermediateEdgeNames[edge.FromNodeIndex].find(edge.FromNodeOutputIndex) != intermediateEdgeNames[edge.FromNodeIndex].end())
        {
            edgeName = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
        }
        else
        {
            if (edge.Name.empty())
            {
                THROW_IF_FAILED(E_INVALIDARG);
            }
            edgeName = builder.CreateString(edge.Name.c_str());
            intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = edgeName;
        }
        nodeToInputNames[edge.ToNodeIndex][edge.ToNodeInputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
        nodeToOutputNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
    }
}

flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc)
{

    flatbuffers::FlatBufferBuilder builder(1024);
    if (graphDesc.Nodes.empty())
    {
        return builder.Release();
    }

    // Set graphInputIndexToNameMap
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphInputIndexToNameMap;
    uint32_t maxInputIndex = 0;
    PopulateEdgeIndexToNameMap<DmlInputSerializedGraphEdge>(
        graphDesc.InputEdges,
        builder,
        graphInputIndexToNameMap,
        maxInputIndex);

    // Set graphOutputIndexToNameMap
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphOutputIndexToNameMap;
    uint32_t maxOutputIndex = 0;
    PopulateEdgeIndexToNameMap<DmlOutputSerializedGraphEdge>(
        graphDesc.OutputEdges,
        builder,
        graphOutputIndexToNameMap,
        maxOutputIndex);

    // Calculate number of input/output for each operator to allocate
    // appropriate amount of memory for each node to store input/output names.
    std::vector<uint32_t> nodeInputCounts(graphDesc.Nodes.size(), 0);
    std::vector<uint32_t> nodeOutputCounts(graphDesc.Nodes.size(), 0);
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToInputNames(graphDesc.Nodes.size());
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToOutputNames(graphDesc.Nodes.size());
    
    // we need this for constant nodes
    PopulateNodeInputOutputMaxIndex<DmlInputSerializedGraphEdge>(graphDesc.InputEdges, nodeInputCounts, nodeOutputCounts);
    PopulateNodeInputOutputMaxIndex<DmlOutputSerializedGraphEdge>(graphDesc.OutputEdges, nodeInputCounts, nodeOutputCounts);
    PopulateNodeInputOutputMaxIndex<DmlIntermediateSerializedGraphEdge>(graphDesc.IntermediateEdges, nodeInputCounts, nodeOutputCounts);
    VerifyNodeInputOutputMaxIndex(graphDesc.Nodes, nodeInputCounts, nodeOutputCounts);

    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(graphDesc.Nodes.size()); nodeIndex++)
    {
        nodeToInputNames[nodeIndex] = std::vector<flatbuffers::Offset<flatbuffers::String>>(nodeInputCounts[nodeIndex], builder.CreateString(nullptr, 0));
        nodeToOutputNames[nodeIndex] = std::vector<flatbuffers::Offset<flatbuffers::String>>(nodeOutputCounts[nodeIndex], builder.CreateString(nullptr, 0));
    }
    PopulateNodeInputOutputNames(builder, graphDesc, graphInputIndexToNameMap, graphOutputIndexToNameMap, nodeToInputNames, nodeToOutputNames);

    // Create flatbuffer node objects
    std::vector<flatbuffers::Offset<dml::ir::DmlGraphNode>> nodes(graphDesc.Nodes.size());
    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(graphDesc.Nodes.size()); nodeIndex++)
    {
        nodes[nodeIndex] = SerializeNode(
                            builder,
                            graphDesc.Nodes[nodeIndex],
                            nodeToInputNames[nodeIndex],
                            nodeToOutputNames[nodeIndex]);
    }

    std::vector<flatbuffers::Offset<flatbuffers::String>> graphInputNames(graphDesc.InputCount, builder.CreateString(nullptr, 0));
    std::vector<flatbuffers::Offset<flatbuffers::String>> graphOutputNames(graphDesc.OutputCount, builder.CreateString(nullptr, 0));
    
    for (const auto& [key, value] : graphInputIndexToNameMap)
    {
        graphInputNames[key] = value;
    }

    for (const auto& [key, value] : graphOutputIndexToNameMap)
    {
        graphOutputNames[key] = value;
    }

    flatbuffers::Offset<dml::ir::DmlGraphDesc> dmlGraphDescOffset = dml::ir::CreateDmlGraphDescDirect(
        builder,
        &nodes,
        &graphInputNames,
        &graphOutputNames);
    builder.Finish(dmlGraphDescOffset);
    return builder.Release();
}
