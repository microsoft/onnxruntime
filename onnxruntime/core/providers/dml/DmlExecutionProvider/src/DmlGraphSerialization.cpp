// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "precomp.h"

flatbuffers::Offset<void> serializeActivation(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& activationOperatorDesc)
{
    std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> attributeDescs;
    SerializeAttributeDescs(builder, activationOperatorDesc, attributeDescs);
    
    flatbuffers::Offset<dml::ir::operatorFieldTypes::Activation> offset = dml::ir::operatorFieldTypes::CreateActivationDirect(
        builder,
        activationOperatorDesc.schema->OperatorName,
        &attributeDescs);
    return offset.Union();
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
                continue;
            }
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_Activation,
                serializeActivation(builder, fusedActivation.value()));
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
                continue;
            }
            offset = dml::ir::operatorFieldTypes::CreateAttributeDescDirect(
                builder,
                field.GetSchema()->Name,
                dml::ir::operatorFieldTypes::AttributeFieldVariant_ScaleBias,
                builder.CreateStruct(
                    dml::ir::operatorFieldTypes::ScaleBias(scaleBias.value().Scale, scaleBias.value().Bias)).Union());
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
            flatbuffers::Offset<dml::ir::operatorFieldTypes::ScalarUnionData> scalarUnionOffset = 
                dml::ir::operatorFieldTypes::CreateScalarUnionData(
                    builder,
                    dml::ir::operatorFieldTypes::ScalarVariant_ByteArray,
                    builder.CreateVector(field.AsScalarUnion().Bytes, 8).Union());

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
        strides);
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
        // TODO: copy the raw bytes.
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
    const std::string& prefix,
    /*out*/ std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>>& edgeIndexToNameMap)
{
    // If an edge has a name, then use that. Otherwise assign default 
    // name to all unassigned inputs.
    for (auto& edge : edges)
    {
        int index;
        if constexpr (std::is_same_v<Edge, DmlInputSerializedGraphEdge>)
        {
            index = edge.GraphInputIndex;
        }
        else if constexpr (std::is_same_v<Edge, DmlOutputSerializedGraphEdge>)
        {
            index = edge.GraphOutputIndex;
        }
        if (edgeIndexToNameMap.find(index) == edgeIndexToNameMap.end()
            && !edge.Name.empty())
        {
            edgeIndexToNameMap[index] = builder.CreateString(edge.Name);
        }
    }
    /*std::for_each(edgeIndexToNameMap.begin(), edgeIndexToNameMap.end(), [&builder, &prefix, idx = 0](auto& name) mutable {
        if (name.IsNull())
        {
            name = builder.CreateString(prefix + std::to_string(idx++));
        }
    });*/
}

template <typename Edge>
void PopulateNodeInputOutputCount(
    const std::vector<Edge>& edges,
    /*out*/std::vector<uint32_t>& nodeInputCounts,
    /*out*/std::vector<std::unordered_set<uint32_t>>& nodeOutputCounts)
{
    for (auto& edge : edges)
    {
        if constexpr (std::is_same<Edge, DmlInputSerializedGraphEdge>::value)
        {
            nodeInputCounts[edge.ToNodeIndex]++;
        }
        else if constexpr (std::is_same<Edge, DmlOutputSerializedGraphEdge>::value)
        {
            nodeOutputCounts[edge.FromNodeIndex].insert(edge.FromNodeOutputIndex);
        }
        else if constexpr (std::is_same<Edge, DmlIntermediateSerializedGraphEdge>::value)
        {
            nodeInputCounts[edge.ToNodeIndex]++;
            nodeOutputCounts[edge.FromNodeIndex].insert(edge.FromNodeOutputIndex);
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
    for (uint32_t edgeIdx = 0; edgeIdx < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); edgeIdx++)
    {
        auto& edge = graphDesc.IntermediateEdges[edgeIdx];
        flatbuffers::Offset<flatbuffers::String> edgeName;
        
        if (intermediateEdgeNames.find(edge.FromNodeIndex) != intermediateEdgeNames.end() &&
            intermediateEdgeNames[edge.FromNodeIndex].find(edge.FromNodeOutputIndex) != intermediateEdgeNames[edge.FromNodeIndex].end())
        {
            edgeName = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
        }
        else
        {
            edgeName = !edge.Name.empty() ? builder.CreateString(edge.Name.c_str()) : 
                                            builder.CreateString("IntermediateEdge" + std::to_string(edgeIdx));
            intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = edgeName;
        }
        nodeToInputNames[edge.ToNodeIndex][edge.ToNodeInputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
        nodeToOutputNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
    }
}

flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc)
{

    flatbuffers::FlatBufferBuilder builder(1024);
    if (graphDesc.Nodes.size() == 0)
    {
        builder.Release();
    }

    // Set graphInputIndexToNameMap
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphInputIndexToNameMap;
    PopulateEdgeIndexToNameMap<DmlInputSerializedGraphEdge>(graphDesc.InputEdges, builder, "GraphInput", graphInputIndexToNameMap);

    // Set graphOutputIndexToNameMap
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphOutputIndexToNameMap;
    PopulateEdgeIndexToNameMap<DmlOutputSerializedGraphEdge>(graphDesc.OutputEdges, builder, "GraphOutput", graphOutputIndexToNameMap);

    // Calculate number of input/output for each operator to allocate
    // appropriate amount of memory for each node to store input/output names.
    std::vector<uint32_t> nodeInputCounts(graphDesc.Nodes.size());
    std::vector<std::unordered_set<uint32_t>> nodeOutputCounts(graphDesc.Nodes.size());
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToInputNames(graphDesc.Nodes.size());
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToOutputNames(graphDesc.Nodes.size());
    
    PopulateNodeInputOutputCount<DmlInputSerializedGraphEdge>(graphDesc.InputEdges, nodeInputCounts, nodeOutputCounts);
    PopulateNodeInputOutputCount<DmlOutputSerializedGraphEdge>(graphDesc.OutputEdges, nodeInputCounts, nodeOutputCounts);
    PopulateNodeInputOutputCount<DmlIntermediateSerializedGraphEdge>(graphDesc.IntermediateEdges, nodeInputCounts, nodeOutputCounts);
    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(graphDesc.Nodes.size()); nodeIndex++)
    {
        nodeToInputNames[nodeIndex] = std::vector<flatbuffers::Offset<flatbuffers::String>>(nodeInputCounts[nodeIndex]);
        nodeToOutputNames[nodeIndex] = std::vector<flatbuffers::Offset<flatbuffers::String>>(static_cast<uint32_t>(nodeOutputCounts[nodeIndex].size()));
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

    std::vector<flatbuffers::Offset<flatbuffers::String>> graphInputNames;
    std::vector<flatbuffers::Offset<flatbuffers::String>> graphOutputNames;
    
    for (const auto& [key, value] : graphInputIndexToNameMap)
    {
        graphInputNames.push_back(value);
    }

    for (const auto& [key, value] : graphOutputIndexToNameMap)
    {
        graphOutputNames.push_back(value);
    }

    flatbuffers::Offset<dml::ir::DmlGraphDesc> dmlGraphDescOffset = dml::ir::CreateDmlGraphDescDirect(
        builder,
        &nodes,
        &graphInputNames,
        &graphOutputNames);
    /*dml::ir::DmlGraphDescBuilder dmlGraphDescBuilder(builder);
    dmlGraphDescBuilder.add_graphInputNames(builder.CreateVector(graphInputIndexToNameMap));
    dmlGraphDescBuilder.add_graphOutputNames(builder.CreateVector(graphOutputIndexToNameMap));
    dmlGraphDescBuilder.add_nodes(builder.CreateVector(nodes));
    dmlGraphDescBuilder.Finish();*/
    builder.Finish(dmlGraphDescOffset);
    return builder.Release();
}
