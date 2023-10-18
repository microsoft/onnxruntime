// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include "precomp.h"

template <typename T>
T* ReadAs(uint8_t* base, size_t byteOffset)
{
    return reinterpret_cast<T*>(base + byteOffset);
}

void SerializeAttributeDescs(
    flatbuffers::FlatBufferBuilder& builder,
    const AbstractOperatorDesc& operatorDesc,
    /*out*/ std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>& attributeDescs);

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
        if (field.GetSchema()->Kind == DML_SCHEMA_FIELD_KIND_INPUT_TENSOR || 
            field.GetSchema()->Kind == DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR)
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
    uint32_t nodeIndex,
    const DmlSerializedGraphNodeConstantVariant& constantNodeDesc)
{
    flatbuffers::Offset<dml::ir::ConstantNodeDesc> offset;
    
    if (std::holds_alternative<ConstantName>(constantNodeDesc))
    {
        auto& constantName = std::get<ConstantName>(constantNodeDesc);
        if (constantName.name.empty())
        {
            throw std::invalid_argument("Graph constant node at index:" + std::to_string(nodeIndex) +
                                        " doesn't have the constant data name.");
        }

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
        std::transform(constantData.data, constantData.data + constantData.dataSize, 
                       std::back_inserter(rawBytes), [](std::byte b) {return static_cast<uint8_t>(b); });
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
    const uint32_t nodeIndex,
    const DmlSerializedGraphNode& graphNode,
    const std::vector<flatbuffers::Offset<flatbuffers::String>>& nodeInputNames,
    const std::vector<flatbuffers::Offset<flatbuffers::String>>& nodeOutputNames)
{
    if (graphNode.Name.empty())
    {        
        throw std::invalid_argument("Graph node at index:" + std::to_string(nodeIndex) + 
                                    " does not have any name.");
    }

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
            SerializeConstantNodeDesc(builder, nodeIndex, constantNodeVariant),
            builder.CreateString(graphNode.Name),
            builder.CreateVector(nodeInputNames),
            builder.CreateVector(nodeOutputNames));
    }
    return offset;
}

/*
* validates input/output edges and throws exception if an edge 
* does not have a name or if an edge has more than 1 names.
*/
template <typename Edge>
std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> ConvertToEdgeIndexToNameMap(
    const std::vector<Edge>& edges,
    flatbuffers::FlatBufferBuilder& builder)
{
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> edgeIndexToNameMap;
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
        
        if (edge.Name.empty())
        {
            throw std::invalid_argument("Graph input or output edge at index " + std::to_string(index) + " does not have name.");
        }

        if (edgeIndexToNameMap.find(index) != edgeIndexToNameMap.end())
        {
            flatbuffers::String* edgeName = ReadAs<flatbuffers::String>(
                builder.GetCurrentBufferPointer(),
                builder.GetSize() - edgeIndexToNameMap[index].o);
            if (edge.Name != edgeName->str())
            {
                throw std::invalid_argument("Graph input or output edge at index " + std::to_string(index) + " has more than 1 names.");
            }
        }

        edgeIndexToNameMap[index] = builder.CreateString(edge.Name);
    }
    return edgeIndexToNameMap; // NRVO will automatically move it. no need to use std::move
}

void PopulateNonConstantNodeInputOutputCount(
    const std::vector<DmlSerializedGraphNode>& nodes,
    /*out*/ std::vector<uint32_t>& nodeInputCounts,
    /*out*/ std::vector<uint32_t>& nodeOutputCounts)
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

void PopulateConstantNodeInputOutputCount(
    const std::vector<DmlIntermediateSerializedGraphEdge>& edges,
    /*out*/std::vector<uint32_t>& maxInputIndexForNodes,
    /*out*/std::vector<uint32_t>& maxOutputIndexForNodes)
{
    for (auto& edge : edges)
    {
        maxInputIndexForNodes[edge.ToNodeIndex] = std::max(maxInputIndexForNodes[edge.ToNodeIndex], edge.ToNodeInputIndex + 1);
        maxOutputIndexForNodes[edge.FromNodeIndex] = std::max(maxOutputIndexForNodes[edge.FromNodeIndex], edge.FromNodeOutputIndex + 1);
    }
}

/*
* validates intermediate edge and throws exception if an edge 
* does not have a name or if an edge has more than 1 names.
*/
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
        if (edge.Name.empty())
        {
            throw std::invalid_argument(
                    "Graph intermediate edge from nodeIndex:" + std::to_string(edge.FromNodeIndex) + 
                    " & nodeOutputIndex:" + std::to_string(edge.FromNodeOutputIndex) + " doesn't have name.");
        }
        
        if (intermediateEdgeNames.find(edge.FromNodeIndex) != intermediateEdgeNames.end() &&
            intermediateEdgeNames[edge.FromNodeIndex].find(edge.FromNodeOutputIndex) != intermediateEdgeNames[edge.FromNodeIndex].end())
        {
            flatbuffers::Offset edgeNameOffset = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
            flatbuffers::String* edgeName = ReadAs<flatbuffers::String>(
                builder.GetCurrentBufferPointer(),
                builder.GetSize() - edgeNameOffset.o);

            if (edgeName->str() != edge.Name)
            {
                throw std::invalid_argument(
                    "Graph intermediate edge from nodeIndex:" + std::to_string(edge.FromNodeIndex) + 
                    " & nodeOutputIndex:" + std::to_string(edge.FromNodeOutputIndex) + " has more than 1 names.");
            }
        }
        else
        {
            intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = builder.CreateString(edge.Name.c_str());
        }
        nodeToInputNames[edge.ToNodeIndex][edge.ToNodeInputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
        nodeToOutputNames[edge.FromNodeIndex][edge.FromNodeOutputIndex] = intermediateEdgeNames[edge.FromNodeIndex][edge.FromNodeOutputIndex];
    }
}


/*
* - If an edge is connected to multiple nodes, then there will be multiple instances 
*   of input or intermediate edges, all with the same name.
* - The input <graphDesc> will be validated incrementally throughout the execution 
*   of the method.
* - Handling of empty optional input/output/attibute for non-constant node:
*   input/output
*   - <DmlGraphNode.inputNames> and <DmlGraphNode.outputNames> will have an null entry
*      but the actual OperatorNodeDesc variant's <OperatorNodeDesc.inputs> 
*      and <OperatorNodeDesc.outputs> will not have any entry.
*   attribute
*   - <OperatorNodeDesc.attributes> will have null entry
*/
flatbuffers::DetachedBuffer SerializeDmlGraph(const DmlSerializedGraphDesc& graphDesc)
{

    flatbuffers::FlatBufferBuilder builder(1024);
    if (graphDesc.Nodes.empty())
    {
        return builder.Release();
    }

    // create input/output edge index to name map
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphInputIndexToNameMap = 
        ConvertToEdgeIndexToNameMap<DmlInputSerializedGraphEdge>(graphDesc.InputEdges, builder);
    std::unordered_map<uint32_t, flatbuffers::Offset<flatbuffers::String>> graphOutputIndexToNameMap = 
        ConvertToEdgeIndexToNameMap<DmlOutputSerializedGraphEdge>(graphDesc.OutputEdges, builder);

    /*
    * - Calculate number of input/output for each operator to allocate
    *   appropriate amount of memory for each node to store input/output names.
    * - Non-constant node's input/output count can be determined by the
    *   AbstractOperatorDesc.
    * - Constant node will only have outgoing edges and those outgoing edges 
    *   will be intermediate edges.
    */
    std::vector<uint32_t> nodeInputCounts(graphDesc.Nodes.size(), 0);
    std::vector<uint32_t> nodeOutputCounts(graphDesc.Nodes.size(), 0);
    PopulateNonConstantNodeInputOutputCount(graphDesc.Nodes, nodeInputCounts, nodeOutputCounts);
    PopulateConstantNodeInputOutputCount(graphDesc.IntermediateEdges, nodeInputCounts, nodeOutputCounts);
    
    // populate node input/output names.
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToInputNames(graphDesc.Nodes.size());
    std::vector<std::vector<flatbuffers::Offset<flatbuffers::String>>> nodeToOutputNames(graphDesc.Nodes.size());
    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(graphDesc.Nodes.size()); nodeIndex++)
    {
        nodeToInputNames[nodeIndex].assign(nodeInputCounts[nodeIndex], builder.CreateString(nullptr, 0));
        nodeToOutputNames[nodeIndex].assign(nodeOutputCounts[nodeIndex], builder.CreateString(nullptr, 0));
    }
    PopulateNodeInputOutputNames(builder, graphDesc, graphInputIndexToNameMap, graphOutputIndexToNameMap, nodeToInputNames, nodeToOutputNames);

    // Create flatbuffer node objects
    std::vector<flatbuffers::Offset<dml::ir::DmlGraphNode>> nodes(graphDesc.Nodes.size());
    for (uint32_t nodeIndex = 0; nodeIndex < static_cast<uint32_t>(graphDesc.Nodes.size()); nodeIndex++)
    {
        nodes[nodeIndex] = SerializeNode(
                            builder,
                            nodeIndex,
                            graphDesc.Nodes[nodeIndex],
                            nodeToInputNames[nodeIndex],
                            nodeToOutputNames[nodeIndex]);
    }

    // Convert to std::vector to create the <dml::ir::DmlGraphDesc> object.
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
