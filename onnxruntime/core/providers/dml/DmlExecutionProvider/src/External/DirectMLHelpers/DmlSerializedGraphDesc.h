//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

struct ConstantName
{
    std::string name;
};

struct ConstantData
{
    std::byte* data;
    uint64_t dataSize;
};

using DmlSerializedGraphNodeConstantVariant = std::variant<
    ConstantName,
    ConstantData
>;

using DmlSerializedGraphNodeDescVariant = std::variant<
    AbstractOperatorDesc,
    DmlSerializedGraphNodeConstantVariant
>;

struct DmlSerializedGraphNode   
{
    DmlSerializedGraphNodeDescVariant Desc;
    std::string Name; 
};

struct DmlInputSerializedGraphEdge
{
    uint32_t GraphInputIndex; 
    uint32_t ToNodeIndex; 
    uint32_t ToNodeInputIndex; 
    std::string Name; 
};

struct DmlOutputSerializedGraphEdge
{
    uint32_t FromNodeIndex; 
    uint32_t FromNodeOutputIndex; 
    uint32_t GraphOutputIndex; 
    std::string Name; 
};

struct DmlIntermediateSerializedGraphEdge
{
    uint32_t FromNodeIndex; 
    uint32_t FromNodeOutputIndex; 
    uint32_t ToNodeIndex; 
    uint32_t ToNodeInputIndex; 
    std::string Name; 
};

struct DmlSerializedGraphDesc
{
    uint32_t InputCount;
    uint32_t OutputCount;
    // Nodes must be present in topological order for deserialization to work
    // because while creating an intermediate edge during deserialization, the node from
    // which given intermediate edge is outputting must be visited before the node
    // to which given intermediate edge is inputting.
    std::vector<DmlSerializedGraphNode> Nodes;
    std::vector<DmlInputSerializedGraphEdge> InputEdges;
    std::vector<DmlOutputSerializedGraphEdge> OutputEdges;
    std::vector<DmlIntermediateSerializedGraphEdge> IntermediateEdges;
};
