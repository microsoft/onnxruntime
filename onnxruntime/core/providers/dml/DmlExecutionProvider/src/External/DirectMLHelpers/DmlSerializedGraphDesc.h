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
    byte* data;
    uint32_t size;
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
    UINT GraphInputIndex; 
    UINT ToNodeIndex; 
    UINT ToNodeInputIndex; 
    std::string Name; 
};

struct DmlOutputSerializedGraphEdge
{
    UINT FromNodeIndex; 
    UINT FromNodeOutputIndex; 
    UINT GraphOutputIndex; 
    std::string Name; 
};

struct DmlIntermediateSerializedGraphEdge
{
    UINT FromNodeIndex; 
    UINT FromNodeOutputIndex; 
    UINT ToNodeIndex; 
    UINT ToNodeInputIndex; 
    std::string Name; 
};

struct DmlSerializedGraphDesc
{
    uint32_t InputCount;
    uint32_t OutputCount;
    // nodes must be present in topological order for deserialization to work
    // because while creating a intermediate edge during deserialization, node (from
    // which given intermediate edge is outputting) must be visited before than the node
    // (to which given intermediate edge is inputting)
    std::vector<DmlSerializedGraphNode> Nodes;
    std::vector<DmlInputSerializedGraphEdge> InputEdges;
    std::vector<DmlOutputSerializedGraphEdge> OutputEdges;
    std::vector<DmlIntermediateSerializedGraphEdge> IntermediateEdges;
};
