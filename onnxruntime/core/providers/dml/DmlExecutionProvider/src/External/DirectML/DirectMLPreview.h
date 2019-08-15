//  Copyright (c) Microsoft Corporation.  All rights reserved.

#pragma once

#include "DirectML.h"

// -----------------------------------------------------------------------------------------
// DML graph

// Graph edges

enum DML_PREVIEW_GRAPH_EDGE_TYPE
{
    DML_PREVIEW_GRAPH_EDGE_TYPE_INVALID,
    DML_PREVIEW_GRAPH_EDGE_TYPE_INPUT,
    DML_PREVIEW_GRAPH_EDGE_TYPE_OUTPUT,
    DML_PREVIEW_GRAPH_EDGE_TYPE_INTERMEDIATE,
};

struct DML_PREVIEW_GRAPH_EDGE
{
    DML_PREVIEW_GRAPH_EDGE_TYPE Type;
    _Field_size_(_Inexpressible_("Dependent on edge type")) const void* Desc;
};

struct DML_PREVIEW_INPUT_GRAPH_EDGE
{
    UINT GraphInputIndex;
    UINT ToNodeIndex;
    UINT ToNodeInputIndex;
};

struct DML_PREVIEW_OUTPUT_GRAPH_EDGE
{
    UINT FromNodeIndex;
    UINT FromNodeOutputIndex;
    UINT GraphOutputIndex;
};

struct DML_PREVIEW_INTERMEDIATE_GRAPH_EDGE
{
    UINT FromNodeIndex;
    UINT FromNodeOutputIndex;
    UINT ToNodeIndex;
    UINT ToNodeInputIndex;
};

// Graph nodes

enum DML_PREVIEW_GRAPH_NODE_TYPE
{
    DML_PREVIEW_GRAPH_NODE_TYPE_INVALID,
    DML_PREVIEW_GRAPH_NODE_TYPE_OPERATOR,
};

struct DML_PREVIEW_GRAPH_NODE
{
    DML_PREVIEW_GRAPH_NODE_TYPE Type;
    _Field_size_(_Inexpressible_("Dependent on node type")) const void* Desc;
};

struct DML_PREVIEW_OPERATOR_GRAPH_NODE
{
    IDMLOperator* Operator;
};

// Graph desc

struct DML_PREVIEW_GRAPH_DESC
{
    UINT InputCount;
    UINT OutputCount;

    UINT NodeCount;
    _Field_size_(NodeCount) const DML_PREVIEW_GRAPH_NODE* Nodes;

    UINT InputEdgeCount;
    _Field_size_opt_(InputEdgeCount) const DML_PREVIEW_GRAPH_EDGE* InputEdges;

    UINT OutputEdgeCount;
    _Field_size_(OutputEdgeCount) const DML_PREVIEW_GRAPH_EDGE* OutputEdges;

    UINT IntermediateEdgeCount;
    _Field_size_opt_(IntermediateEdgeCount) const DML_PREVIEW_GRAPH_EDGE* IntermediateEdges;
};

// -----------------------------------------------------------------------------------------

interface DML_DECLARE_INTERFACE("e405881b-3a4a-4ead-870a-76d20b685029") IDMLDevicePreview : IUnknown
{
    IFACEMETHOD(CompileGraph)(
        const DML_PREVIEW_GRAPH_DESC* desc,
        DML_EXECUTION_FLAGS flags,
        REFIID riid, // expected: IDMLCompiledOperator
        _COM_Outptr_opt_ void** ppv
        ) = 0;
};
