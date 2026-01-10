// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once
#include <queue>

inline void PerformTopologicalSortAndCheckIsAcyclic(
    const DmlSerializedGraphDesc& graphDesc,
    std::vector<uint32_t>& nodesInTopologicalOrder)
{
    uint32_t nodeCount = static_cast<uint32_t>(graphDesc.Nodes.size());
    std::queue<uint32_t> queue;
    std::vector<uint32_t> inDegree(nodeCount, 0);
    std::vector<std::vector<uint32_t>> children(nodeCount);

    // Don't need to iterate through InputEdges because those inputs don't represent any node
    // and the purpose of this topological sort is to come up with a order to correctly iterate 
    // through nodes .
    for (const DmlIntermediateSerializedGraphEdge& intermediateEdge : graphDesc.IntermediateEdges)
    {
        inDegree[intermediateEdge.ToNodeIndex]++;
        children[intermediateEdge.FromNodeIndex].push_back(intermediateEdge.ToNodeIndex);
    }

    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
    {
        if (inDegree[nodeIndex] == 0)
        {
            queue.push(nodeIndex);
        }
    }

    uint32_t nodeIndex = 0;
    while (!queue.empty())
    {
        if (nodeIndex >= nodeCount)
        {
            throw std::invalid_argument("Given graph is not acyclic.");
        }

        uint32_t currNodeIndex = queue.front();
        queue.pop();
        nodesInTopologicalOrder[nodeIndex++] = currNodeIndex;

        for (uint32_t child : children[currNodeIndex])
        {
            if (--inDegree[child] == 0)
            {
                queue.push(child);
            }
        }
    }
}
