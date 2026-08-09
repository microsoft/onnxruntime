// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning::Adapter
{
    // edges and unused edges have an empty array of dimensions.
    class EdgeShapes
    {
    public:
        EdgeShapes() = default;

        EdgeShapes(size_t count) : m_shapes(count) {}

        const std::vector<uint32_t>& GetShape(size_t edgeIndex) const
        {
            return m_shapes[edgeIndex];
        }

        std::vector<uint32_t>& GetMutableShape(size_t edgeIndex)
        {
            return m_shapes[edgeIndex];
        }

        size_t EdgeCount() const { return m_shapes.size(); }

        void Reset(size_t edge_count)
        {
            m_shapes.clear();
            m_shapes.resize(edge_count);
        }

        bool operator!=(const EdgeShapes& other) const noexcept
        {
            return (m_shapes != other.m_shapes);
        }

    private:
        std::vector<std::vector<uint32_t>> m_shapes;
    };
}
