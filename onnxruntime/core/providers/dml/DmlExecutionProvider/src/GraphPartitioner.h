// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include "core/providers/dml/DmlExecutionProvider/src/GraphDescBuilder.h"

namespace Dml
{
    class GraphPartition
    {
    public:
        GraphPartition() = default;
        ~GraphPartition() = default;
        GraphPartition* GetRootMergedPartition();
        std::vector<onnxruntime::NodeIndex>& GetNodeIndices();
        std::set<std::string>& GetInputs();
        std::set<std::string>& GetOutputs();
        bool IsFinalized();
        void SetFinalized();
        bool IsDmlPartition();
        bool IsDmlGraphPartition();
        void SetIsDmlPartition(bool isDmlPartition);
        void SetIsDmlGraphPartition(bool isDmlGraphPartition);
        void AddNodeIndex(onnxruntime::NodeIndex index);
        void AddInput(const std::string& name);
        void AddOutput(const std::string& name);
        void Merge(gsl::span<GraphPartition*> partitionsToMerge);

    private:
        std::vector<onnxruntime::NodeIndex> m_nodeIndices;
        std::set<std::string> m_inputs;
        std::set<std::string> m_outputs;
        bool m_finalized = false;
        bool m_isDmlGraphPartition = false;
        bool m_isDmlPartition = false;

        // If not null, this partition has been merged into another, and that partition
        // should be used instead.
        GraphPartition* m_mergedPartition = nullptr;
    };

    std::vector<std::unique_ptr<GraphPartition>>
    BuildPartitions(
        const onnxruntime::GraphViewer& graph,
        const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap& internalRegInfoMap,
        const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& graphNodePropertyMap,
        std::unordered_set<std::string>& requiredInitializerMap,
        std::unordered_set<std::string>& dynamicCpuInputMap,
        gsl::span<const onnxruntime::NodeIndex> additionalSplittingNodes,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputs,
        bool allowDmlGraphDynamicShapes);
} // namespace Dml
