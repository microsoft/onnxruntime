// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlGraphFusionTransformer.h"

namespace Dml
{
    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
        Windows::AI::MachineLearning::Adapter::EdgeShapes& outputShapes,
        std::vector<DML_INPUT_GRAPH_EDGE_DESC>& inputEdges,
        bool reuseCommandList,
        std::vector<uint8_t>& inputsConstant,
        std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap,
        const gsl::span<const std::string> fusedNodeInputArgOriginalNames
    );
} // namespace Dml
