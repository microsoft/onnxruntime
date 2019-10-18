// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"

namespace Dml
{
    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info, 
        const std::unordered_map<std::string, GraphNodeProperties> &graphNodePropertyMap,
        std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap
    );
} // namespace Dml
