// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/session/onnxruntime_lite_custom_op.h"

using CreateCustomKernelFunc = std::function<Ort::Custom::OrtLiteCustomOp()>;

namespace onnxruntime{
    class CustomExecutionProvider{
        public:
        CustomExecutionProvider();

        std::vector<CreateCustomKernelFunc> GetRegisteredKernels();
    };
}
