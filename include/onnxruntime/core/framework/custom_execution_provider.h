// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_c_api.h"

//using CreateCustomKernelFunc = std::function<Ort::Custom::OrtLiteCustomOp()>;

namespace onnxruntime{
    class CustomExecutionProvider{
        public:
        CustomExecutionProvider() {};
        virtual ~CustomExecutionProvider() = default;

        //std::vector<CreateCustomKernelFunc> GetRegisteredKernels();
        std::vector<OrtAllocator*>& GetAllocators() { return allocators_; }
        std::vector<Ort::Custom::OrtLiteCustomOp*>& GetKernelDefinitions() { return kernel_definitions_; }

        protected:
        std::vector<OrtAllocator*> allocators_;
        std::vector<Ort::Custom::OrtLiteCustomOp*> kernel_definitions_;
    };
}
