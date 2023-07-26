// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime{
    class CustomExecutionProvider{
        public:
        CustomExecutionProvider() {};
        virtual ~CustomExecutionProvider() = default;

        std::vector<OrtAllocator*>& GetAllocators() { return allocators_; }
        //std::vector<std::unique_ptr<Ort::Custom::OrtLiteCustomOp>>& GetKernelDefinitions() { return kernel_definitions_; }
        size_t GetKernelDefinitionCount() { return kernel_definitions_.size(); }
        Ort::Custom::ExternalKernelDef* GetKernelDefinition(size_t index) {
            if (index >= kernel_definitions_.size()) return nullptr;
            return kernel_definitions_[index].get();
        }
        std::string& GetType() { return type_; }

        protected:
        std::vector<OrtAllocator*> allocators_;
        //std::vector<std::unique_ptr<Ort::Custom::OrtLiteCustomOp>> kernel_definitions_;
        std::vector<std::unique_ptr<Ort::Custom::ExternalKernelDef>> kernel_definitions_;
        std::string type_;
    };
}
