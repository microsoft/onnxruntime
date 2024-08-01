// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/providers.h"
#include "provider_adapter.h"

namespace onnxruntime {
struct ExecutionProviderFactoryAdapter : IExecutionProviderFactory {
ExecutionProviderFactoryAdapter(OrtExecutionProviderFactory* ep_factory, const char* const* provider_option_keys, const char* const* provider_option_values, size_t provider_option_length)
    : ep_factory_(ep_factory), provider_option_keys_(provider_option_keys), provider_option_values_(provider_option_values), provider_option_length_(provider_option_length) {}
std::unique_ptr<IExecutionProvider> CreateProvider() override {
    void* ep = ep_factory_->CreateExecutionProvider(ep_factory_, provider_option_keys_, provider_option_values_, provider_option_length_);
    return std::make_unique<ExecutionProviderAdapter>(reinterpret_cast<OrtExecutionProvider*>(ep));
}
OrtExecutionProviderFactory* ep_factory_;
const char* const* provider_option_keys_;
const char* const* provider_option_values_;
size_t provider_option_length_;
};
}
