// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/providers.h"
#include "provider_adapter.h"

namespace onnxruntime {
struct ExecutionProviderFactoryAdapter : IExecutionProviderFactory {
ExecutionProviderFactoryAdapter(OrtExecutionProviderFactory* ep_factory, const char* const* provider_option_keys, const char* const* provider_option_values, size_t provider_option_length)
    : ep_factory_(ep_factory), provider_option_length_(provider_option_length) {
        provider_option_keys_.reserve(provider_option_length);
        provider_option_values_.reserve(provider_option_length);
        keys_.reserve(provider_option_length);
        values_.reserve(provider_option_length);
        for (size_t i = 0; i < provider_option_length; i++) {
            provider_option_keys_.push_back(provider_option_keys[i]);
            provider_option_values_.push_back(provider_option_values[i]);
            keys_.push_back(provider_option_keys_[i].c_str());
            values_.push_back(provider_option_values_[i].c_str());
        }
    }

std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<ExecutionProviderAdapter>(ep_factory_->CreateExecutionProvider(ep_factory_, keys_.data(), values_.data(), provider_option_length_));
}
OrtExecutionProviderFactory* ep_factory_;
//const char* const* provider_option_keys_;
//const char* const* provider_option_values_;
std::vector<std::string> provider_option_keys_, provider_option_values_;
std::vector<const char*> keys_, values_;
size_t provider_option_length_;
};
}
