/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/hailo/hailo_provider_factory.h"
#include "hailo_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {

void Shutdown_DeleteRegistry();

struct HailoProviderFactory : IExecutionProviderFactory {
    HailoProviderFactory(bool create_arena) : create_arena_(create_arena) {}
    ~HailoProviderFactory() override {}

    std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
    bool create_arena_;
};

std::unique_ptr<IExecutionProvider> HailoProviderFactory::CreateProvider() {
    HailoExecutionProviderInfo info;
    info.create_arena = create_arena_;
    return std::make_unique<HailoExecutionProvider>(info);
}

struct Hailo_Provider : Provider {
    std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int use_arena) override {
        return std::make_shared<HailoProviderFactory>(use_arena != 0);
    }

    void Shutdown() override {
        Shutdown_DeleteRegistry();
    }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
    return &onnxruntime::g_provider;
}
}
