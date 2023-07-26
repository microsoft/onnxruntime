#pragma once
#include "core/framework/custom_execution_provider.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/framework_provider_common.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/allocator_adapters.h"
#include "core/session/custom_ops.h"
#include <memory>

namespace onnxruntime {
    class ExternalExecutionProvider : public IExecutionProvider{
    public:
        ExternalExecutionProvider(CustomExecutionProvider* external_ep)
            : IExecutionProvider(external_ep->GetType()), external_ep_impl_(external_ep){
                kernel_registry_ = std::make_shared<KernelRegistry>();
                size_t kernelsCount = external_ep_impl_->GetKernelDefinitionCount();
                for (size_t i = 0; i < kernelsCount; i++) {
                    OrtLiteCustomOp2KernelRegistry(external_ep_impl_->GetKernelDefinition(i));
                }
            }

        virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
            return kernel_registry_;
        }

        std::vector<AllocatorPtr> CreatePreferredAllocators() override {
            std::vector<AllocatorPtr> ret;
            std::vector<OrtAllocator*> allocators = external_ep_impl_->GetAllocators();
            for (auto& allocator : allocators) {
                AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
                ret.push_back(std::move(alloc_ptr));
            }
            return ret;
        }

    private:
        std::unique_ptr<CustomExecutionProvider> external_ep_impl_;
        std::shared_ptr<KernelRegistry> kernel_registry_;

        void OrtLiteCustomOp2KernelRegistry(Ort::Custom::ExternalKernelDef* kernel_definition) {
            KernelCreateInfo kernel_create_info = CreateKernelCreateInfo(kernel_definition->domain_, kernel_definition->custom_op_.get());
            ORT_THROW_IF_ERROR(kernel_registry_->Register(std::move(kernel_create_info)));
        }
    };
}
