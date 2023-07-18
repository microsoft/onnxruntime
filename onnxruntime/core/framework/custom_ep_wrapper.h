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
                std::vector<Ort::Custom::OrtLiteCustomOp*> kernel_defn = external_ep_impl_->GetKernelDefinitions();
                for (auto& k : kernel_defn) {
                //    OrtLiteCustomOp2KernelRegistry(k);
                    std::unique_ptr<OrtCustomOpDomain> custom_op_domain = std::make_unique<OrtCustomOpDomain>();
                    custom_op_domain->domain_ = "test";
                    custom_op_domain->custom_ops_.push_back(k);
                    custom_op_domain_list_.push_back(custom_op_domain.release());
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

        void GetCustomOpDomainList(std::vector<OrtCustomOpDomain*>& custom_op_domain_list) const override {
            custom_op_domain_list = custom_op_domain_list_;
        }

    private:
        std::unique_ptr<CustomExecutionProvider> external_ep_impl_;
        std::shared_ptr<KernelRegistry> kernel_registry_;
        std::vector<OrtCustomOpDomain*> custom_op_domain_list_;

        void OrtLiteCustomOp2KernelRegistry(Ort::Custom::OrtLiteCustomOp* kernel_definition) {
            KernelCreateInfo kernel_create_info = CreateKernelCreateInfo("", kernel_definition);
            ORT_THROW_IF_ERROR(kernel_registry_->Register(std::move(kernel_create_info)));
        }
    };
}
