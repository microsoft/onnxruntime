#include "core/framework/custom_execution_provider.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"

#include <memory>

namespace onnxruntime {
    class ExternalExecutionProvider : public IExecutionProvider{
    public:
        ExternalExecutionProvider(std::unique_ptr<CustomExecutionProvider> external_ep)
            : IExecutionProvider("test"), external_ep_impl_(std::move(external_ep)){
                std::vector<CreateCustomKernelFunc> kernel_funcs = external_ep_impl_->GetRegisteredKernels();
                kernel_registry_ = std::make_shared<KernelRegistry>();
                for (auto& kernel_func : kernel_funcs){
                    ORT_UNUSED_PARAMETER(kernel_func);
                    // kernel_registry.Register(ToKernelBuildInfo(kernel_func));
                }
            }

        virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
            return kernel_registry_;
        }

    private:
        std::unique_ptr<CustomExecutionProvider> external_ep_impl_;
        std::shared_ptr<KernelRegistry> kernel_registry_;
    };
}
