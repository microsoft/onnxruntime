#pragma once

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif
#include <core/framework/custom_execution_provider.h>

namespace onnxruntime {

struct CustomEp2Info {
    int int_property;
    std::string str_property;
};

class CustomEp2 : public CustomExecutionProvider {
public:
    CustomEp2(const CustomEp2Info& info);
    ~CustomEp2() override = default;
    bool CanCopy(const OrtDevice& src, const OrtDevice& dest) override;
    void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) override;
    void RegisterKernels(lite::IKernelRegistry& kernel_registry) override;
private:
    CustomEp2Info info_;
};

}

#ifdef __cplusplus
extern "C" {
#endif

//ORT_API(onnxruntime::CustomEp2*, GetExternalProvider, const void* options);

EXPORT_API onnxruntime::CustomEp2* GetExternalProvider(const void* provider_options);

#ifdef __cplusplus
}
#endif
