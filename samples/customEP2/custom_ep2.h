#pragma once
#include "core/framework/custom_execution_provider.h"

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
    std::vector<std::unique_ptr<SubGraphDef>> GetCapability(GraphViewRef*) override;
    common::Status Compile(std::vector<std::unique_ptr<GraphViewRef>>&, std::vector<std::unique_ptr<NodeViewRef>>&, std::vector<NodeComputeInfo>&) override;
private:
    CustomEp2Info info_;
};

}

#ifdef __cplusplus
extern "C" {
#endif

ORT_API(onnxruntime::CustomEp2*, GetExternalProvider, const void* options);

#ifdef __cplusplus
}
#endif
