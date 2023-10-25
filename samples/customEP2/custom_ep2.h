#pragma once
#include "interface/provider/provider.h"

namespace onnxruntime {

struct CustomEp2Info {
    int int_property;
    std::string str_property;
};

class CustomEp2 : public interface::ExecutionProvider {
public:
    CustomEp2(const CustomEp2Info& info);
    ~CustomEp2() override = default;
    bool CanCopy(const OrtDevice& src, const OrtDevice& dest) override;
    void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) override;
    std::vector<std::unique_ptr<SubGraphDef>> GetCapability(interface::GraphViewRef*) override;
    common::Status Compile(std::vector<std::unique_ptr<interface::GraphViewRef>>&, std::vector<std::unique_ptr<interface::NodeViewRef>>&, std::vector<NodeComputeInfo>&) override;
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
