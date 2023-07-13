#pragma once
#include "core/framework/custom_execution_provider.h"

namespace onnxruntime {

class CustomEp2 : public CustomExecutionProvider {
public:
    CustomEp2();
private:
    std::string type_;
};

}

#ifdef __cplusplus
extern "C" {
#endif

ORT_API(onnxruntime::CustomEp2*, GetExternalProvider);

#ifdef __cplusplus
}
#endif
