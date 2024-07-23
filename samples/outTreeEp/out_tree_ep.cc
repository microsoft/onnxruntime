#include "out_tree_ep.h"
#include <memory>
namespace onnxruntime {

OutTreeEpFactory::OutTreeEpFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_) -> void* {
        std::unique_ptr<OutTreeEp> ret = std::make_unique<OutTreeEp>();
        return ret.release(); };
}

}

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::OutTreeEpFactory> ret = std::make_unique<onnxruntime::OutTreeEpFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
