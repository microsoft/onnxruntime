#include "out_tree_ep.h"
#include <memory>
namespace onnxruntime {

OutTreeEpFactory::OutTreeEpFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> void* {
        OutTreeEpInfo info;
        for (size_t i = 0; i < option_size; i++) {
            if (!strcmp(ep_option_keys[i], "int_property")) info.int_property = std::atoi(ep_option_values[i]);
            else if (!strcmp(ep_option_keys[i], "str_property")) info.str_property = ep_option_values[i];
            // TODO(leca): else throw
        }
        std::unique_ptr<OutTreeEp> ret = std::make_unique<OutTreeEp>("outTreeEp", std::move(info));
        return ret.release();
    };
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
