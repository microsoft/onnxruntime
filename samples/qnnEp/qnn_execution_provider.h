#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include <string>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

namespace onnxruntime {

struct QNNExecutionProvider : public OrtExecutionProvider {
    QNNExecutionProvider(const char* ep_type, const ProviderOptions& provider_options);
private:
    std::string context_cache_path_cfg_ = "";
};

struct QNNExecutionProviderFactory : public OrtExecutionProviderFactory {
    QNNExecutionProviderFactory();
};
}

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API OrtExecutionProviderFactory* RegisterCustomEp();

#ifdef __cplusplus
}
#endif
