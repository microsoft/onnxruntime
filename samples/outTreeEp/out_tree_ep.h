#pragma once
#include "core/session/onnxruntime_c_api.h"
#include <string>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

namespace onnxruntime {

struct OutTreeEpInfo {
    int int_property;
    std::string str_property;
};

struct OutTreeEp : public OrtExecutionProvider {
    OutTreeEp(const char* ep_type, const OutTreeEpInfo& ep_info);
    OutTreeEpInfo info;
};

struct OutTreeEpFactory : public OrtExecutionProviderFactory {
    OutTreeEpFactory();
};
}

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API OrtExecutionProviderFactory* RegisterCustomEp();

#ifdef __cplusplus
}
#endif
