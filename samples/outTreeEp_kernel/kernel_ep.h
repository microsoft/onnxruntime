#pragma once
#include "core/session/onnxruntime_c_api.h"
#include <string>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

namespace onnxruntime {

struct KernelEpInfo {
    int int_property;
    std::string str_property;
};

struct KernelEp : public OrtExecutionProvider {
    KernelEp(const char* ep_type, const KernelEpInfo& ep_info);
    KernelEpInfo info;
};

struct KernelEpFactory : public OrtExecutionProviderFactory {
    KernelEpFactory();
};
}

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API OrtExecutionProviderFactory* RegisterCustomEp();

#ifdef __cplusplus
}
#endif
