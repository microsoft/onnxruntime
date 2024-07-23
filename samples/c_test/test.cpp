#include "core/session/onnxruntime_c_api.h"

inline void THROW_ON_ERROR(OrtStatus* status) {
    if (status != nullptr) abort();
}

int main() {
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* p_env = nullptr;
    OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
    THROW_ON_ERROR(g_ort->CreateEnv(log_level, "", &p_env));
    THROW_ON_ERROR(g_ort->RegisterOrtExecutionProviderLibrary("/home/leca/code/onnxruntime/samples/outTreeEp/build/liboutTreeEp.so", p_env, "outTreeEp"));
    return 0;
}
