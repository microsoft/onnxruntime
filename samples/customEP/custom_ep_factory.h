#include <memory>	// for the definition of std::unique_ptr used in core/providers/providers.h
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/providers.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API(onnxruntime::IExecutionProviderFactory*, GetEPFactory);

ORT_API_STATUS(RegisterCustomOp, _In_ OrtSessionOptions* options);

#ifdef __cplusplus
}
#endif
