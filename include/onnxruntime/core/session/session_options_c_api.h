// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/visibility_macros.h"
#include "core/framework/error_code.h"
#include "core/framework/onnx_object.h"

#ifdef __cplusplus
extern "C" {
#endif

DEFINE_RUNTIME_CLASS(ONNXRuntimeProvider);

//Inherented from ONNXObject
typedef struct ONNXRuntimeProviderFactoryInterface {
  ONNXObject parent;
  ONNXStatusPtr(ONNXRUNTIME_API_STATUSCALL* CreateProvider)(void* this_, ONNXRuntimeProviderPtr* out);
} ONNXRuntimeProviderFactoryInterface;

typedef ONNXRuntimeProviderFactoryInterface* ONNXRuntimeProviderFactoryPtr;

struct ONNXRuntimeSessionOptions;
typedef struct ONNXRuntimeSessionOptions ONNXRuntimeSessionOptions;
typedef ONNXRuntimeSessionOptions* ONNXRuntimeSessionOptionsPtr;

/**
 * \return A pointer of the newly created object. The pointer should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCreateSessionOptions, void);

/// create a copy of an existing ONNXRuntimeSessionOptions
ONNXRUNTIME_API(ONNXRuntimeSessionOptions*, ONNXRuntimeCloneSessionOptions, ONNXRuntimeSessionOptions*);
ONNXRUNTIME_API(void, ONNXRuntimeEnableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableSequentialExecution, _In_ ONNXRuntimeSessionOptions* options);

// enable profiling for this session.
ONNXRUNTIME_API(void, ONNXRuntimeEnableProfiling, _In_ ONNXRuntimeSessionOptions* options, _In_ const char* profile_file_prefix);
ONNXRUNTIME_API(void, ONNXRuntimeDisableProfiling, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
ONNXRUNTIME_API(void, ONNXRuntimeEnableMemPattern, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableMemPattern, _In_ ONNXRuntimeSessionOptions* options);

// enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
ONNXRUNTIME_API(void, ONNXRuntimeEnableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);
ONNXRUNTIME_API(void, ONNXRuntimeDisableCpuMemArena, _In_ ONNXRuntimeSessionOptions* options);

///< logger id to use for session output
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogId, _In_ ONNXRuntimeSessionOptions* options, const char* logid);

///< applies to session load, initialization, etc
ONNXRUNTIME_API(void, ONNXRuntimeSetSessionLogVerbosityLevel, _In_ ONNXRuntimeSessionOptions* options, uint32_t session_log_verbosity_level);

///How many threads in the session thread pool.
ONNXRUNTIME_API(int, ONNXRuntimeSetSessionThreadPoolSize, _In_ ONNXRuntimeSessionOptions* options, int session_thread_pool_size);

/**
  * The order of invocation indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
  */
ONNXRUNTIME_API(void, ONNXRuntimeSessionOptionsAppendExecutionProvider, _In_ ONNXRuntimeSessionOptions* options, _In_ ONNXRuntimeProviderFactoryPtr* f);

ONNXRUNTIME_API(void, ONNXRuntimeAddCustomOp, _In_ ONNXRuntimeSessionOptions* options, const char* custom_op_path);

#ifdef __cplusplus
}
#endif
