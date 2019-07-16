// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <vector>
#endif
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#ifdef _WIN32
#define my_strtol wcstol
#define my_strrchr wcsrchr
#define my_strcasecmp _wcsicmp
#define my_strdup _strdup
#else
#define my_strtol strtol
#define my_strrchr strrchr
#define my_strcasecmp strcasecmp
#define my_strdup strdup
#endif

#ifdef _WIN32
using ONNXRUNTIME_CALLBACK_INSTANCE = PTP_CALLBACK_INSTANCE;
using ONNXRUNTIME_EVENT = HANDLE;
#define ONNXRUNTIME_CALLBACK __stdcall
using ONNXRUNTIME_WORK = PTP_WORK;
using PThreadPoolCallbackEnv = PTP_CALLBACK_ENVIRON;
using ONNXRUNTIME_CALLBACK_FUNCTION = PTP_WORK_CALLBACK;
#define OnnxRuntimeCloseThreadpoolWork CloseThreadpoolWork
inline PThreadPoolCallbackEnv GetDefaultThreadPool() { return nullptr; }
#else
#define ONNXRUNTIME_CALLBACK
namespace Eigen {
class ThreadPoolInterface;
}
using PThreadPoolCallbackEnv = Eigen::ThreadPoolInterface*;
#define ONNXRUNTIME_WORK void*
struct OnnxRuntimeEvent;
using ONNXRUNTIME_EVENT = OnnxRuntimeEvent*;

class OnnxRuntimeCallbackInstance;
using ONNXRUNTIME_CALLBACK_INSTANCE = OnnxRuntimeCallbackInstance*;
using ONNXRUNTIME_CALLBACK_FUNCTION = void ONNXRUNTIME_CALLBACK (*)(ONNXRUNTIME_CALLBACK_INSTANCE pci, void* context,
                                                                    ONNXRUNTIME_WORK work);
#endif

// The returned value will be used with CreateAndSubmitThreadpoolWork function
PThreadPoolCallbackEnv GetDefaultThreadPool();
// On Windows, the last parameter can be null, in that case it will use the default thread pool.
// On Linux, there is no per process default thread pool. You have to pass a non-null pointer.
// Caller must delete the data pointer if this function returns a non-ok status. Otherwise, the ownership is transferred
void CreateAndSubmitThreadpoolWork(_In_ ONNXRUNTIME_CALLBACK_FUNCTION callback, _In_ void* data,
                                   _In_opt_ PThreadPoolCallbackEnv pool);
ONNXRUNTIME_EVENT CreateOnnxRuntimeEvent();
// pci is a pointer, can be NULL. If pci is NULL, signal the event immediately
void OnnxRuntimeSetEventWhenCallbackReturns(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci,
                                            _In_ ONNXRUNTIME_EVENT finish_event);
void WaitAndCloseEvent(_In_ ONNXRUNTIME_EVENT finish_event);
