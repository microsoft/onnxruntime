// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <core/common/status.h>
#include <core/common/common.h>
#include <core/platform/env.h>
#include <core/platform/thread_pool_interface.h>

#define ORT_CALLBACK

using PThreadPool = onnxruntime::ThreadPoolInterface*;
#define ORT_WORK void*
struct OnnxRuntimeEvent;
using ORT_EVENT = OnnxRuntimeEvent*;

class OnnxRuntimeCallbackInstance;
using ORT_CALLBACK_INSTANCE = OnnxRuntimeCallbackInstance*;
using ORT_CALLBACK_FUNCTION = void ORT_CALLBACK (*)(ORT_CALLBACK_INSTANCE pci, void* context, ORT_WORK work);
// Do nothing
inline void OnnxRuntimeCloseThreadpoolWork(ORT_WORK) {
}

// The returned value will be used with CreateAndSubmitThreadpoolWork function
PThreadPool GetDefaultThreadPool(const ::onnxruntime::Env& env);
// On Windows, the last parameter can be null, in that case it will use the default thread pool.
// On Linux, there is no per process default thread pool. You have to pass a non-null pointer.
// Caller must delete the data pointer if this function returns a non-ok status. Otherwise, the ownership is transferred
::onnxruntime::common::Status CreateAndSubmitThreadpoolWork(ORT_CALLBACK_FUNCTION callback, void* data,
                                                            PThreadPool pool);
::onnxruntime::common::Status CreateOnnxRuntimeEvent(ORT_EVENT* out);
// pci is a pointer, can be NULL. If pci is NULL, signal the event immediately
::onnxruntime::common::Status OnnxRuntimeSetEventWhenCallbackReturns(ORT_CALLBACK_INSTANCE pci, ORT_EVENT finish_event);
::onnxruntime::common::Status WaitAndCloseEvent(ORT_EVENT finish_event);
void OrtCloseEvent(ORT_EVENT finish_event);
