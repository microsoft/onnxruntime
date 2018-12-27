// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <vector>
#endif
#include <core/common/status.h>
#include <core/common/common.h>
#include <core/platform/env.h>
#include <core/common/task_thread_pool.h>

#ifdef _WIN32
using ORT_EVENT = HANDLE;
using PThreadPool = onnxruntime::TaskThreadPool*;
#define OnnxRuntimeCloseThreadpoolWork CloseThreadpoolWork
using ORT_CALLBACK_INSTANCE = void*;
using ORT_CALLBACK_FUNCTION = void (*)(ORT_CALLBACK_INSTANCE pci, void* data);

inline PThreadPool GetDefaultThreadPool(const ::onnxruntime::Env&) {
  return new onnxruntime::TaskThreadPool(std::thread::hardware_concurrency() / 2);
}
#else
namespace Eigen {
class ThreadPoolInterface;
}
using PThreadPool = Eigen::ThreadPoolInterface*;
#define ORT_WORK void*
struct OnnxRuntimeEvent;
using ORT_EVENT = OnnxRuntimeEvent*;
class OnnxRuntimeCallbackInstance;
using ORT_CALLBACK_INSTANCE = OnnxRuntimeCallbackInstance*;
using ORT_CALLBACK_FUNCTION = void (*)(ORT_CALLBACK_INSTANCE pci, void* context);
#endif

//The returned value will be used with CreateAndSubmitThreadpoolWork function
PThreadPool GetDefaultThreadPool(const ::onnxruntime::Env& env);
//Caller must delete the data pointer if this function returns a non-ok status. Otherwise, the ownership is transferred
::onnxruntime::common::Status CreateAndSubmitThreadpoolWork(ORT_CALLBACK_FUNCTION callback, void* data, PThreadPool pool);
::onnxruntime::common::Status CreateOnnxRuntimeEvent(ORT_EVENT* out);
::onnxruntime::common::Status OnnxRuntimeSetEventWhenCallbackReturns(ORT_CALLBACK_INSTANCE pci, ORT_EVENT finish_event);
::onnxruntime::common::Status WaitAndCloseEvent(ORT_EVENT finish_event);
void OrtCloseEvent(ORT_EVENT finish_event);
