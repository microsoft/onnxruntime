// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sync_api.h"

void CreateAndSubmitThreadpoolWork(_In_ ONNXRUNTIME_CALLBACK_FUNCTION callback, _In_ void* data,
                                   _In_opt_ PThreadPoolCallbackEnv pool) {
  PTP_WORK work = CreateThreadpoolWork(callback, data, pool);
  if (!work) {
    throw std::runtime_error("create thread pool task failed");
  }
  SubmitThreadpoolWork(work);
}

void WaitAndCloseEvent(_In_ ONNXRUNTIME_EVENT finish_event) {
  DWORD dwWaitResult = WaitForSingleObject(finish_event, INFINITE);
  (void)CloseHandle(finish_event);
  if (dwWaitResult != WAIT_OBJECT_0) {
    throw std::runtime_error("WaitForSingleObject failed");
  }
}

ONNXRUNTIME_EVENT CreateOnnxRuntimeEvent() {
  HANDLE finish_event = CreateEvent(NULL,   // default security attributes
                                    TRUE,   // manual-reset event
                                    FALSE,  // initial state is nonsignaled
                                    NULL);
  if (finish_event == NULL) {
    throw std::runtime_error("unable to create finish event");
  }
  return finish_event;
}

void OnnxRuntimeSetEventWhenCallbackReturns(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci,
                                            _In_ ONNXRUNTIME_EVENT finish_event) {
  if (pci)
    SetEventWhenCallbackReturns(pci, finish_event);
  else if (!SetEvent(finish_event)) {
    throw std::runtime_error("SetEvent failed");
  }
}
