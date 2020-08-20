// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sync_api.h"
#include <memory>
#include <mutex>

#if defined(_MSC_VER)
#pragma warning(disable : 4267)
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <core/platform/EigenNonBlockingThreadPool.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <core/common/common.h>
#include <core/common/logging/logging.h>
#include <core/platform/ort_mutex.h>
#include "onnxruntime_event.h"

using onnxruntime::common::Status;

//this can be passed to one of the following functions:
//OnnxRuntimeSetEventWhenCallbackReturns
class OnnxRuntimeCallbackInstance {
 private:
  std::vector<ORT_EVENT> events_to_signal_;

 public:
  void AddEvent(ORT_EVENT event);
  onnxruntime::common::Status SignalAllEvents();
};

Status WaitAndCloseEvent(ORT_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  {
    std::unique_lock<onnxruntime::OrtMutex> lock(finish_event->finish_event_mutex);
    while (!finish_event->finished) {
      finish_event->finish_event_data.wait(lock);
    }
  }
  delete finish_event;
  return Status::OK();
}

Status CreateAndSubmitThreadpoolWork(ORT_CALLBACK_FUNCTION callback, void* data, PThreadPool pool) {
  if (callback == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "callback cannot be NULL");
  if (pool == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "pool cannot be NULL");
  pool->Schedule([=]() {
    OnnxRuntimeCallbackInstance instance;
    callback(&instance, data, nullptr);
    Status st = instance.SignalAllEvents();
    if (!st.IsOK()) {
      LOGF_DEFAULT(ERROR, "SignalAllEvents failed:%s. aborting...\n", st.ErrorMessage().c_str());
      abort();
    }
  });
  return Status::OK();
}

using DefaultThreadPoolType = Eigen::ThreadPool;
static std::unique_ptr<DefaultThreadPoolType> default_pool;
static std::once_flag default_pool_init;

PThreadPool GetDefaultThreadPool(const onnxruntime::Env& env) {
  std::call_once(default_pool_init, [&env] {
    int core_num = env.GetNumCpuCores();
    default_pool = onnxruntime::make_unique<DefaultThreadPoolType>(core_num);
  });
  return default_pool.get();
}

Status OnnxRuntimeSetEventWhenCallbackReturns(ORT_CALLBACK_INSTANCE pci, ORT_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  if (pci == nullptr) {
    {
      std::unique_lock<onnxruntime::OrtMutex> lock(finish_event->finish_event_mutex);
      finish_event->finished = true;
    }
    finish_event->finish_event_data.notify_all();
    return Status::OK();
  }
  pci->AddEvent(finish_event);
  return Status::OK();
}

void OnnxRuntimeCallbackInstance::AddEvent(ORT_EVENT event) {
  events_to_signal_.push_back(event);
}

Status OnnxRuntimeCallbackInstance::SignalAllEvents() {
  for (ORT_EVENT finish_event : events_to_signal_) {
    {
      std::unique_lock<onnxruntime::OrtMutex> lock(finish_event->finish_event_mutex);
      finish_event->finished = true;
    }
    finish_event->finish_event_data.notify_all();
  }
  return Status::OK();
}

Status CreateOnnxRuntimeEvent(ORT_EVENT* out) {
  if (out == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  *out = new OnnxRuntimeEvent();
  return Status::OK();
}

void OrtCloseEvent(ORT_EVENT finish_event) {
  delete finish_event;
}
