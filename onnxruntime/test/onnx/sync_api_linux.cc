// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sync_api.h"
#include <mutex>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <core/common/common.h>
#include <core/common/logging/logging.h>
#include "simple_thread_pool.h"
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
  pthread_mutex_lock(&finish_event->finish_event_mutex);
  while (!finish_event->finished) {
    pthread_cond_wait(&finish_event->finish_event_data, &finish_event->finish_event_mutex);
  }
  pthread_mutex_unlock(&finish_event->finish_event_mutex);
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

using DefaultThreadPoolType = onnxruntime::SimpleThreadPoolTempl<onnxruntime::Env>;
static std::unique_ptr<DefaultThreadPoolType> default_pool;
static std::once_flag default_pool_init;

PThreadPool GetDefaultThreadPool(const onnxruntime::Env& env) {
  std::call_once(default_pool_init, [&env] {
    int core_num = env.GetNumCpuCores();
    default_pool.reset(new DefaultThreadPoolType(core_num, env));
  });
  return default_pool.get();
}

void CloseDefaultThreadPool() {
  default_pool.reset();
}

Status OnnxRuntimeSetEventWhenCallbackReturns(ORT_CALLBACK_INSTANCE pci, ORT_EVENT finish_event) {
  if (finish_event == nullptr)
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  if (pci == nullptr) {
    if (pthread_mutex_lock(&finish_event->finish_event_mutex)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "lock failed");
    }
    finish_event->finished = true;
    if (pthread_mutex_unlock(&finish_event->finish_event_mutex))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unlock failed");
    if (!pthread_cond_broadcast(&finish_event->finish_event_data))
      return Status::OK();
    else
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "pthread_cond_broadcast failed");
  } else {
    pci->AddEvent(finish_event);
    return Status::OK();
  }
}

void OnnxRuntimeCallbackInstance::AddEvent(ORT_EVENT event) {
  events_to_signal_.push_back(event);
}

Status OnnxRuntimeCallbackInstance::SignalAllEvents() {
  for (ORT_EVENT finish_event : events_to_signal_) {
    if (pthread_mutex_lock(&finish_event->finish_event_mutex)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "lock failed");
    }
    finish_event->finished = true;
    if (pthread_mutex_unlock(&finish_event->finish_event_mutex))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unlock failed");
    if (pthread_cond_broadcast(&finish_event->finish_event_data))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "pthread_cond_broadcast failed");
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
