// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <pthread.h>
#include <core/common/common.h>

struct OnnxRuntimeEvent {
 public:
  pthread_mutex_t finish_event_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t finish_event_data = PTHREAD_COND_INITIALIZER;
  bool finished = false;
  OnnxRuntimeEvent() = default;

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeEvent);
};

using ONNXRUNTIME_EVENT = OnnxRuntimeEvent*;
