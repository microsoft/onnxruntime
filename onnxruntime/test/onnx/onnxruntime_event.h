// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <condition_variable>
#include <core/common/common.h>

struct OnnxRuntimeEvent {
 public:
  std::mutex finish_event_mutex;
  std::condition_variable finish_event_data;
  bool finished = false;
  OnnxRuntimeEvent() = default;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeEvent);
};

using ORT_EVENT = OnnxRuntimeEvent*;
