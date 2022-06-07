// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/common.h>
#include <core/platform/ort_mutex.h>

struct OnnxRuntimeEvent {
 public:
  onnxruntime::OrtMutex finish_event_mutex;
  onnxruntime::OrtCondVar finish_event_data;
  bool finished = false;
  OnnxRuntimeEvent() = default;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeEvent);
};

using ORT_EVENT = OnnxRuntimeEvent*;
