// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/* Modifications Copyright (c) Microsoft. */

#include "ready_event.h"

#include <cassert>
#include <mutex>
#include <queue>
#include <unordered_map>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

namespace onnxruntime {
namespace cuda {

struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

ORTReadyEvent::ORTReadyEvent(int device) : device_(device) {
  ORT_ENFORCE(device != INVALID_DEVICE_ID);
  std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
  auto& queue = ready_event_registry.cuda_events[device_];
  if (!queue.empty()) {
    cuda_event_ = queue.front();
    queue.pop();
  } else {
    CUDA_CALL_THROW(cudaEventCreateWithFlags(
        &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
  }
  //We only use default stream, which is nullptr
  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaEventRecord(cuda_event_, stream));
}

ORTReadyEvent::~ORTReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

bool ORTReadyEvent::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  CUDA_CALL_THROW(status);
  return true;
}
}  //namespace cuda
}  // namespace onnxruntime
