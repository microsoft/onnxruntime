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

#pragma once

#include "core/providers/cuda/cuda_common.h"
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif
#include "operations.h"

namespace onnxruntime {
namespace cuda {
#define INVALID_DEVICE_ID (-1)
class ORTReadyEvent : public horovod::common::ReadyEvent {
 public:
  ORTReadyEvent(int device);
  ~ORTReadyEvent();
  virtual bool Ready() const override;

 private:
  int device_ = INVALID_DEVICE_ID;
  cudaEvent_t cuda_event_ = nullptr;
};
}  //namespace cuda
}  // namespace onnxruntime

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif