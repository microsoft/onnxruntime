// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/profiler_common.h"
#include "core/session/onnxruntime_c_api.h"

// Concrete definition of the opaque OrtEpProfilingEventsContainer type.
// ORT creates an instance wrapping a profiling::Events vector and passes it to the EP's EndProfiling.
// The EP calls OrtEpApi::EpProfilingEventsContainer_AddEvents to push events into it.
//
// This is an ORT-internal type. EP plugins only see the opaque forward declaration.
struct OrtEpProfilingEventsContainer {
  onnxruntime::profiling::Events& events;
};
