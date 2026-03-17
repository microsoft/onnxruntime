// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/profiler_common.h"
#include "core/session/onnxruntime_c_api.h"

// Concrete definition of the opaque OrtEpProfilingEventsContainer type declared in the public C API.
// ORT creates an instance wrapping a profiling::Events vector and passes it to the EP's
// OrtEpProfilerImpl::EndProfiling() function.
// The EP calls OrtEpApi::EpProfilingEventsContainer_AddEvents to push events into this container.
struct OrtEpProfilingEventsContainer {
  onnxruntime::profiling::Events& events;
};
