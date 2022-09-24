#pragma once

#include <cstdint>

namespace onnxruntime {

namespace profiling {

int32_t systemThreadId();
int32_t threadId();
int32_t processId();

} 
}

using namespace onnxruntime::profiling;
