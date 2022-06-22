#include "ThreadUtil.h"

#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace profiling {

namespace {
thread_local int32_t _pid = 0;
thread_local int32_t _tid = 0;
thread_local int32_t _sysTid = 0;
}

int32_t processId() {
  if (!_pid) {
    _pid = onnxruntime::logging::GetProcessId();
  }
  return _pid;
}

int32_t systemThreadId() {
  if (!_sysTid) {
    _sysTid = onnxruntime::logging::GetThreadId();
  }
  return _sysTid;
}

int32_t threadId() {
  if (!_tid) {
    _tid = 0;  //defeat for now
  }
  return _tid;
}

} // namespace profiling
} // namespace onnxruntime
