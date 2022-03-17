#include "ThreadUtil.h"

#ifndef _MSC_VER
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#else // _MSC_VER
#include <locale>
#include <codecvt>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <processthreadsapi.h>
#undef ERROR
#endif // _MSC_VER

//namespace onnxruntime {

//namespace profiling {

namespace {
thread_local int32_t _pid = 0;
thread_local int32_t _tid = 0;
thread_local int32_t _sysTid = 0;
}

int32_t processId() {
  if (!_pid) {
#ifndef _MSC_VER
    _pid = (int32_t)getpid();
#else
    _pid = (int32_t)GetCurrentProcessId();
#endif
  }
  return _pid;
}

int32_t systemThreadId() {
  if (!_sysTid) {
#ifdef __APPLE__
    _sysTid = (int32_t)syscall(SYS_thread_selfid);
#elif defined _MSC_VER
    _sysTid = (int32_t)GetCurrentThreadId();
#else
    _sysTid = (int32_t)syscall(SYS_gettid);
#endif
  }
  return _sysTid;
}

int32_t threadId() {
  if (!_tid) {
#ifdef __APPLE__
    uint64_t tid;
    pthread_threadid_np(nullptr, &tid);
    _tid = tid;
#elif defined _MSC_VER
  _tid = (int32_t)GetCurrentThreadId();
#else
  pthread_t pth = pthread_self();
  int32_t* ptr = reinterpret_cast<int32_t*>(&pth);
  _tid = *ptr;
#endif
  }
  return _tid;
}

//}
//}
