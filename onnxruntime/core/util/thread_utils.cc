#include "thread_utils.h"
#include <algorithm>

#include <core/common/make_unique.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>

namespace onnxruntime {
namespace concurrency {
#ifdef _WIN32
class Win32ThreadPool : public ThreadPoolInterface {
 private:
  PTP_POOL pool_ = NULL;
  TP_CALLBACK_ENVIRON CallBackEnviron_;
  PTP_CLEANUP_GROUP cleanupgroup_;
  const int thread_pool_size_;
  static void CALLBACK MyWorkCallback(PTP_CALLBACK_INSTANCE, PVOID param, PTP_WORK work) {
    std::unique_ptr<std::function<void()>> w((std::function<void()>*)param);
    (*w)();
    CloseThreadpoolWork(work);
  }

 public:
  explicit Win32ThreadPool(int thread_pool_size)
      : pool_(CreateThreadpool(NULL)),
        cleanupgroup_(CreateThreadpoolCleanupGroup()),
        thread_pool_size_(thread_pool_size) {
    InitializeThreadpoolEnvironment(&CallBackEnviron_);
    ORT_ENFORCE(thread_pool_size > 1);
    SetThreadpoolCallbackPool(&CallBackEnviron_, pool_);
    SetThreadpoolCallbackCleanupGroup(&CallBackEnviron_, cleanupgroup_, NULL);
  }
  ~Win32ThreadPool() {
    CloseThreadpoolCleanupGroupMembers(cleanupgroup_, FALSE, NULL);
    CloseThreadpoolCleanupGroup(cleanupgroup_);
    CloseThreadpool(pool_);
  }
  // Submits a closure to be run by a thread in the pool.
  void Schedule(std::function<void()> fn) {
    PTP_WORK work = CreateThreadpoolWork(MyWorkCallback, new std::function<void()>(fn), &CallBackEnviron_);
    if (work == nullptr)
      ORT_THROW("create thread pool work failed");

    SubmitThreadpoolWork(work);
  }

  // If implemented, stop processing the closures that have been enqueued.
  // Currently running closures may still be processed.
  // If not implemented, does nothing.
  virtual void Cancel() {
  }

  // Returns the number of threads in the pool.
  virtual int NumThreads() const {
    return thread_pool_size_;
  }
};

#endif
static std::unique_ptr<ThreadPool> CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options) {
  if (options.thread_pool_size == 1)
    return nullptr;
  std::vector<size_t> cpu_list;
  ThreadOptions to;
  if (options.affinity_vec_len != 0) {
    to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
  }
  if (options.thread_pool_size <= 0) {  // default
    cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
  }
#ifdef _WIN32
  if (options.impl_type == ORT_THREAD_POOL_TYPE_WIN32) {
    return onnxruntime::make_unique<ThreadPool>(new Win32ThreadPool(options.thread_pool_size), true);
  }
#endif
  return onnxruntime::make_unique<ThreadPool>(env, to, options.name, options.thread_pool_size, options.allow_spinning);
}

std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, OrtThreadPoolParams options, ThreadPoolType tpool_type) {
// If openmp is enabled we don't want to create any additional threadpools for sequential execution.
// However, parallel execution relies on the existence of a separate threadpool. Hence we allow eigen threadpools
// to be created for parallel execution.
#ifdef _OPENMP
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(options);
  if (tpool_type != ThreadPoolType::INTER_OP) {
    return nullptr;
  } else {
    return CreateThreadPoolHelper(env, options);
  }
#else
  ORT_UNUSED_PARAMETER(tpool_type);
  return CreateThreadPoolHelper(env, options);
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
namespace OrtApis {
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out) {
  *out = new OrtThreadingOptions();
  return nullptr;
}

ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions* p) {
  delete p;
}
}  // namespace OrtApis