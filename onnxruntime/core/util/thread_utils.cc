#include "thread_utils.h"

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(const std::string& name, int thread_pool_size) {
  if (thread_pool_size <= 0) {  // default
    thread_pool_size = std::thread::hardware_concurrency() / 2;
  }

  // since we use the main thread for execution we don't have to create any threads on the thread pool when
  // the requested size is 1. For other cases, we will have thread_pool_size + 1 threads for execution
  return thread_pool_size == 1 ? nullptr : std::make_unique<concurrency::ThreadPool>(name, thread_pool_size);
}
}  // namespace concurrency
}  // namespace onnxruntime