#include "thread_utils.h"

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(const std::string& name, ThreadPoolType thread_pool_type,
                                             int thread_pool_size) {
  if (thread_pool_size <= 0) {  // default
    thread_pool_size = std::thread::hardware_concurrency() / 2;
  } else {
    // In the case of intra op thread pool we use the main thread as well for execution, hence we need to
    // subtract 1 from the thread pool size. This is to accomodate users who require only one thread to be
    // reserved for ORT.
    thread_pool_size = thread_pool_type == ThreadPoolType::kIntraOp ? thread_pool_size - 1 : thread_pool_size;
  }

  return thread_pool_size == 0 ? nullptr : std::make_unique<concurrency::ThreadPool>(name, thread_pool_size);
}
}  // namespace concurrency
}  // namespace onnxruntime