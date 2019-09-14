#include "thread_utils.h"

namespace onnxruntime {
namespace concurrency {
std::unique_ptr<ThreadPool> CreateThreadPool(const std::string& name, int thread_pool_size) {
  if (thread_pool_size < 0) thread_pool_size = std::thread::hardware_concurrency() / 2;
  return thread_pool_size > 0 ? std::make_unique<concurrency::ThreadPool>(name, thread_pool_size) : nullptr;
}
}  // namespace concurrency
}  // namespace onnxruntime