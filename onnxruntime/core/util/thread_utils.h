#include "core/platform/threadpool.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {

enum class ThreadPoolType {
  kIntraOp,
  kInterOp
};

std::unique_ptr<ThreadPool> CreateThreadPool(const std::string& name, ThreadPoolType thread_pool_type, int thread_pool_size);
}  // namespace concurrency
}  // namespace onnxruntime