#include "core/platform/threadpool.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(const std::string& name, int thread_pool_size);
}  // namespace concurrency
}  // namespace onnxruntime