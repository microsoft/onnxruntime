#pragma once
#include <vector>
#include <map>
#include "core/common/common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {
namespace transformers {

class AttentionMemoryPlanner {
 public:
  AttentionMemoryPlanner(AllocatorPtr allocator, size_t stream_idx)
      : allocator_(allocator), stream_idx_(stream_idx) {}

  void* Allocate(size_t size, const std::vector<int64_t>& shape);
  void Free(void* p);

  static size_t PredictWorkspaceSize(int64_t batch_size, int64_t num_heads, int64_t seq_len, int64_t head_dim, size_t element_size);

 private:
  struct Allocation {
    void* ptr;
    size_t size;
    bool free;
    std::vector<int64_t> shape;
  };

  AllocatorPtr allocator_;
  size_t stream_idx_;
  std::vector<Allocation> allocations_;

  size_t BucketSize(size_t size) const {
    constexpr size_t kBucketSize = 256 * 1024; // 256 KB
    return ((size + kBucketSize - 1) / kBucketSize) * kBucketSize;
  }
};

}  // namespace transformers
}  // namespace cuda
}  // namespace onnxruntime
