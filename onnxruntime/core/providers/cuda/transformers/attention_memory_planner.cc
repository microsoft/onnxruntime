#include "core/providers/cuda/transformers/attention_memory_planner.h"
#include <algorithm>

namespace onnxruntime {
namespace cuda {
namespace transformers {

size_t AttentionMemoryPlanner::PredictWorkspaceSize(int64_t batch_size, int64_t num_heads, int64_t seq_len, int64_t head_dim, size_t element_size) {
    // predicted = batch * num_heads * seq_len * head_dim * sizeof(float)
    size_t predicted = batch_size * num_heads * seq_len * head_dim * element_size;
    
    // Heuristic H2: Predictive Workspace Clamping
    // limit = min(512MB, predicted * 1.25)
    constexpr size_t kLimit = 512 * 1024 * 1024;
    return std::min(kLimit, static_cast<size_t>(predicted * 1.25));
}

void* AttentionMemoryPlanner::Allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t bucketed_size = BucketSize(size);
    
    // Heuristic H3: Tensor Lifetime Reuse (Best Fit / First Fit with size >= requested)
    // We look for a free block that is large enough.
    // Since we bucket, we are likely to find exact matches or slightly larger ones.
    // We pick the first one that fits to avoid scanning the whole list (First Fit).
    // Ideally we might want Best Fit, but First Fit is faster and usually sufficient with bucketing.
    
    for (auto& alloc : allocations_) {
        if (alloc.free && alloc.size >= bucketed_size) {
             alloc.free = false;
             return alloc.ptr;
        }
    }

    // Allocate new
    void* p = allocator_->Alloc(bucketed_size);
    allocations_.push_back({p, bucketed_size, false});
    return p;
}

void AttentionMemoryPlanner::Free(void* p) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& alloc : allocations_) {
        if (alloc.ptr == p) {
            alloc.free = true;
            return;
        }
    }
}

}
}
}
