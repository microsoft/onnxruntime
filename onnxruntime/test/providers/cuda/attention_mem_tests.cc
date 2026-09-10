#include "gtest/gtest.h"
#include "core/providers/cuda/transformers/attention_memory_planner.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {
namespace transformers {
namespace test {

class MockAllocator : public IAllocator {
public:
    MockAllocator() : IAllocator(OrtMemoryInfo("CUDA", OrtAllocatorType::OrtDeviceAllocator)) {}
    void* Alloc(size_t size) override {
        return malloc(size);
    }
    void Free(void* p) override {
        free(p);
    }
};

TEST(AttentionMemoryPlannerTest, PredictWorkspaceSize) {
    // predicted = 1 * 32 * 1024 * 128 * 4 = 16,777,216 bytes = 16 MB
    // limit = min(512MB, predicted * 1.25) = 16MB * 1.25 = 20 MB = 20,971,520 bytes
    size_t size = AttentionMemoryPlanner::PredictWorkspaceSize(1, 32, 1024, 128, 4);
    EXPECT_EQ(size, 20971520);
}

TEST(AttentionMemoryPlannerTest, AllocationReuse_BestFit) {
    auto allocator = std::make_shared<MockAllocator>();
    AttentionMemoryPlanner planner(allocator, 0);

    // Allocate 1MB (will be bucketed to 1MB if bucket size is 256KB)
    size_t size1 = 1024 * 1024; 
    void* p1 = planner.Allocate(size1);
    
    planner.Free(p1);
    
    // Allocate slightly smaller size, should reuse p1
    size_t size2 = size1 - 1024;
    void* p2 = planner.Allocate(size2);
    EXPECT_EQ(p1, p2); // Should reuse the same pointer
    
    planner.Free(p2);
}

TEST(AttentionMemoryPlannerTest, MetadataStability_Autoregressive) {
    auto allocator = std::make_shared<MockAllocator>();
    AttentionMemoryPlanner planner(allocator, 0);

    // Simulate autoregressive generation: seq_len increases, so buffer size increases
    // We want to ensure we don't keep allocating new blocks without reusing old ones if they fit.
    // Note: In a real scenario, we'd likely free the old smaller buffer and allocate a new larger one.
    // If we free the old one, it becomes available.
    
    void* p_prev = nullptr;
    
    // Step 1: Allocate 100KB
    void* p1 = planner.Allocate(100 * 1024);
    p_prev = p1;
    
    // Step 2: Free p1, Allocate 110KB
    // Since 100KB < 256KB bucket, it's not bucketed in our current logic (if size < kBucketSize return size).
    // Wait, let's check the logic: "if (size < kBucketSize) return size;"
    // So small allocations are exact.
    
    planner.Free(p1);
    
    // If we allocate larger, we can't reuse the smaller block.
    void* p2 = planner.Allocate(110 * 1024);
    EXPECT_NE(p1, p2); // Can't reuse smaller block for larger request
    
    planner.Free(p2);
    
    // Step 3: Large allocations (bucketed)
    // Allocate 1MB
    void* p3 = planner.Allocate(1024 * 1024);
    planner.Free(p3);
    
    // Allocate 0.9MB (should reuse 1MB bucket)
    void* p4 = planner.Allocate(900 * 1024);
    EXPECT_EQ(p3, p4);
    planner.Free(p4);
}

}
}
}
}
