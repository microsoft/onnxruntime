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
    size_t size = AttentionMemoryPlanner::PredictWorkspaceSize(1, 32, 1024, 128, 4);
    // 1 * 32 * 1024 * 128 * 4 = 16,777,216 bytes = 16 MB
    EXPECT_EQ(size, 16777216);
}

TEST(AttentionMemoryPlannerTest, AllocationReuse) {
    auto allocator = std::make_shared<MockAllocator>();
    AttentionMemoryPlanner planner(allocator, 0);

    std::vector<int64_t> shape1 = {1, 32, 1024, 128};
    void* p1 = planner.Allocate(100, shape1);
    
    planner.Free(p1);
    
    void* p2 = planner.Allocate(100, shape1);
    EXPECT_EQ(p1, p2); // Should reuse exact shape
    
    planner.Free(p2);
    
    std::vector<int64_t> shape2 = {1, 32, 1024, 64};
    void* p3 = planner.Allocate(100, shape2);
    EXPECT_EQ(p3, p2); // Should reuse size-compatible buffer
}

}
}
}
}
