#include "gtest/gtest.h"
#include "core/framework/execution_steps.h"

namespace onnxruntime {
namespace onnxruntime::test {

#ifdef ENABLE_TRAINING
TEST(ExecutionStepsTest, RetrieveRegionBoundaryFromProgramCounter) {
  std::vector<std::unique_ptr<SequentialExecutionPlan::ExecutionStep>> steps;
  steps.emplace_back(std::make_unique<BarrierStep>(0));
  steps.emplace_back(std::make_unique<LaunchKernelStep>(10));
  steps.emplace_back(std::make_unique<TriggerDownstreamStep>(0));
  steps.emplace_back(std::make_unique<BarrierStep>(0));
  steps.emplace_back(std::make_unique<LaunchKernelStep>(20));
  steps.emplace_back(std::make_unique<TriggerDownstreamStep>(0));
  steps.emplace_back(std::make_unique<LaunchKernelStep>(30));
  steps.emplace_back(std::make_unique<LaunchKernelStep>(40));
  steps.emplace_back(std::make_unique<TriggerDownstreamStep>(0));
  InlinedHashMap<NodeIndex, size_t> node_index_2_toposort_index{{10, 0}, {20, 1}, {30, 2}, {40, 3}};
  size_t start_region = 0, end_region = 0;
  RetrieveRegionBoundaryFromProgramCounter(steps, node_index_2_toposort_index, 0, 2, start_region, end_region); // Node with Index 30 is yieldOp
  EXPECT_EQ(start_region, 0);
  EXPECT_EQ(end_region, 6);

  RetrieveRegionBoundaryFromProgramCounter(steps, node_index_2_toposort_index, 2, node_index_2_toposort_index.size(), start_region, end_region);
  EXPECT_EQ(start_region, 6);
  EXPECT_EQ(end_region, steps.size());
}
#endif
}
}
