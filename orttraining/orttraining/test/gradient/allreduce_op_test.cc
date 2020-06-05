// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "orttraining/core/framework/mpi_setup.h"

namespace onnxruntime {
namespace test {
TEST(AllreduceTest, HorovodAllreduceTest) {
  auto mpi_context = training::setup_horovod();
  OpTester test("HorovodAllreduce", 1, onnxruntime::kMSDomain);
  if (mpi_context.world_rank == 0){
   test.AddInput<float>("G", {3}, {4, 5, 6});
  }
  else {
   test.AddInput<float>("G", {3}, {7, 8, 9});     
  }
  test.AddOutput<float>("G_new", {3}, {-1.f, -0.5f, 0.f});
  test.AddOutput<float>("Ready", true);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
