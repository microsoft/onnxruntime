// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/eager/ort_kernel_invoker.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

TEST(InvokerTest, Basic) {
  std::unique_ptr<IExecutionProvider> cpu_execution_provider = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  const std::string logger_id{"InvokerTest"};
  auto logging_manager = onnxruntime::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kVERBOSE, false,
      logging::LoggingManager::InstanceType::Default,
      &logger_id); 
  ORTInvoker kernel_invoker(std::move(cpu_execution_provider), logging_manager->DefaultLogger());

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue A, B;
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &A);
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &B);
  std::vector<OrtValue> result(1);
  auto status = kernel_invoker.Invoke("Add", {A, B}, result, nullptr);
  ASSERT_TRUE(status.IsOK());
  const Tensor& C = result.back().Get<Tensor>();
  auto& c_shape = C.Shape();
  EXPECT_EQ(c_shape.GetDims(), dims_mul_x);

  std::vector<float> expected_result = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  auto* c_data = C.Data<float>();
  for (auto i = 0; i < c_shape.Size(); ++i) {
    EXPECT_EQ(c_data[i], expected_result[i]);
  }
}
}  // namespace test
}  // namespace onnxruntime
