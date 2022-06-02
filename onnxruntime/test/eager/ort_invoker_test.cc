// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/eager/ort_kernel_invoker.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/framework/test_utils.h"
#include "asserts.h"
#include <core/framework/kernel_registry.h>
#include <core/framework/op_kernel.h>
#include <core/graph/schema_registry.h>
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace test {

TEST(InvokerTest, Basic) {
  std::unique_ptr<IExecutionProvider> cpu_execution_provider =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false, false));
  const std::string logger_id{"InvokerTest"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kVERBOSE, false,
      logging::LoggingManager::InstanceType::Default,
      &logger_id);
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), env));
  IOnnxRuntimeOpSchemaRegistryList tmp_op_registry = {};
  ORTInvoker kernel_invoker(std::move(cpu_execution_provider), env->GetLoggingManager()->DefaultLogger(), tmp_op_registry);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue A, B;
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &A);
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &B);
  std::vector<OrtValue> result(1);
  ASSERT_STATUS_OK(kernel_invoker.Invoke("Add", {A, B}, result, nullptr));
  const Tensor& C = result.back().Get<Tensor>();
  auto& c_shape = C.Shape();
  EXPECT_EQ(c_shape.GetDims(), gsl::make_span(dims_mul_x));

  std::vector<float> expected_result = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  auto* c_data = C.Data<float>();
  for (auto i = 0; i < c_shape.Size(); ++i) {
    EXPECT_EQ(c_data[i], expected_result[i]);
  }
}

TEST(InvokerTest, Inplace) {
  std::unique_ptr<IExecutionProvider> cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false, false));
  const std::string logger_id{"InvokerTest"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kVERBOSE, false,
      logging::LoggingManager::InstanceType::Default,
      &logger_id);
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), env));
  IOnnxRuntimeOpSchemaRegistryList tmp_op_registry = {};
  ORTInvoker kernel_invoker(std::move(cpu_execution_provider), env->GetLoggingManager()->DefaultLogger(), tmp_op_registry);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue A, B;
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &A);
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &B);
  std::vector<OrtValue> result;
  result.push_back(A);
  ASSERT_STATUS_OK(kernel_invoker.Invoke("Add", {A, B}, result, nullptr));

  std::vector<float> expected_result = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  auto* a_data = A.Get<Tensor>().Data<float>();
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(a_data[i], expected_result[i]);
  }
}

class TestKernel final : public OpKernel {
 public:
  TestKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override {
    return Status::OK();
  }
};

Status CreateTestKernel(FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
  out = std::make_unique<TestKernel>(info);
  return Status::OK();
}

TEST(InvokerTest, CustomOp) {
  std::unique_ptr<IExecutionProvider> cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false, false));
  // register "Test" kernel to "FakeDomain"
  auto kernel_registry = cpu_execution_provider->GetKernelRegistry();
  auto kernel_def = KernelDefBuilder()
                        .MayInplace(0, 0)
                        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                        .SetName("Test")
                        .SetDomain("FakeDomain")
                        .SinceVersion(1)
                        .Provider(kCpuExecutionProvider)
                        .Build();
  ASSERT_STATUS_OK(kernel_registry->Register(KernelCreateInfo(std::move(kernel_def), CreateTestKernel)));
  // create custom op schema for "Test" op
  std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry> schema_registry = std::make_shared<OnnxRuntimeOpSchemaRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schema = {
      ONNX_NAMESPACE::OpSchema().SetName("Test").Input(0, "X", "A N-D input tensor that is to be processed.", "T").Output(0, "Y", "desc", "T").TypeConstraint("T", ONNX_NAMESPACE::OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.").SetDomain("FakeDomain")};
  ASSERT_STATUS_OK(schema_registry->RegisterOpSet(schema, "FakeDomain", 0, 1));
  std::list<std::shared_ptr<IOnnxRuntimeOpSchemaCollection>> regs = {schema_registry};

  const std::string logger_id{"InvokerTest"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kVERBOSE, false,
      logging::LoggingManager::InstanceType::Default,
      &logger_id);
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), env));
  ORTInvoker kernel_invoker(std::move(cpu_execution_provider), env->GetLoggingManager()->DefaultLogger(), regs);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue A;
  CreateMLValue<float>(kernel_invoker.GetCurrentExecutionProvider().GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &A);
  std::vector<OrtValue> result;
  result.push_back(A);
  ASSERT_STATUS_OK(kernel_invoker.Invoke("Test", {A}, result, nullptr, "FakeDomain"));
}

}  // namespace test
}  // namespace onnxruntime
