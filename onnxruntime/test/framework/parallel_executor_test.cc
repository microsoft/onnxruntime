// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "test/providers/provider_test_utils.h"
#include "test_utils.h"
#include "core/session/inference_session.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

// Test kernel that will return success, or failure, or throw based on the input
struct TestOp {
  static constexpr const char* OpName = "TestOp";
  static constexpr const char* OpDomain = "testing";

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Return success, error, or throw based on the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .Input(0, "action", "Action to take.", "T", OpSchema::Single)
        .Output(0, "action_out", "Return input as is", "T", OpSchema::Single)
        .TypeConstraint("T", {"tensor(int64)"}, "Type of the action and values component");
    return schema;
  }

  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      const Tensor& action_tensor = *ctx->Input<Tensor>(0);
      const int64_t* action = action_tensor.Data<int64_t>();

      Status status = Status::OK();

      switch (*action) {
        case 0: {
          // success
          Tensor* Y = ctx->Output(0, action_tensor.Shape());
          void* target = Y->MutableData<int64_t>();
          memcpy(target, action, action_tensor.SizeInBytes());
          break;
        }
        case 1: {
          // fail
          status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Action was ", *action);
          break;
        }
        default: {
          ORT_THROW("Throwing as action was ", *action);
        }
      }

      return status;
    }
  };

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .Provider(onnxruntime::kCpuExecutionProvider);

    return def;
  }
};

// test that the status from TestOp is correctly returned from InferenceSession::Run
TEST(ParallelExecutor, TestStatusPropagation) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) { return new typename TestOp::OpKernelImpl(info); };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  {  // test success
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*success*/ 0});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    // TensorRT doesn't handle a custom op. Possibly it should, but that would be a separate PR
    tester.Run(OpTester::ExpectResult::kExpectSuccess, {}, {kTensorrtExecutionProvider}, nullptr, nullptr,
               ExecutionMode::ORT_PARALLEL);
  }

  {  // test failure
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*failure*/ 1});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    tester.Run(OpTester::ExpectResult::kExpectFailure, "Action was 1", {kTensorrtExecutionProvider}, nullptr, nullptr,
               ExecutionMode::ORT_PARALLEL);
  }

  {  // test exception
    OpTester tester{"TestOp", 10, TestOp::OpDomain};
    tester.AddCustomOpRegistry(registry);

    tester.AddInput<int64_t>("action", {1}, {/*exception*/ 2});
    tester.AddOutput<int64_t>("action_out", {1}, {0});
    tester.Run(OpTester::ExpectResult::kExpectFailure, "Throwing as action was 2", {kTensorrtExecutionProvider}, nullptr, nullptr, ExecutionMode::ORT_PARALLEL);
  }
}

class ParallelExecutorThreadPoolTest : public testing::TestWithParam<int> {
};

TEST_P(ParallelExecutorThreadPoolTest, TestNullInterOpThreadPool) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) { return new typename TestOp::OpKernelImpl(info); };
  auto kernel_def = TestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  OpTester tester{"TestOp", 10, TestOp::OpDomain};
  tester.AddCustomOpRegistry(registry);

  tester.AddInput<int64_t>("action", {1}, {/*success*/ 0});
  tester.AddOutput<int64_t>("action_out", {1}, {0});
  // TensorRT doesn't handle a custom op. Possibly it should, but that would be a separate PR
  onnxruntime::SessionOptions so;
  so.session_logid = "TestOp";
  so.session_log_verbosity_level = 1;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  so.inter_op_param.thread_pool_size = GetParam();
  tester.Run(so, OpTester::ExpectResult::kExpectSuccess, {}, {kTensorrtExecutionProvider}, nullptr, nullptr);
}

INSTANTIATE_TEST_SUITE_P(ParallelExecutorThreadPoolTests, ParallelExecutorThreadPoolTest,
                        testing::Values(1, 0));
}  // namespace test
}  // namespace onnxruntime
