// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <atomic>
#include <sstream>
#include <thread>

#include "core/framework/cancellation.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"

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

struct CancellationTestOp {
  static constexpr const char* OpName = "CancellationTestOp";
  static constexpr const char* OpDomain = "testing";

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Wait for run cancellation when requested by the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .Input(0, "wait_for_cancellation", "Whether to wait for cancellation.", "T", OpSchema::Single)
        .Output(0, "output", "The input value.", "T", OpSchema::Single)
        .TypeConstraint("T", {"tensor(int64)"}, "Constrain input and output types to int64 tensors.");
    return schema;
  }

  class OpKernelImpl final : public OpKernel {
   public:
    explicit OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      const Tensor& input = *ctx->Input<Tensor>(0);
      if (*input.Data<int64_t>() == 0) {
        Tensor* output = ctx->Output(0, input.Shape());
        memcpy(output->MutableData<int64_t>(), input.DataRaw(), input.SizeInBytes());
        return Status::OK();
      }

      const auto terminate_token =
          static_cast<OpKernelContextInternal*>(ctx)->GetCancellationToken();
      started_count_.fetch_add(1, std::memory_order_release);

      std::atomic<bool> stop_requested{terminate_token.stop_requested()};
      onnxruntime::CancellationCallback callback(terminate_token, [&]() {
        stop_requested.store(true, std::memory_order_release);
        stop_requested.notify_one();
      });
      stop_requested.wait(false, std::memory_order_acquire);

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exiting due to terminate flag being set to true.");
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

  static void ResetStartedCount() {
    started_count_.store(0, std::memory_order_relaxed);
  }

  static bool WaitForStartedCount(int expected_count,
                                  const std::atomic<int>& completed_count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (started_count_.load(std::memory_order_acquire) < expected_count) {
      if (completed_count.load(std::memory_order_acquire) == expected_count ||
          std::chrono::steady_clock::now() >= deadline) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    return true;
  }

 private:
  inline static std::atomic<int> started_count_{0};
};

// test that the status from TestOp is correctly returned from InferenceSession::Run
TEST(ParallelExecutor, TestStatusPropagation) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) { out = std::make_unique<typename TestOp::OpKernelImpl>(info); return Status::OK(); };
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

TEST(ParallelExecutor, ConcurrentRunCancellationFanoutAndReset) {
  constexpr const char* kCancellationMessage = "Exiting due to terminate flag being set to true.";

  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{CancellationTestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(
                   schemas, CancellationTestOp::OpDomain, 10, 11))
                  .IsOK())
      << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info,
                                       std::unique_ptr<OpKernel>& out) {
    out = std::make_unique<CancellationTestOp::OpKernelImpl>(info);
    return Status::OK();
  };
  auto kernel_def = CancellationTestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model_proto.add_opset_import();
  opset->set_domain(CancellationTestOp::OpDomain);
  opset->set_version(10);

  auto* graph = model_proto.mutable_graph();
  graph->set_name("cancellation_test");
  auto* input = graph->add_input();
  input->set_name("wait_for_cancellation");
  auto* input_tensor_type = input->mutable_type()->mutable_tensor_type();
  input_tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  input_tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_tensor_type = output->mutable_type()->mutable_tensor_type();
  output_tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  output_tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  auto* node = graph->add_node();
  node->set_name("node1");
  node->set_op_type(CancellationTestOp::OpName);
  node->set_domain(CancellationTestOp::OpDomain);
  node->add_input("wait_for_cancellation");
  node->add_output("output");

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));
  std::istringstream model_stream{model_data};

  // Use one CPU-only session directly so this test still executes when OpTester
  // restricts provider enumeration, such as in TensorRT builds.
  SessionOptions session_options;
  InferenceSession session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  OrtValue wait_input;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                         {1}, {1}, &wait_input);
  NameMLValMap wait_feeds{{"wait_for_cancellation", wait_input}};
  const std::vector<std::string> output_names{"output"};

  RunOptions run_options;
  CancellationTestOp::ResetStartedCount();
  std::atomic<int> completed_count{0};
  std::array<Status, 2> run_statuses;
  auto run_until_cancelled = [&](size_t run_index) {
    std::vector<OrtValue> fetches;
    run_statuses[run_index] = session.Run(run_options, wait_feeds, output_names, &fetches);
    completed_count.fetch_add(1, std::memory_order_release);
  };

  std::thread first_run(run_until_cancelled, 0);
  std::thread second_run(run_until_cancelled, 1);
  const bool both_runs_started = CancellationTestOp::WaitForStartedCount(2, completed_count);
  run_options.RequestTerminate();
  first_run.join();
  second_run.join();
  ASSERT_TRUE(both_runs_started);
  for (const auto& run_status : run_statuses) {
    EXPECT_FALSE(run_status.IsOK());
    EXPECT_NE(run_status.ErrorMessage().find(kCancellationMessage), std::string::npos);
  }

  run_options.ResetTerminate();
  OrtValue no_wait_input;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                         {1}, {0}, &no_wait_input);
  NameMLValMap no_wait_feeds{{"wait_for_cancellation", no_wait_input}};
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(run_options, no_wait_feeds, output_names, &fetches));
  ASSERT_EQ(fetches.size(), 1U);
  EXPECT_EQ(fetches[0].Get<Tensor>().Data<int64_t>()[0], 0);
}

class ParallelExecutorThreadPoolTest : public testing::TestWithParam<int> {
};

TEST_P(ParallelExecutorThreadPoolTest, TestNullInterOpThreadPool) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{TestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, TestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) { out = std::make_unique<typename TestOp::OpKernelImpl>(info); return Status::OK(); };
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
