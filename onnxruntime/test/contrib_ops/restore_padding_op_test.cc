// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <type_traits>

#include "contrib_ops/cuda/bert/bert_padding.h"
#include "core/graph/model.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

static void RunRestorePadding(
    const std::vector<float>& input_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& token_offset_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int total_tokens,
    bool use_float16 = false,
    const bool disable_cpu = true,
    const bool disable_cuda = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cpu || enable_cuda) {
    OpTester tester("RestorePadding", 1, onnxruntime::kMSDomain);

    // shape of inputs:
    //   input:                (total_tokens, hidden_size)
    //   token_offset:         (batch_size, sequence_length)
    // shape of outputs:
    //   output:               (batch_size, sequence_length, hidden_size)
    std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> input_dims = {total_tokens, hidden_size};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset_data);

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunRestorePaddingTests(
    const std::vector<float>& input_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& token_offset_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int total_tokens) {
  bool use_float16 = false;
  constexpr bool disable_cpu = true;
  constexpr bool disable_cuda = false;
  RunRestorePadding(input_data, output_data, token_offset_data, batch_size, sequence_length, hidden_size, total_tokens,
                    use_float16, disable_cpu, disable_cuda);

  use_float16 = true;
  RunRestorePadding(input_data, output_data, token_offset_data, batch_size, sequence_length, hidden_size, total_tokens,
                    use_float16, disable_cpu, disable_cuda);
}

TEST(RestorePaddingTest, RestorePaddingBatch1_NoPadding) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int total_tokens = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> output_data = input_data;

  std::vector<int32_t> token_offset_data = {0, 1};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

TEST(RestorePaddingTest, InvalidTokenOffset_CUDA) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "RestorePadding token offset validation requires a CUDA device.";
  }

  constexpr int kBatchSize = 1;
  constexpr int kSequenceLength = 2;
  constexpr int kHiddenSize = 8;
  constexpr int kTotalTokens = 1;
  const std::vector<bool> use_float16_values =
      HasCudaEnvironment(530) ? std::vector<bool>{false, true} : std::vector<bool>{false};

  for (const bool use_float16 : use_float16_values) {
    for (const std::vector<int32_t>& token_offset : {
             std::vector<int32_t>{-1, 1},
             std::vector<int32_t>{0, kBatchSize * kSequenceLength}}) {
      SCOPED_TRACE(use_float16 ? "float16" : "float");
      SCOPED_TRACE(token_offset[0]);
      SCOPED_TRACE(token_offset[1]);

      OpTester tester("RestorePadding", 1, onnxruntime::kMSDomain);
      if (use_float16) {
        tester.AddInput<MLFloat16>("input", {kTotalTokens, kHiddenSize},
                                   std::vector<MLFloat16>(kHiddenSize));
        tester.AddOutput<MLFloat16>(
            "output", {kBatchSize, kSequenceLength, kHiddenSize},
            std::vector<MLFloat16>(kBatchSize * kSequenceLength * kHiddenSize));
      } else {
        tester.AddInput<float>("input", {kTotalTokens, kHiddenSize},
                               std::vector<float>(kHiddenSize));
        tester.AddOutput<float>(
            "output", {kBatchSize, kSequenceLength, kHiddenSize},
            std::vector<float>(kBatchSize * kSequenceLength * kHiddenSize));
      }
      tester.AddInput<int32_t>(
          "token_offset", {kBatchSize, kSequenceLength}, token_offset);

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(
          OpTester::ExpectResult::kExpectFailure,
          "token_offset values must be in [0, batch_size * sequence_length).",
          {}, nullptr, &execution_providers);
    }
  }
}

TEST(RestorePaddingTest, TokenOffsetValidationIndexProgressionUsesInt64) {
  constexpr int64_t index = 2147221504;
  constexpr int64_t next_index =
      AdvancePaddingTokenOffsetValidationIndex(index, 1024, 256);

  EXPECT_EQ(next_index, 2147483648LL);
  EXPECT_GT(next_index, std::numeric_limits<int32_t>::max());
}

TEST(RestorePaddingTest, ValidTokenOffsetCudaGraphCaptureAndReplay) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "RestorePadding CUDA graph test requires a CUDA device.";
  }

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSequenceLength = 2;
  constexpr int64_t kHiddenSize = 4;
  constexpr int64_t kTotalTokens = 2;

  auto make_tensor_type = [](int32_t element_type, std::initializer_list<int64_t> shape) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    for (int64_t dimension : shape) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
    return type;
  };

  auto input_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kTotalTokens, kHiddenSize});
  auto token_offset_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, {kBatchSize, kSequenceLength});
  auto output_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
      {kBatchSize, kSequenceLength, kHiddenSize});

  Model model("restore_padding_cuda_graph", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("input", &input_type),
      &graph.GetOrCreateNodeArg("token_offset", &token_offset_type)};
  std::vector<NodeArg*> outputs{
      &graph.GetOrCreateNodeArg("output", &output_type)};
  graph.AddNode("restore_padding", "RestorePadding", "", inputs, outputs, nullptr, kMSDomain);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.enable_cuda_graph = true;
  auto cuda_ep = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(cuda_ep, nullptr);
  IExecutionProvider* cuda_ep_ptr = cuda_ep.get();

  SessionOptions session_options;
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto gpu_allocators = cuda_ep_ptr->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu_allocator, gpu_allocators.end());
  auto allocator = session.GetAllocator((*gpu_allocator)->Info());
  ASSERT_NE(allocator, nullptr);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Tensor cpu_tensor(
        DataTypeImpl::GetType<Element>(), shape,
        const_cast<Element*>(values.data()), cpu_allocator->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, allocator);
    ORT_THROW_IF_ERROR(cuda_ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };

  auto input = make_gpu_value(
      std::vector<float>(kTotalTokens * kHiddenSize),
      TensorShape{kTotalTokens, kHiddenSize});
  auto token_offset = make_gpu_value(
      std::vector<int32_t>{0, 1}, TensorShape{kBatchSize, kSequenceLength});
  Tensor output_tensor(
      DataTypeImpl::GetType<float>(),
      TensorShape{kBatchSize, kSequenceLength, kHiddenSize}, allocator);
  OrtValue output;
  Tensor::InitOrtValue(std::move(output_tensor), output);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("input", input));
  ASSERT_STATUS_OK(binding->BindInput("token_offset", token_offset));
  ASSERT_STATUS_OK(binding->BindOutput("output", output));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());

  RunOptions run_options;
  ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
  for (int i = 0; i < 4; ++i) {
    ASSERT_STATUS_OK(session.Run(run_options, *binding));
  }
  EXPECT_TRUE(cuda_ep_ptr->IsGraphCaptured(1));
}

TEST(RestorePaddingTest, RestorePaddingBatch3_TwoWithPadding) {
  int batch_size = 3;
  int sequence_length = 4;
  int hidden_size = 8;
  int total_tokens = 7;

  std::vector<float> output_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.735f, 0.835f, 5.9f, 6.0f, 6.1f, 6.2f, 6.3f, 6.4f};

  std::vector<int32_t> sequence_token_count_data = {1, 2, 4};

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.735f, 0.835f, 5.9f, 6.0f, 6.1f, 6.2f, 6.3f, 6.4f};

  std::vector<int32_t> token_offset_data = {0, 4, 5, 8, 9, 10, 11, 1, 2, 3, 6, 7};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

TEST(RestorePaddingTest, RestorePaddingBatch3_AllWithPadding) {
  int batch_size = 3;
  int sequence_length = 4;
  int hidden_size = 8;
  int total_tokens = 6;

  std::vector<float> output_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f};

  std::vector<int32_t> token_offset_data = {0, 4, 5, 8, 9, 10, 1, 2, 3, 6, 7, 11};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

}  // namespace test
}  // namespace onnxruntime
