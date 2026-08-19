// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <gsl/gsl>
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/util/include/current_test_name.h"
#include "test/unittest_util/model_tester.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_encoder.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(WhisperEncoderSubgraphTest, ReportsInvalidSecondInputName) {
  NodeArg encoder_input("encoder_input_ids", nullptr);
  NodeArg invalid_decoder_input("wrong_name", nullptr);

  const auto status = contrib::transformers::ValidateWhisperEncoderInputNames(encoder_input, invalid_decoder_input);

  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("got: wrong_name"));
}

TEST(BeamSearchParametersTest, SetSubgraphParametersRejectsOversizedVocabSize) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 150;

  EXPECT_THROW(parameters.SetSubgraphParameters(128, 1, 1, 1), OnnxRuntimeException);
}

TEST(BeamSearchParametersTest, SetSubgraphParametersAllowsPaddedVocabSize) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 64;

  parameters.SetSubgraphParameters(128, 2, 4, 6);

  EXPECT_EQ(parameters.vocab_size, 64);
  EXPECT_EQ(parameters.num_heads, 2);
}

TEST(BeamSearchParametersTest, SetSubgraphParametersUsesSubgraphSizeWhenAttributeIsDefault) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = -1;

  parameters.SetSubgraphParameters(128, 2, 4, 6);

  EXPECT_EQ(parameters.vocab_size, 128);
  EXPECT_EQ(parameters.num_heads, 2);
}

TEST(BeamSearchParametersTest, RejectsNegativeWhisperBeginningTimestampTokenId) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 128;
  parameters.model_type = contrib::transformers::IGenerationParameters::kModelTypeWhisper;
  parameters.logits_processor = contrib::transformers::IGenerationParameters::kLogitsProcessorTypeWhisper;
  parameters.beginning_timestamp_token_id = -1;

  EXPECT_THROW(parameters.ValidateWhisperTimestampTokenId(), OnnxRuntimeException);
}

TEST(BeamSearchParametersTest, RejectsZeroWhisperBeginningTimestampTokenId) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 128;
  parameters.model_type = contrib::transformers::IGenerationParameters::kModelTypeWhisper;
  parameters.logits_processor = contrib::transformers::IGenerationParameters::kLogitsProcessorTypeWhisper;
  parameters.beginning_timestamp_token_id = 0;

  EXPECT_THROW(parameters.ValidateWhisperTimestampTokenId(), OnnxRuntimeException);
}

TEST(BeamSearchParametersTest, RejectsWhisperBeginningTimestampTokenIdEqualToVocabSize) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 128;
  parameters.model_type = contrib::transformers::IGenerationParameters::kModelTypeWhisper;
  parameters.logits_processor = contrib::transformers::IGenerationParameters::kLogitsProcessorTypeWhisper;
  parameters.beginning_timestamp_token_id = 128;

  EXPECT_THROW(parameters.ValidateWhisperTimestampTokenId(), OnnxRuntimeException);
}

TEST(BeamSearchParametersTest, AcceptsValidWhisperBeginningTimestampTokenId) {
  contrib::transformers::BeamSearchParameters parameters;
  parameters.vocab_size = 128;
  parameters.model_type = contrib::transformers::IGenerationParameters::kModelTypeWhisper;
  parameters.logits_processor = contrib::transformers::IGenerationParameters::kLogitsProcessorTypeWhisper;
  parameters.beginning_timestamp_token_id = 1;

  EXPECT_NO_THROW(parameters.ValidateWhisperTimestampTokenId());
}

TEST(BeamSearchTest, ExpandBufferSupportsRankGreaterThanFour) {
  AllocatorPtr allocator = CPUAllocator::DefaultInstance();
  OrtValue input;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({1, 2, 3, 4, 5}), allocator, input);

  OrtValue expanded;
  ASSERT_STATUS_OK(contrib::GenerationCpuDeviceHelper::ExpandBuffer<float>(
      nullptr, input, 2, allocator, expanded, true, 0));

  EXPECT_EQ(expanded.Get<Tensor>().Shape(), TensorShape({2, 2, 3, 4, 5}));
}

namespace {

class TestSubgraph final : public contrib::transformers::Subgraph {
 public:
  TestSubgraph(const Node& node, const GraphViewer& subgraph)
      : Subgraph(node, "decoder", subgraph) {}

  using Subgraph::GetParameters;

  Status Validate(const std::vector<const NodeArg*>&,
                  const std::vector<const NodeArg*>&) override {
    return Status::OK();
  }
};

void LoadGptBeamSearchGraph(std::shared_ptr<Model>& model,
                            const Node*& beam_search_node,
                            const Graph*& decoder_graph) {
  ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx"),
                               model, nullptr, DefaultLoggingManager().DefaultLogger()));

  beam_search_node = nullptr;
  for (const Node& node : model->MainGraph().Nodes()) {
    if (node.OpType() == "BeamSearch") {
      beam_search_node = &node;
      break;
    }
  }

  ASSERT_NE(beam_search_node, nullptr);
  decoder_graph = beam_search_node->GetGraphAttribute("decoder");
  ASSERT_NE(decoder_graph, nullptr);
}

}  // namespace

TEST(BeamSearchTest, GptSubgraphRejectsMissingLogitsShape) {
  std::shared_ptr<Model> model;
  const Node* beam_search_node = nullptr;
  const Graph* decoder_graph = nullptr;
  ASSERT_NO_FATAL_FAILURE(LoadGptBeamSearchGraph(model, beam_search_node, decoder_graph));

  GraphViewer decoder_viewer(*decoder_graph);
  contrib::transformers::GptSubgraph gpt_subgraph(*beam_search_node, "decoder", decoder_viewer);
  auto outputs = decoder_viewer.GetOutputs();

  auto* logits_type = const_cast<ONNX_NAMESPACE::TypeProto*>(outputs[0]->TypeAsProto());
  ASSERT_NE(logits_type, nullptr);
  logits_type->mutable_tensor_type()->clear_shape();

  const Status status = gpt_subgraph.Validate(decoder_viewer.GetInputs(), outputs);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("logits output shape cannot be nullptr"));
}

TEST(BeamSearchTest, SubgraphParametersRejectMissingPastShape) {
  std::shared_ptr<Model> model;
  const Node* beam_search_node = nullptr;
  const Graph* decoder_graph = nullptr;
  ASSERT_NO_FATAL_FAILURE(LoadGptBeamSearchGraph(model, beam_search_node, decoder_graph));

  GraphViewer decoder_viewer(*decoder_graph);
  TestSubgraph subgraph(*beam_search_node, decoder_viewer);
  const Status status = subgraph.GetParameters(nullptr, decoder_viewer.GetOutputs()[0]->Shape(), true);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("past state shape cannot be nullptr"));
}

void RunGptBeamSearchFp32() {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2 cuda_options;
  cuda_options.use_tf32 = false;
  session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx"), session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}

TEST(BeamSearchTest, GptBeamSearchFp32) {
  RunGptBeamSearchFp32();
}

TEST(BeamSearchTest, GptBeamSearchFp32_DisableFastTopK) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{onnxruntime::contrib::transformers::kBeamSearchUseFastTopK, "0"}}};
  RunGptBeamSearchFp32();
}

TEST(BeamSearchTest, GptBeamSearchFp16) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}

TEST(BeamSearchTest, GptBeamSearchFp16_ScoresOutputTypeAndShape) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx"), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t output_count = session.GetOutputCount();
    std::string scores_output_name;
    for (size_t i = 0; i < output_count; ++i) {
      auto output_name_alloc = session.GetOutputNameAllocated(i, allocator);
      if (output_name_alloc.get() != nullptr && std::string(output_name_alloc.get()) == "scores") {
        scores_output_name = output_name_alloc.get();
        break;
      }
    }

    if (scores_output_name.empty()) {
      GTEST_SKIP() << "Skipping because tiny_gpt2_beamsearch_fp16.onnx does not expose optional 'scores' output in this test environment.";
    }

    const char* output_names[] = {scores_output_name.c_str()};

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& scores = ort_outputs[0];
    ASSERT_TRUE(scores.IsTensor());

    auto scores_ts = scores.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, scores_ts.GetElementType());

    const auto scores_shape = scores_ts.GetShape();
    ASSERT_EQ(scores_shape.size(), static_cast<size_t>(4));
    ASSERT_EQ(scores_shape[0], static_cast<int64_t>(max_length[0]) - input_ids_shape[1]);
    ASSERT_EQ(scores_shape[1], input_ids_shape[0]);
    ASSERT_EQ(scores_shape[2], num_beams[0]);
    ASSERT_GT(scores_shape[3], 0);
  }
}

TEST(BeamSearchTest, GptBeamSearchWithInitDecoderFp16) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to True as is the default option)
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}
TEST(BeamSearchTest, GptBeamSearchFp16_VocabPadded) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

    // The following model was obtained by padding the vocabulary size in testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx
    // from 1000 to 1600 (just for illustrative and testing purposes) to see if the beam search implementation can handle
    // such a scenario
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16_padded_vocab.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}

TEST(BeamSearchTest, DummyT5) {
  // dummy_t5.onnx model generated using following command:
  // python onnxruntime/test/testdata/dummy_t5_generator.py --output-path dummy_t5.onnx
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/dummy_t5.onnx"));
  tester.ConfigEp(DefaultCpuExecutionProvider());
  tester.AddInput("encoder_input_ids", {1, 5}, {14, 6, 13, 9, 7});
  tester.AddOutput("sequences", {1, 3, 10}, {2, 16, 6, 14, 1, 15, 6, 14, 1, 15, 2, 3, 4, 15, 6, 14, 1, 15, 6, 14, 2, 16, 6, 14, 1, 15, 6, 14, 1, 14});
#ifdef USE_CUDA
  tester.ConfigEp(DefaultCudaExecutionProvider());
#endif
  tester.RunWithConfig();
}

TEST(BeamSearchTest, DummyT5WithOuterScopeInitializers) {
  // dummy_t5_with_outer_scope_initializers.onnx model generated using following command:
  // python onnxruntime/test/testdata/dummy_t5_generator.py --output-path dummy_t5_with_outer_scope_initializers.onnx --move-initializers
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/dummy_t5_with_outer_scope_initializers.onnx"));
  tester.ConfigEp(DefaultCpuExecutionProvider());
  tester.AddInput("encoder_input_ids", {1, 5}, {14, 6, 13, 9, 7});
  tester.AddOutput("sequences", {1, 3, 10}, {2, 16, 6, 14, 1, 15, 6, 14, 1, 15, 2, 3, 4, 15, 6, 14, 1, 15, 6, 14, 2, 16, 6, 14, 1, 15, 6, 14, 1, 14});
#ifdef USE_CUDA
  tester.ConfigEp(DefaultCudaExecutionProvider());
#endif
  tester.RunWithConfig();
}

TEST(BeamSearchTest, DummyT5WithSequenceInputIds) {
  // dummy_t5_with_sequence_input_ids.onnx model generated using following command:
  // python onnxruntime/test/testdata/dummy_t5_generator.py --output-path dummy_t5_with_sequence_input_ids.onnx --sequence-as-input
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/dummy_t5_with_sequence_input_ids.onnx"));
  tester.ConfigEp(DefaultCpuExecutionProvider());
  tester.AddInput("encoder_input_ids", {1, 5}, {16, 17, 1, 0, 8});
  tester.AddOutput("sequences", {1, 3, 10}, {2, 19, 18, 3, 8, 8, 8, 8, 8, 8, 2, 19, 18, 3, 10, 19, 18, 3, 8, 8, 2, 19, 18, 15, 13, 13, 13, 13, 13, 13});
#ifdef USE_CUDA
  tester.ConfigEp(DefaultCudaExecutionProvider());
#endif
  tester.RunWithConfig();
}

TEST(BeamSearchTest, DummyWhisperWithSequenceInputIds) {
  // dummy_whisper_with_sequence_input_ids.onnx model generated using following command:
  // python onnxruntime/test/testdata/dummy_whisper_model_generator.py
  //     --output-path dummy_whisper_with_sequence_input_ids.onnx --sequence-as-input
  // The decoder subgraph leaves input_ids second dim symbolic, so the decoder feeds are built from the
  // running sequence (use_sequence_as_input_ids_ == true), exercising the multi-token initial feed path.
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/dummy_whisper_with_sequence_input_ids.onnx"));
  tester.ConfigEp(DefaultCpuExecutionProvider());
  tester.AddInput("input_features", {1, 8, 5},
                  {-0.3f, -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, -0.3f, -0.2f, -0.1f,
                   0.0f, 0.1f, 0.2f, 0.3f, -0.3f, -0.2f, -0.1f, 0.0f, 0.1f, 0.2f,
                   0.3f, -0.3f, -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, -0.3f, -0.2f,
                   -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, -0.3f, -0.2f, -0.1f, 0.0f, 0.1f});
  tester.AddInput("decoder_input_ids", {1, 2}, {2, 5});
  tester.AddOutput("sequences", {1, 1, 10}, {2, 5, 1, 1, 1, 1, 1, 1, 1, 1});
  tester.AddOutput<float>("scores", {1, 1}, {-0.05625312775373459f}, false /* sort_output */, 1e-4f /* rel_error */,
                          1e-4f /* abs_error */);
#ifdef USE_CUDA
  tester.ConfigEp(DefaultCudaExecutionProvider());
#endif
  tester.RunWithConfig();
}

TEST(BeamSearchTest, DummyT5PointerGenerator) {
  // dummy_t5_pointer_generator.onnx model generated using following command:
  // python onnxruntime/test/testdata/dummy_t5_generator.py --output-path dummy_t5_pointer_generator.onnx --decoder-needs-input-ids
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/dummy_t5_pointer_generator.onnx"));
  tester.ConfigEp(DefaultCpuExecutionProvider());
  tester.AddInput("encoder_input_ids", {1, 5}, {14, 6, 13, 9, 7});
  tester.AddOutput("sequences", {1, 3, 10}, {2, 3, 6, 7, 3, 6, 7, 18, 3, 6, 2, 3, 6, 7, 18, 3, 6, 7, 18, 3, 2, 3, 6, 7, 3, 6, 7, 3, 6, 7});
#ifdef USE_CUDA
  tester.ConfigEp(DefaultCudaExecutionProvider());
#endif
  tester.RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
