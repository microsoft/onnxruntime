// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// End-to-end companion to test_cases/einsum_stream_allocation_test.cc. Those tests call Einsum's CUDA
// device helpers directly; these run the CUDA Einsum kernel in a session, so they also cover how
// Einsum::ComputeInternal binds the run's assets into the CreateTensor callback it installs on the
// compute preprocessor and on each typed compute processor.
//
// The session's CUDA device allocator is replaced, through an environment allocator, by one that
// records whether each allocation was tagged with a stream. Every device allocation made during
// Run() - the input copies, the Einsum node's intermediates and scratch buffers, and its output -
// must be tagged with the run's stream.

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/stream_handles.h"
#include "core/graph/model.h"
#include "core/session/allocator_adapters.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// A stream aware CUDA device allocator that records which entry point each allocation came in on.
class StreamRecordingCudaAllocator : public IAllocator {
 public:
  explicit StreamRecordingCudaAllocator(const OrtDevice& device)
      : IAllocator(OrtMemoryInfo(CUDA, OrtDeviceAllocator, device, OrtMemTypeDefault)) {}

  bool IsStreamAware() const override { return true; }

  void* Alloc(size_t size) override {
    ++untagged_allocs;
    return CudaAlloc(size);
  }

  void* AllocOnStream(size_t size, Stream* stream) override {
    ++stream_allocs;
    streams.push_back(stream);
    return CudaAlloc(size);
  }

  void Free(void* p) override { cudaFree(p); }

  void Reset() {
    untagged_allocs = 0;
    stream_allocs = 0;
    streams.clear();
  }

  int untagged_allocs = 0;
  int stream_allocs = 0;
  std::vector<Stream*> streams;

 private:
  void* CudaAlloc(size_t size) {
    void* p = nullptr;
    return size == 0 || cudaMalloc(&p, size) == cudaSuccess ? p : nullptr;
  }
};

template <typename T>
ONNX_NAMESPACE::TensorProto_DataType ElemType();
template <>
ONNX_NAMESPACE::TensorProto_DataType ElemType<float>() { return ONNX_NAMESPACE::TensorProto_DataType_FLOAT; }
template <>
ONNX_NAMESPACE::TensorProto_DataType ElemType<double>() { return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE; }
template <>
ONNX_NAMESPACE::TensorProto_DataType ElemType<MLFloat16>() { return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16; }

struct EinsumInput {
  std::vector<int64_t> dims;
  std::vector<float> values;
};

}  // namespace

class EinsumCudaSessionAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }

    auto logging_manager = std::make_unique<logging::LoggingManager>(
        std::make_unique<logging::CLogSink>(), logging::Severity::kWARNING, false,
        logging::LoggingManager::InstanceType::Temporal);
    ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), env_));

    const OrtDevice device = DefaultCudaExecutionProvider()->GetOrtDeviceByMemType(OrtMemTypeDefault);
    recorder_ = std::make_shared<StreamRecordingCudaAllocator>(device);
    ort_allocator_ = std::make_unique<OrtAllocatorImplWrappingIAllocator>(recorder_);
    ASSERT_STATUS_OK(env_->RegisterAllocator(ort_allocator_.get()));
  }

  void TearDown() override {
    if (env_ && recorder_) {
      ASSERT_STATUS_OK(env_->UnregisterAllocator(recorder_->Info()));
    }
  }

  // Runs a single Einsum node on the CUDA EP and checks both the result and that every device
  // allocation made during Run() was tagged with one and the same stream.
  template <typename T>
  void RunEinsum(const std::string& equation, const std::vector<EinsumInput>& inputs,
                 const std::vector<int64_t>& expected_dims, const std::vector<float>& expected_values) {
    std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 12}};
    Model model("einsum", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                domain_to_version, {}, env_->GetLoggingManager()->DefaultLogger());
    auto& graph = model.MainGraph();

    ONNX_NAMESPACE::TypeProto tensor_type;
    tensor_type.mutable_tensor_type()->set_elem_type(ElemType<T>());

    std::vector<NodeArg*> node_inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      node_inputs.push_back(&graph.GetOrCreateNodeArg("X" + std::to_string(i), &tensor_type));
    }
    std::vector<NodeArg*> node_outputs{&graph.GetOrCreateNodeArg("Y", &tensor_type)};
    graph.AddNode("einsum", "Einsum", "", node_inputs, node_outputs).AddAttribute("equation", equation);
    ASSERT_STATUS_OK(graph.Resolve());

    std::string model_bytes;
    ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));

    // Keep the session as plain as possible, so that the allocations it makes during Run() are
    // predictable: no graph optimizations beyond the required ones, no memory pattern and a single
    // stream.
    SessionOptions so;
    so.graph_optimization_level = TransformerLevel::Default;
    so.enable_mem_pattern = false;
    so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1"));
    InferenceSession session{so, *env_};
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    auto cpu_allocator = CPUAllocator::DefaultInstance();
    NameMLValMap feeds;
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::vector<T> values;
      for (float v : inputs[i].values) {
        values.push_back(static_cast<T>(v));
      }
      OrtValue value;
      CreateMLValue<T>(cpu_allocator, inputs[i].dims, values, &value);
      feeds.emplace("X" + std::to_string(i), value);
    }

    const std::string context = equation + " (" + DataTypeImpl::ToString(DataTypeImpl::GetType<T>()) + ")";
    recorder_->Reset();
    RunOptions run_options;
    const std::vector<std::string> output_names{"Y"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));

    // One allocation per input copy and one for the output are made by the session; anything beyond
    // those comes from the kernel itself, so the recorder must have seen at least one of those too.
    EXPECT_EQ(recorder_->untagged_allocs, 0) << context;
    EXPECT_GT(recorder_->stream_allocs, static_cast<int>(inputs.size()) + 1) << context;
    for (Stream* stream : recorder_->streams) {
      EXPECT_NE(stream, nullptr) << context;
      EXPECT_EQ(stream, recorder_->streams.front()) << context;
    }

    ASSERT_EQ(fetches.size(), 1u);
    const auto& output = fetches[0].Get<Tensor>();
    EXPECT_EQ(output.Shape(), TensorShape(expected_dims)) << context;
    std::vector<float> actual;
    for (const T& v : output.DataAsSpan<T>()) {
      actual.push_back(static_cast<float>(v));
    }
    EXPECT_EQ(actual, expected_values) << context;
  }

  std::unique_ptr<Environment> env_;
  std::shared_ptr<StreamRecordingCudaAllocator> recorder_;
  std::unique_ptr<OrtAllocatorImplWrappingIAllocator> ort_allocator_;
};

// Preprocessing: a repeated subscript takes the diagonal.
TEST_F(EinsumCudaSessionAllocationTest, DiagonalInPreprocessing) {
  RunEinsum<float>("ii->i", {{{3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}}}, {3}, {0, 4, 8});
}

// Preprocessing: an input whose subscripts are out of order is transposed through CreateTensor.
TEST_F(EinsumCudaSessionAllocationTest, TransposeInPreprocessing) {
  RunEinsum<float>("ij,ji->ij", {{{2, 3}, {0, 1, 2, 3, 4, 5}}, {{3, 2}, {1, 1, 1, 1, 1, 1}}},
                   {2, 3}, {0, 1, 2, 3, 4, 5});
}

// Typed compute: a contraction goes through MatMul, with its output created through CreateTensor.
// Run for each of the three typed compute processors.
TEST_F(EinsumCudaSessionAllocationTest, MatMulInTypedCompute) {
  const std::vector<EinsumInput> inputs{{{2, 3}, {0, 1, 2, 3, 4, 5}},
                                        {{3, 2}, {1, 2, 3, 4, 5, 6}}};
  const std::vector<float> expected{13, 16, 40, 52};
  RunEinsum<float>("ij,jk->ik", inputs, {2, 2}, expected);
  RunEinsum<double>("ij,jk->ik", inputs, {2, 2}, expected);
  RunEinsum<MLFloat16>("ij,jk->ik", inputs, {2, 2}, expected);
}

// Typed compute: a summed out subscript goes through ReduceSum, which also takes a scratch buffer.
TEST_F(EinsumCudaSessionAllocationTest, ReduceSumInTypedCompute) {
  RunEinsum<float>("ij->i", {{{2, 3}, {0, 1, 2, 3, 4, 5}}}, {2}, {3, 12});
}

}  // namespace test
}  // namespace onnxruntime
