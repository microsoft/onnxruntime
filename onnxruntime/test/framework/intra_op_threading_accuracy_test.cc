// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// End-to-end accuracy guards for the intra-op threading tuning:
//   * OrtThreadPoolParams::spin_backoff_max default 1 -> 2
//   * the ParallelFor go/no-go cost scale (ORT_PARALLEL_COST_SCALE,
//     ORT_DEFAULT_PARALLEL_COST_SCALE, see threadpool.cc)
//
// Both are latency/power tunings that must not move a single output value. The
// cost scale is the risky one: by lowering the threshold it makes ParallelFor
// split loops that previously ran whole on the calling thread, so any kernel
// whose result depends on how its loop was carved up would start drifting.
//
// These tests take one model, run it once with a single-threaded intra-op pool
// (no ParallelFor splitting at all - the reference), then re-run it across a
// range of intra-op thread counts and spin_backoff_max values, and compare every
// output against that reference.
//
// The tests live in onnxruntime_test_all, so a normal
//   build.py --build --test
// verifies accuracy as part of the build. To also cover the pre-change decision
// point, run the binary a second time with ORT_PARALLEL_COST_SCALE=1:
//   onnxruntime_test_all --gtest_filter=IntraOpThreadingAccuracy.*
// (threadpool.cc caches that lookup in a function-local static, so the scale is
// fixed for the lifetime of a process and cannot be varied from inside a test.)

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr int kOpsetVersion = 13;

// How a run's intra-op thread pool is configured.
struct ThreadingConfig {
  int intra_op_num_threads = 1;
  // nullptr leaves session.intra_op.spin_backoff_max unset, i.e. exercises
  // whatever OrtThreadPoolParams::spin_backoff_max currently defaults to.
  const char* spin_backoff_max = nullptr;

  std::string Describe() const {
    std::ostringstream out;
    out << "intra_op_num_threads=" << intra_op_num_threads
        << " spin_backoff_max=" << (spin_backoff_max == nullptr ? "<default>" : spin_backoff_max);
    return out.str();
  }
};

// The reference: a single-threaded intra-op pool. CreateThreadPoolHelper returns
// no pool for a size of 1, so every ParallelFor runs whole on the calling thread
// and the cost scale cannot influence the result.
const ThreadingConfig kReferenceConfig{/*intra_op_num_threads*/ 1, /*spin_backoff_max*/ nullptr};

constexpr std::array<int, 3> kThreadCounts{2, 4, 8};
// <default> pins the shipped default (2); 1 is the pre-change value; 8 and the
// clamped-above-limit value cover the rest of the range.
constexpr std::array<const char*, 5> kSpinBackoffMaxValues{nullptr, "1", "2", "8", "1024"};

std::vector<ThreadingConfig> ThreadingConfigsUnderTest() {
  std::vector<ThreadingConfig> configs;
  configs.reserve(kThreadCounts.size() * kSpinBackoffMaxValues.size());
  for (int num_threads : kThreadCounts) {
    for (const char* backoff : kSpinBackoffMaxValues) {
      configs.emplace_back(ThreadingConfig{num_threads, backoff});
    }
  }
  return configs;
}

// Builds the model once and serializes it, so every run below sees byte-identical
// input. ModelTestBuilder seeds its value generator deterministically, so the
// feeds are reproducible too.
class ThreadingAccuracyModel {
 public:
  // Two-phase construction: the gtest ASSERT_* macros expand to a `return`
  // statement, which a constructor is not allowed to use, so the fallible part
  // of the setup lives here. Callers wrap this in ASSERT_NO_FATAL_FAILURE.
  void Build(const std::function<void(ModelTestBuilder&)>& build_graph) {
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = kOpsetVersion;
    Model model("IntraOpThreadingAccuracy", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    ModelTestBuilder helper(model.MainGraph());
    build_graph(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    model.ToProto().SerializeToString(&model_data_);
    feeds_ = helper.feeds_;
    output_names_ = helper.output_names_;
  }

  void Run(const ThreadingConfig& config, std::vector<OrtValue>& fetches) const {
    SessionOptions session_options;
    // Pin everything except the threading knobs so the runs differ only in
    // how work is partitioned.
    session_options.graph_optimization_level = TransformerLevel::Level2;
    session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    session_options.use_per_session_threads = true;
    session_options.intra_op_param.thread_pool_size = config.intra_op_num_threads;
    if (config.spin_backoff_max != nullptr) {
      ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
          kOrtSessionOptionsConfigIntraOpSpinBackoffMax, config.spin_backoff_max));
    }

    InferenceSessionWrapper session{session_options, GetEnvironment()};
    std::istringstream model_istream(model_data_);
    ASSERT_STATUS_OK(session.Load(model_istream));
    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, feeds_, output_names_, &fetches));
    ASSERT_EQ(fetches.size(), output_names_.size());
  }

  const std::vector<std::string>& output_names() const { return output_names_; }

 private:
  std::string model_data_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
};

// Compares one float output. tolerance == 0 means bit-identical.
void ExpectFloatOutputsMatch(const OrtValue& actual, const OrtValue& expected,
                             double relative_tolerance, const std::string& context) {
  ASSERT_TRUE(actual.IsTensor()) << context;
  ASSERT_TRUE(expected.IsTensor()) << context;
  const Tensor& actual_tensor = actual.Get<Tensor>();
  const Tensor& expected_tensor = expected.Get<Tensor>();
  ASSERT_EQ(actual_tensor.DataType(), DataTypeImpl::GetType<float>()) << context;
  ASSERT_EQ(actual_tensor.Shape(), expected_tensor.Shape()) << context;

  const auto actual_data = actual_tensor.DataAsSpan<float>();
  const auto expected_data = expected_tensor.DataAsSpan<float>();
  ASSERT_EQ(actual_data.size(), expected_data.size()) << context;

  size_t mismatches = 0;
  double worst_relative_diff = 0.0;
  size_t worst_index = 0;
  for (size_t i = 0; i < actual_data.size(); ++i) {
    const float a = actual_data[i];
    const float e = expected_data[i];
    if (a == e) {
      continue;
    }
    ++mismatches;
    // The reference graphs produce finite values; any non-finite actual value is a failure.
    const double denominator = std::max(std::abs(static_cast<double>(e)), 1e-30);
    const double relative_diff =
        std::isfinite(a)
            ? std::abs(static_cast<double>(a) - static_cast<double>(e)) / denominator
            : relative_tolerance + 1.0;
    if (relative_diff > worst_relative_diff) {
      worst_relative_diff = relative_diff;
      worst_index = i;
    }
  }

  if (relative_tolerance == 0.0) {
    EXPECT_EQ(mismatches, 0u)
        << context << ": " << mismatches << " of " << actual_data.size()
        << " values changed; worst at index " << worst_index << " (" << actual_data[worst_index]
        << " vs " << expected_data[worst_index] << ", relative " << worst_relative_diff << ")";
  } else {
    EXPECT_LE(worst_relative_diff, relative_tolerance)
        << context << ": " << mismatches << " of " << actual_data.size()
        << " values differ; worst at index " << worst_index << " (" << actual_data[worst_index]
        << " vs " << expected_data[worst_index] << ")";
  }
}

void ExpectAccuracyAcrossThreadingConfigs(const std::function<void(ModelTestBuilder&)>& build_graph,
                                          double relative_tolerance) {
  ThreadingAccuracyModel model;
  ASSERT_NO_FATAL_FAILURE(model.Build(build_graph));

  std::vector<OrtValue> reference_fetches;
  ASSERT_NO_FATAL_FAILURE(model.Run(kReferenceConfig, reference_fetches));

  for (const ThreadingConfig& config : ThreadingConfigsUnderTest()) {
    std::vector<OrtValue> fetches;
    ASSERT_NO_FATAL_FAILURE(model.Run(config, fetches));
    ASSERT_EQ(fetches.size(), reference_fetches.size()) << config.Describe();
    for (size_t i = 0; i < fetches.size(); ++i) {
      const std::string context = config.Describe() + " output=" + model.output_names()[i];
      ASSERT_NO_FATAL_FAILURE(
          ExpectFloatOutputsMatch(fetches[i], reference_fetches[i], relative_tolerance, context));
    }
  }
}

// An elementwise/shape-only graph. Every op here computes each output element
// from a fixed set of input elements, independent of the loop partitioning, so
// the results must be bit-identical no matter how ParallelFor splits the work.
// Shapes are large enough (32768 elements) that the cost model genuinely has a
// parallelize/serialize choice to make.
void BuildElementwiseGraph(ModelTestBuilder& builder) {
  auto* x = builder.MakeInput<float>({64, 512}, -1.0f, 1.0f);
  auto* w = builder.MakeInitializer<float>({64, 512}, -1.0f, 1.0f);

  auto* sum = builder.MakeIntermediate();
  builder.AddNode("Add", {x, w}, {sum});
  auto* squared = builder.MakeIntermediate();
  builder.AddNode("Mul", {sum, sum}, {squared});
  auto* shifted = builder.MakeIntermediate();
  builder.AddNode("Sub", {squared, w}, {shifted});
  auto* relu = builder.MakeIntermediate();
  builder.AddNode("Relu", {shifted}, {relu});
  auto* sigmoid = builder.MakeIntermediate();
  builder.AddNode("Sigmoid", {relu}, {sigmoid});
  auto* tanh = builder.MakeIntermediate();
  builder.AddNode("Tanh", {sigmoid}, {tanh});
  auto* absolute = builder.MakeIntermediate();
  builder.AddNode("Abs", {tanh}, {absolute});
  auto* root = builder.MakeIntermediate();
  builder.AddNode("Sqrt", {absolute}, {root});
  auto* erf = builder.MakeIntermediate();
  builder.AddNode("Erf", {root}, {erf});

  // Two outputs: a divide (sigmoid is strictly positive, so no inf/nan can
  // creep in and turn a bitwise comparison into a NaN != NaN false positive)
  // and a transposed copy, so the layout pass through Transpose is covered too.
  builder.AddNode("Div", {erf, sigmoid}, {builder.MakeOutput()});
  builder.AddNode("Transpose", {erf}, {builder.MakeOutput()});
}

// A graph with the reduction-bearing ops that actually dominate inference time:
// MatMul and Gemm (MLAS, which shards over M/N) plus Softmax (per-row
// reductions). These are compared with a tolerance rather than bit-exactly,
// because splitting a reduction across threads is allowed to reassociate the
// float adds.
void BuildComputeGraph(ModelTestBuilder& builder) {
  auto* x = builder.MakeInput<float>({128, 256}, -1.0f, 1.0f);
  auto* w1 = builder.MakeInitializer<float>({256, 512}, -0.5f, 0.5f);
  auto* w2 = builder.MakeInitializer<float>({512, 256}, -0.5f, 0.5f);
  auto* bias = builder.MakeInitializer<float>({512}, -0.5f, 0.5f);

  auto* matmul1 = builder.MakeIntermediate();
  builder.AddNode("MatMul", {x, w1}, {matmul1});
  auto* biased = builder.MakeIntermediate();
  builder.AddNode("Add", {matmul1, bias}, {biased});
  auto* activated = builder.MakeIntermediate();
  builder.AddNode("Relu", {biased}, {activated});
  auto* matmul2 = builder.MakeIntermediate();
  builder.AddNode("MatMul", {activated, w2}, {matmul2});
  builder.AddNode("Softmax", {matmul2}, {builder.MakeOutput()});

  auto* gemm = builder.MakeIntermediate();
  builder.AddNode("Gemm", {x, w1, bias}, {gemm});
  builder.AddNode("Sigmoid", {gemm}, {builder.MakeOutput()});
}

}  // namespace

// Elementwise kernels have no cross-iteration dependency, so re-partitioning
// their loops - which is exactly what the lowered cost threshold causes - must
// leave every output bit-identical to the single-threaded run.
TEST(IntraOpThreadingAccuracy, ElementwiseGraphIsBitIdenticalAcrossThreadingConfigs) {
  ExpectAccuracyAcrossThreadingConfigs(BuildElementwiseGraph, /*relative_tolerance*/ 0.0);
}

// MatMul/Gemm/Softmax may legitimately reassociate their reductions when the
// work is spread over more threads, so hold them to a tight relative tolerance
// instead. A real accuracy regression (a mis-split range, a dropped tail block)
// blows past 1e-5 by orders of magnitude.
TEST(IntraOpThreadingAccuracy, ComputeGraphMatchesSingleThreadWithinTolerance) {
  ExpectAccuracyAcrossThreadingConfigs(BuildComputeGraph, /*relative_tolerance*/ 1e-5);
}

}  // namespace test
}  // namespace onnxruntime
