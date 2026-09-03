// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "core/graph/constants.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(
    T start,
    T limit,
    T delta,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output) {
  // ONNX domain opset-11
  OpTester test1("Range", 11);
  test1.AddInput<T>("start", {}, {start});
  test1.AddInput<T>("limit", {}, {limit});
  test1.AddInput<T>("delta", {}, {delta});
  test1.AddOutput<T>("output", output_dims, output);
  // TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test2("Range", 1, kMSDomain);
  test2.AddInput<T>("start", {}, {start});
  test2.AddInput<T>("limit", {}, {limit});

  if (delta != T{1})  // only contrib schema allows optional 'delta' input
    test2.AddInput<T>("delta", {}, {delta});

  test2.AddOutput<T>("output", output_dims, output);
  // TensorRT doesn't fully support opset 11 yet
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#endif
}  // namespace test

TEST(RangeTest, Int32_DeltaDefault) {
  RunTest<int32_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Int64_DeltaDefault) {
  RunTest<int64_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Float_DeltaDefault) {
  RunTest<float>(0.f, 5.f, 1.f, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
}

TEST(RangeTest, Double_DeltaDefault) {
  RunTest<double>(0., 5., 1., {5}, {0., 1., 2., 3., 4.});
}

TEST(RangeTest, Int32_Delta_NonDefault) {
  RunTest<int32_t>(0, 10, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_0) {
  RunTest<int64_t>(0, 9, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_1) {
  RunTest<int64_t>(1, 2, 2, {1}, {1});
}

TEST(RangeTest, Int32_NegativeDelta_0) {
  RunTest<int32_t>(2, -9, -2, {6}, {2, 0, -2, -4, -6, -8});
}

TEST(RangeTest, Int32_NegativeDelta_1) {
  RunTest<int32_t>(2, 9, -2, {0}, {});
}

TEST(RangeTest, Float_NegativeDelta_0) {
  RunTest<float>(2.0f, -8.1f, -2.0f, {6}, {2.0f, 0.0f, -2.0f, -4.0f, -6.0f, -8.0f});
}

TEST(RangeTest, Float_SameStartAndLimit) {
  RunTest<float>(2.0f, 2.0f, 1, {0}, {});
}

TEST(RangeTest, AlmostSameStartAndLimitHighDelta) {
  RunTest<float>(2.0f, 2.01f, 1000000.0f, {1}, {2.0f});
}

#ifndef DISABLE_CONTRIB_OPS

namespace {
// Describes a single Range input supplied as a graph initializer: its declared dimensions and
// either the exact raw_data bytes or a typed int32_data payload. Tests use this to craft
// well-formed, truncated (e.g. dims=[0] with empty raw_data), and typed-field initializers for
// the contrib Range schema. When int32_data is non-empty it is stored in the typed field instead
// of raw_data (ONNX packs sub-32-bit int types such as INT16 into int32_data).
struct RangeInputSpec {
  std::vector<int64_t> dims;
  std::string raw_data;
  std::vector<int32_t> int32_data;
};

// Serializes a scalar value to its raw_data byte representation.
template <typename T>
std::string ToRawData(T value) {
  std::string bytes(sizeof(T), '\0');
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

// A correctly-sized scalar initializer (no dims) holding a single value.
template <typename T>
RangeInputSpec ScalarInput(T value) {
  return RangeInputSpec{{}, ToRawData(value), {}};
}

// A zero-element initializer (dims=[0]) whose raw_data is empty. The declared size matches
// the (empty) payload, so initializer size validation accepts it; shape inference, however,
// still attempts to read the first element.
RangeInputSpec EmptyInput() {
  return RangeInputSpec{{0}, std::string{}, {}};
}

// A scalar initializer whose single value is carried in the typed int32_data field. ONNX packs
// INT16 (and other sub-32-bit int) initializer values into int32_data, so this exercises the
// typed-field shape-inference path rather than the raw_data path.
RangeInputSpec Int16TypedInput(int16_t value) {
  return RangeInputSpec{{}, std::string{}, {static_cast<int32_t>(value)}};
}

void AddInitializer(ONNX_NAMESPACE::GraphProto* graph, const std::string& name, int data_type,
                    const RangeInputSpec& spec) {
  auto* initializer = graph->add_initializer();
  initializer->set_name(name);
  initializer->set_data_type(data_type);
  for (int64_t dim : spec.dims) {
    initializer->add_dims(dim);
  }
  if (!spec.int32_data.empty()) {
    for (int32_t value : spec.int32_data) {
      initializer->add_int32_data(value);
    }
  } else {
    initializer->set_raw_data(spec.raw_data);
  }
}

// Builds a single com.microsoft Range node model whose start/limit/delta inputs are supplied
// as graph initializers of the given element type, then loads and initializes it in a session.
common::Status BuildAndInitializeContribRangeModel(int data_type,
                                                   const RangeInputSpec& start,
                                                   const RangeInputSpec& limit,
                                                   const RangeInputSpec& delta,
                                                   InferenceSession& session_object) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("ContribRangeGraph");

  AddInitializer(graph, "start", data_type, start);
  AddInitializer(graph, "limit", data_type, limit);
  AddInitializer(graph, "delta", data_type, delta);

  auto* output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);

  auto* range_node = graph->add_node();
  range_node->set_name("Range");
  range_node->set_op_type("Range");
  range_node->set_domain(kMSDomain);
  range_node->add_input("start");
  range_node->add_input("limit");
  range_node->add_input("delta");
  range_node->add_output("Y");

  std::string serialized_model;
  if (!model.SerializeToString(&serialized_model)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize test model.");
  }

  std::stringstream model_stream(serialized_model);
  ORT_RETURN_IF_ERROR(session_object.Load(model_stream));
  return session_object.Initialize();
}

// Returns the inferred dim_value of output Y after a successful Initialize, or -1 if absent.
int64_t GetInferredOutputDim(const InferenceSession& session_object) {
  const auto outputs = session_object.GetModelOutputs();
  if (!outputs.first.IsOK() || outputs.second->size() != 1u) {
    return -1;
  }
  const auto* shape = (*outputs.second)[0]->Shape();
  if (shape == nullptr || shape->dim_size() != 1 || !shape->dim(0).has_dim_value()) {
    return -1;
  }
  return shape->dim(0).dim_value();
}
}  // namespace

// Verifies that shape inference clamps an empty/backward range to a zero-sized dimension,
// matching the CPU kernel behavior, instead of emitting a negative dimension value.
TEST(RangeTest, ContribOp_BackwardRange_InfersZeroDim) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_BackwardRange_InfersZeroDim";
  InferenceSession session_object{so, GetEnvironment()};
  // start > limit with a positive delta yields an empty range.
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(5.0), ScalarInput(0.0),
      ScalarInput(1.0), session_object);
  ASSERT_STATUS_OK(status);
  ASSERT_EQ(GetInferredOutputDim(session_object), 0);
}

// Verifies that a correctly-sized raw_data initializer (exactly sizeof(T) bytes) loads
// successfully and produces the expected inferred dimension, so the length check does not
// over-reject valid models.
TEST(RangeTest, ContribOp_ExactSizeRawData_LoadsSuccessfully) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_ExactSizeRawData_LoadsSuccessfully";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(0.0), ScalarInput(5.0),
      ScalarInput(1.0), session_object);
  ASSERT_STATUS_OK(status);
  ASSERT_EQ(GetInferredOutputDim(session_object), 5);
}

// Verifies that shape inference handles an INT16 Range whose initializers store their values in
// the typed int32_data field (the ONNX packing for INT16) rather than raw_data. This exercises
// the int16 typed-field path in get_data<int16_t>: start=0, limit=5, delta=1 -> 5 elements.
TEST(RangeTest, ContribOp_Int16TypedField_InfersDim) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_Int16TypedField_InfersDim";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_INT16, Int16TypedInput(0), Int16TypedInput(5),
      Int16TypedInput(1), session_object);
  ASSERT_STATUS_OK(status);
  ASSERT_EQ(GetInferredOutputDim(session_object), 5);
}

// The following two tests run the contrib Range CPU kernel with runtime (non-constant) inputs.
// Because the inputs are not initializers, shape inference cannot fold the output length, so
// the element-count guards in ComputeRange are exercised at execution time rather than at load.
// They are pinned to the CPU execution provider because the contrib Range kernel is CPU-only.

// Verifies that the kernel rejects an element count that exceeds the int64 range.
TEST(RangeTest, ContribOp_Kernel_CountExceedsInt64Range_FailsAtRuntime) {
  OpTester test("Range", 1, kMSDomain);
  test.AddInput<double>("start", {}, {0.0});
  test.AddInput<double>("limit", {}, {1e19});
  test.AddInput<double>("delta", {}, {1.0});
  test.AddOutput<double>("output", {0}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Range: the computed number of elements exceeds the supported range.", {}, nullptr,
           &execution_providers);
}

// Verifies that the kernel rejects a non-finite computed element count (the difference of the
// two inputs overflows to infinity).
TEST(RangeTest, ContribOp_Kernel_NonFiniteCount_FailsAtRuntime) {
  OpTester test("Range", 1, kMSDomain);
  test.AddInput<double>("start", {}, {-1e308});
  test.AddInput<double>("limit", {}, {1e308});
  test.AddInput<double>("delta", {}, {1.0});
  test.AddOutput<double>("output", {0}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Range: the computed number of elements is not a finite value.", {}, nullptr,
           &execution_providers);
}

// The following tests exercise failure paths that surface via fail_shape_inference, which
// reports the error by throwing an inference error. They are excluded from no-exception
// builds where such a throw would abort rather than yield a failure Status.
#if !defined(ORT_NO_EXCEPTIONS)

// Verifies that Range shape inference rejects an initializer whose raw_data holds fewer bytes
// than its declared data type requires (here: a zero-element initializer with empty raw_data,
// which initializer size validation accepts), returning a clean failure status instead of
// reading more bytes than the initializer declares. One test per input position.
TEST(RangeTest, ContribOp_TruncatedStartRawData_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedStartRawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, EmptyInput(), ScalarInput(5.0),
      ScalarInput(1.0), session_object);
  ASSERT_FALSE(status.IsOK());
}

TEST(RangeTest, ContribOp_TruncatedLimitRawData_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedLimitRawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(0.0), EmptyInput(),
      ScalarInput(1.0), session_object);
  ASSERT_FALSE(status.IsOK());
}

TEST(RangeTest, ContribOp_TruncatedDeltaRawData_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedDeltaRawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(0.0), ScalarInput(5.0),
      EmptyInput(), session_object);
  ASSERT_FALSE(status.IsOK());
}

// Exercises the length check for different sizeof(T) template instantiations (float, int64).
TEST(RangeTest, ContribOp_TruncatedFloatRawData_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedFloatRawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, EmptyInput(), ScalarInput(5.0f),
      ScalarInput(1.0f), session_object);
  ASSERT_FALSE(status.IsOK());
}

TEST(RangeTest, ContribOp_TruncatedInt64RawData_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedInt64RawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_INT64, EmptyInput(), ScalarInput<int64_t>(5),
      ScalarInput<int64_t>(1), session_object);
  ASSERT_FALSE(status.IsOK());
}

// Verifies that a zero delta is rejected cleanly by shape inference.
TEST(RangeTest, ContribOp_ZeroDelta_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_ZeroDelta_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(0.0), ScalarInput(10.0),
      ScalarInput(0.0), session_object);
  ASSERT_FALSE(status.IsOK());
}

// Verifies that a finite element count that exceeds the int64 range is rejected cleanly,
// rather than reaching an out-of-range conversion (start=0, limit=1e19, delta=1).
TEST(RangeTest, ContribOp_CountExceedsInt64Range_FailsCleanly) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_CountExceedsInt64Range_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(
      ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, ScalarInput(0.0), ScalarInput(1e19),
      ScalarInput(1.0), session_object);
  ASSERT_FALSE(status.IsOK());
}

#endif  // !defined(ORT_NO_EXCEPTIONS)

#endif  // DISABLE_CONTRIB_OPS

#ifdef USE_CUDA
// The following tests exercise the standard onnx-domain Range CUDA kernel with runtime
// (non-constant) inputs so the element-count guards in the CUDA ComputeRange are evaluated at
// execution time. They are pinned to the CUDA execution provider and are skipped when a CUDA
// provider is not available in the current build/runtime. The expected failure messages match
// the CPU kernel exactly so both paths stay consistent.

// Verifies that the CUDA kernel rejects an element count that exceeds the int64 range.
TEST(RangeTest, CudaKernel_CountExceedsInt64Range_FailsAtRuntime) {
  auto cuda_provider = DefaultCudaExecutionProvider();
  if (cuda_provider == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }
  OpTester test("Range", 11);
  test.AddInput<double>("start", {}, {0.0});
  test.AddInput<double>("limit", {}, {1e19});
  test.AddInput<double>("delta", {}, {1.0});
  test.AddOutput<double>("output", {0}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_provider));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Range: the computed number of elements exceeds the supported range.", {}, nullptr,
           &execution_providers);
}

// Verifies that the CUDA kernel rejects a non-finite computed element count (the difference of
// the two inputs overflows to infinity).
TEST(RangeTest, CudaKernel_NonFiniteCount_FailsAtRuntime) {
  auto cuda_provider = DefaultCudaExecutionProvider();
  if (cuda_provider == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }
  OpTester test("Range", 11);
  test.AddInput<double>("start", {}, {-1e308});
  test.AddInput<double>("limit", {}, {1e308});
  test.AddInput<double>("delta", {}, {1.0});
  test.AddOutput<double>("output", {0}, {});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_provider));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Range: the computed number of elements is not a finite value.", {}, nullptr,
           &execution_providers);
}

// Note on large-count coverage: a test that actually materializes a valid count > INT_MAX to
// prove the int64 launch-path widening is intentionally omitted. OpTester requires a host-side
// reference output tensor of the same element count, and a value above INT_MAX would need a
// multi-GB host allocation (and matching device memory), which is impractical and OOM-prone on
// CI. The int64 widening (count, grid sizing, and the grid-stride kernel index) is instead
// covered by code review; the two guards above ensure non-finite and >= 2^63 counts are rejected
// before any launch. A device-level large-count test can be added as a separate, opt-in benchmark.
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
