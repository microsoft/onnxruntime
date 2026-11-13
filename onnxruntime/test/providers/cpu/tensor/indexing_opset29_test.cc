// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/framework/customregistry.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

// Keep unreleased schemas local to these tests. Production activation must wait
// for the ONNX dependency update; these schemas exercise the real opset-29 kernels.
std::shared_ptr<CustomRegistry> IndexingSchemas() {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas;
  for (const char* name : {"GatherND", "ScatterND", "ScatterElements"}) {
    const bool gather = std::string_view(name) == "GatherND";
    const auto* previous = ONNX_NAMESPACE::OpSchemaRegistry::Schema(name, gather ? 13 : 18, kOnnxDomain);
    ORT_ENFORCE(previous != nullptr);
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetName(name)
        .SetDomain(kOnnxDomain)
        .SinceVersion(29)
        .Input(0, "data", "", "T")
        .Input(1, "indices", "", "Tind")
        .Output(0, "output", "", "T")
        .TypeConstraint("T", ONNX_NAMESPACE::OpSchema::all_tensor_types_ir4(), "")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "")
        .TypeAndShapeInferenceFunction(previous->GetTypeAndShapeInferenceFunction());
    if (gather) {
      schema.Attr("batch_dims", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0});
    } else {
      schema.Input(2, "updates", "", "T")
          .Attr("reduction", "", ONNX_NAMESPACE::AttributeProto::STRING, std::string("none"));
      if (std::string_view(name) == "ScatterElements") {
        schema.Attr("axis", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0});
      }
    }
    schemas.push_back(std::move(schema));
  }
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 1, 29));
  return registry;
}

class IndexingOpset29Test : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (GetParam() && !DefaultCudaExecutionProvider()) {
      GTEST_SKIP() << "CUDA execution provider is unavailable";
    }
  }

  void Prepare(OpTester& test) {
    test.SetAllowUnreleasedOnnxOpset();
    test.AddCustomOpRegistry(IndexingSchemas());
  }

  void Run(OpTester& test, bool deterministic = false) {
    SessionOptions options;
    options.use_deterministic_compute = deterministic;
    if (GetParam()) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    }
    test.Config(options)
        .ConfigEp(GetParam() ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider())
        .RunWithConfig();
  }

  template <typename T>
  void CheckIntegerMean(T low, T high, T midpoint) {
    for (const char* op : {"ScatterND", "ScatterElements"}) {
      OpTester test(op, 29);
      Prepare(test);
      test.AddAttribute("reduction", std::string("mean"));
      test.AddInput<T>("data", {4}, {high, low, high, low});
      const std::vector<int64_t> shape = std::string_view(op) == "ScatterND" ? std::vector<int64_t>{6, 1}
                                                                             : std::vector<int64_t>{6};
      test.AddInput<int32_t>("indices", shape, {0, 0, 1, 1, 3, 3});
      test.AddInput<T>("updates", {6}, {high, high, high, low, low, low});
      test.AddOutput<T>("output", {4}, {high, midpoint, high, low});
      Run(test);
    }
  }
};

TEST_P(IndexingOpset29Test, GatherNDInt32BatchedNegativeIndices) {
  OpTester test("GatherND", 29);
  Prepare(test);
  test.AddAttribute("batch_dims", int64_t{1});
  test.AddInput<float>("data", {2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
  test.AddInput<int32_t>("indices", {2, 2, 1}, {-1, 0, 0, -1});
  test.AddOutput<float>("output", {2, 2, 2}, {2, 3, 0, 1, 4, 5, 6, 7});
  Run(test);
}

TEST_P(IndexingOpset29Test, ScatterNDInt32Slices) {
  OpTester test("ScatterND", 29);
  Prepare(test);
  test.AddInput<float>("data", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int32_t>("indices", {2, 1}, {-3, -1});
  test.AddInput<float>("updates", {2, 2}, {10, 20, 50, 60});
  test.AddOutput<float>("output", {3, 2}, {10, 20, 3, 4, 50, 60});
  Run(test);
}

TEST_P(IndexingOpset29Test, ScatterNDMeanDuplicateSlices) {
  for (bool deterministic : {false, true}) {
    OpTester test("ScatterND", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<float>("data", {3, 2}, {1, 2, 3, 4, 5, 6});
    test.AddInput<int32_t>("indices", {3, 1}, {0, -3, -1});
    test.AddInput<float>("updates", {3, 2}, {3, 4, 5, 6, 7, 8});
    test.AddOutput<float>("output", {3, 2}, {3, 4, 3, 4, 6, 7});
    Run(test, deterministic);
  }
}

TEST_P(IndexingOpset29Test, ScatterNDInt32ExistingReductions) {
  struct Case {
    const char* reduction;
    std::vector<float> expected;
  };
  const Case cases[] = {
      {"add", {7, 8}},
      {"mul", {12, 12}},
      {"min", {3, 2}},
      {"max", {4, 6}},
  };
  for (const auto& entry : cases) {
    OpTester test("ScatterND", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string(entry.reduction));
    test.AddInput<float>("data", {2}, {3, 6});
    test.AddInput<int32_t>("indices", {2, 1}, {-2, -1});
    test.AddInput<float>("updates", {2}, {4, 2});
    test.AddOutput<float>("output", {2}, entry.expected);
    Run(test);
  }
}

TEST_P(IndexingOpset29Test, ScatterNDMeanEmptyTuples) {
  OpTester test("ScatterND", 29);
  Prepare(test);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddInput<float>("data", {2}, {1, 2});
  test.AddInput<int64_t>("indices", {2, 0}, {});
  test.AddInput<float>("updates", {2, 2}, {3, 4, 5, 6});
  test.AddOutput<float>("output", {2}, {3, 4});
  Run(test);
}

TEST_P(IndexingOpset29Test, ScatterElementsMeanNegativeAxisAndPartialShape) {
  for (bool deterministic : {false, true}) {
    OpTester test("ScatterElements", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddAttribute("axis", int64_t{-1});
    test.AddInput<float>("data", {2, 3}, {1, 2, 3, 4, 5, 6});
    test.AddInput<int64_t>("indices", {1, 4}, {0, -3, 2, -1});
    test.AddInput<float>("updates", {1, 4}, {3, 5, 6, 9});
    test.AddOutput<float>("output", {2, 3}, {3, 2, 6, 4, 5, 6});
    Run(test, deterministic);
  }
}

TEST_P(IndexingOpset29Test, IntegerMeanDoesNotOverflowAndFloorsNegativeResults) {
  CheckIntegerMean<int8_t>(-128, 127, -43);
  CheckIntegerMean<uint8_t>(0, 255, 85);
  CheckIntegerMean<int16_t>(-32768, 32767, -10923);
  CheckIntegerMean<uint16_t>(0, 65535, 21845);
  CheckIntegerMean<int32_t>(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), -715827883);
  CheckIntegerMean<uint32_t>(0, std::numeric_limits<uint32_t>::max(), 1431655765);
  CheckIntegerMean<int64_t>(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(),
                            -3074457345618258603LL);
  CheckIntegerMean<uint64_t>(0, std::numeric_limits<uint64_t>::max(), 6148914691236517205ULL);
}

TEST_P(IndexingOpset29Test, FloatingMeanOverflowAndNonfiniteValues) {
  const double max = std::numeric_limits<double>::max();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const char* op : {"ScatterND", "ScatterElements"}) {
    OpTester test(op, 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<double>("data", {6}, {max, max, inf, inf, nan, -0.0});
    const std::vector<int64_t> shape = std::string_view(op) == "ScatterND" ? std::vector<int64_t>{6, 1}
                                                                           : std::vector<int64_t>{6};
    test.AddInput<int64_t>("indices", shape, {0, 0, 1, 2, 3, 4});
    test.AddInput<double>("updates", {6}, {max, -max, max, 1, -inf, 1});
    test.AddOutput<double>("output", {6}, {max / 3, max, inf, nan, nan, -0.0});
    Run(test);
  }
}

TEST_P(IndexingOpset29Test, HalfMeanAccumulatesBeforeNarrowing) {
  OpTester test("ScatterND", 29);
  Prepare(test);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddInput<MLFloat16>("data", {1}, {MLFloat16(65504.0f)});
  test.AddInput<int32_t>("indices", {2, 1}, {0, 0});
  test.AddInput<MLFloat16>("updates", {2}, {MLFloat16(65504.0f), MLFloat16(65504.0f)});
  test.AddOutput<MLFloat16>("output", {1}, {MLFloat16(65504.0f)});
  Run(test);
}

TEST_P(IndexingOpset29Test, FloatMeanRecoversFiniteAccumulatorOverflow) {
  const float max = std::numeric_limits<float>::max();
  for (const char* op : {"ScatterND", "ScatterElements"}) {
    OpTester test(op, 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<float>("data", {2}, {max, max});
    const std::vector<int64_t> shape = std::string_view(op) == "ScatterND" ? std::vector<int64_t>{4, 1}
                                                                           : std::vector<int64_t>{4};
    test.AddInput<int32_t>("indices", shape, {0, 0, 1, 1});
    test.AddInput<float>("updates", {4}, {max, max, max, -max});
    test.AddOutput<float>("output", {2}, {max, max / 3});
    Run(test);
  }
}

TEST_P(IndexingOpset29Test, BFloatMeanAccumulatesBeforeNarrowing) {
  OpTester test("ScatterElements", 29);
  Prepare(test);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddInput<BFloat16>("data", {1}, {BFloat16(3.0f)});
  test.AddInput<int32_t>("indices", {2}, {0, 0});
  test.AddInput<BFloat16>("updates", {2}, {BFloat16(6.0f), BFloat16(9.0f)});
  test.AddOutput<BFloat16>("output", {1}, {BFloat16(6.0f)});
  Run(test);
}

TEST_P(IndexingOpset29Test, EmptyMeanUpdatesPreserveInput) {
  for (const char* op : {"ScatterND", "ScatterElements"}) {
    OpTester test(op, 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<int64_t>("data", {2}, {std::numeric_limits<int64_t>::max(), -7});
    const std::vector<int64_t> shape = std::string_view(op) == "ScatterND" ? std::vector<int64_t>{0, 1}
                                                                           : std::vector<int64_t>{0};
    test.AddInput<int32_t>("indices", shape, {});
    test.AddInput<int64_t>("updates", {0}, {});
    test.AddOutput<int64_t>("output", {2}, {std::numeric_limits<int64_t>::max(), -7});
    Run(test);
  }
}

TEST_P(IndexingOpset29Test, MeanRejectsBoolEvenWithEmptyUpdates) {
  for (const char* op : {"ScatterND", "ScatterElements"}) {
    OpTester test(op, 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<bool>("data", {1}, {true});
    const std::vector<int64_t> shape = std::string_view(op) == "ScatterND" ? std::vector<int64_t>{0, 1}
                                                                           : std::vector<int64_t>{0};
    test.AddInput<int32_t>("indices", shape, {});
    test.AddInput<bool>("updates", {0}, {});
    test.AddOutput<bool>("output", {1}, {true});
    test.Config(OpTester::ExpectResult::kExpectFailure, "cast data and updates");
    Run(test);
  }
}

TEST_P(IndexingOpset29Test, ScatterNDMeanManyDuplicateSlices) {
  for (bool deterministic : {false, true}) {
    OpTester test("ScatterND", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    constexpr int64_t count = 4096;
    constexpr int64_t width = 32;
    test.AddInput<float>("data", {2, width}, std::vector<float>(2 * width, 3));
    test.AddInput<int32_t>("indices", {count, 1}, std::vector<int32_t>(count, -2));
    test.AddInput<float>("updates", {count, width}, std::vector<float>(count * width, 3));
    test.AddOutput<float>("output", {2, width}, std::vector<float>(2 * width, 3));
    test.SetNumRunCalls(3);
    Run(test, deterministic);
  }
}

TEST_P(IndexingOpset29Test, ScatterNDMeanDenseNumericEdges) {
  const auto check = [this](auto max_value) {
    using T = decltype(max_value);
    constexpr int64_t count = 512;
    constexpr int64_t width = 32;
    const T inf(std::numeric_limits<float>::infinity());
    const T nan(std::numeric_limits<float>::quiet_NaN());
    std::vector<T> data(4 * width, T(7.0f));
    data[0] = max_value;
    data[1] = max_value;
    data[2] = inf;
    data[3] = inf;
    data[4] = nan;
    data[5] = T(3.0f);
    std::vector<T> updates(count * width, T(3.0f));
    for (int64_t i = 0; i < count; ++i) {
      updates[i * width] = max_value;
      updates[i * width + 1] = i % 2 == 0 ? max_value : T(-static_cast<float>(max_value));
      updates[i * width + 3] = T(-std::numeric_limits<float>::infinity());
    }
    auto expected = data;
    expected[1] = T(static_cast<float>(max_value) / static_cast<float>(count + 1));
    expected[3] = nan;
    for (int64_t i = 6; i < width; ++i) {
      expected[i] = T((7.0f + count * 3.0f) / static_cast<float>(count + 1));
    }
    for (bool deterministic : {false, true}) {
      OpTester test("ScatterND", 29);
      Prepare(test);
      test.AddAttribute("reduction", std::string("mean"));
      test.AddInput<T>("data", {4, width}, data);
      test.AddInput<int64_t>("indices", {count, 1}, std::vector<int64_t>(count, -4));
      test.AddInput<T>("updates", {count, width}, updates);
      test.AddOutput<T>("output", {4, width}, expected);
      Run(test, deterministic);
    }
  };
  check(std::numeric_limits<float>::max());
  check(MLFloat16(65504.0f));
  check(BFloat16::FromBits(0x7f7f));
}

TEST_P(IndexingOpset29Test, ScatterNDMeanDenseEmptyTuples) {
  OpTester test("ScatterND", 29);
  Prepare(test);
  test.AddAttribute("reduction", std::string("mean"));
  constexpr int64_t count = 256;
  test.AddInput<float>("data", {1}, {1});
  test.AddInput<int32_t>("indices", {count, 0}, {});
  test.AddInput<float>("updates", {count, 1}, std::vector<float>(count, 3));
  test.AddOutput<float>("output", {1}, {(1.0f + count * 3.0f) / (count + 1)});
  Run(test, true);
}

TEST_P(IndexingOpset29Test, ScatterNDMeanDenseMultipleDestinations) {
  constexpr int64_t count = 512;
  constexpr int64_t width = 3;
  std::vector<int32_t> indices(count);
  std::vector<float> updates(count * width);
  for (int64_t i = 0; i < count; ++i) {
    const int32_t choices[] = {0, -3, 2, -1};
    indices[i] = choices[i % 4];
    for (int64_t j = 0; j < width; ++j) {
      updates[i * width + j] = (i % 4 < 2 ? 3.0f : 5.0f) + static_cast<float>(j);
    }
  }
  for (bool deterministic : {false, true}) {
    OpTester test("ScatterND", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    test.AddInput<float>("data", {3, width}, {1, 1, 1, 7, 8, 9, 2, 2, 2});
    test.AddInput<int32_t>("indices", {count, 1}, indices);
    test.AddInput<float>("updates", {count, width}, updates);
    test.AddOutput<float>("output", {3, width},
                          {769.0f / 257, 1025.0f / 257, 1281.0f / 257, 7, 8, 9,
                           1282.0f / 257, 1538.0f / 257, 1794.0f / 257});
    Run(test, deterministic);
  }
}

TEST_P(IndexingOpset29Test, ScatterNDMeanDenseInvalidIndices) {
  for (bool deterministic : {false, true}) {
    OpTester test("ScatterND", 29);
    Prepare(test);
    test.AddAttribute("reduction", std::string("mean"));
    constexpr int64_t count = 512;
    std::vector<int32_t> indices(count, -2);
    indices[3] = 2;
    indices[300] = -3;
    std::vector<float> updates(count, 3);
    updates[3] = 100;
    updates[300] = 100;
    test.AddInput<float>("data", {2}, {1, 2});
    test.AddInput<int32_t>("indices", {count, 1}, indices);
    test.AddInput<float>("updates", {count}, updates);
    test.AddOutput<float>("output", {2}, {(1.0f + (count - 2) * 3.0f) / (count - 1), 2});
    if (!GetParam()) {
      test.Config(OpTester::ExpectResult::kExpectFailure, "invalid indice found");
    }
    Run(test, deterministic);
  }
}

TEST_P(IndexingOpset29Test, GatherNDInt32EmptyOutput) {
  OpTester test("GatherND", 29);
  Prepare(test);
  test.AddInput<float>("data", {2}, {1, 2});
  test.AddInput<int32_t>("indices", {0, 1}, {});
  test.AddOutput<float>("output", {0}, {});
  Run(test);
}

TEST_P(IndexingOpset29Test, Int32IndexBoundsAndCudaContextRemainsUsable) {
  for (int32_t index : {-3, 2}) {
    for (const char* op : {"GatherND", "ScatterND"}) {
      OpTester test(op, 29);
      Prepare(test);
      test.AddInput<float>("data", {2}, {1, 2});
      test.AddInput<int32_t>("indices", {1, 1}, {index});
      if (std::string_view(op) == "ScatterND") {
        test.AddAttribute("reduction", std::string("mean"));
        test.AddInput<float>("updates", {1}, {3});
        test.AddOutput<float>("output", {2}, {1, 2});
      } else {
        test.AddOutput<float>("output", {1}, {0});
      }
      if (!GetParam()) {
        test.Config(OpTester::ExpectResult::kExpectFailure,
                    std::string_view(op) == "GatherND" ? "invalid index found" : "invalid indice found");
      }
      Run(test);
    }
  }
  OpTester valid("ScatterND", 29);
  Prepare(valid);
  valid.AddAttribute("reduction", std::string("mean"));
  valid.AddInput<float>("data", {2}, {1, 2});
  valid.AddInput<int32_t>("indices", {1, 1}, {0});
  valid.AddInput<float>("updates", {1}, {3});
  valid.AddOutput<float>("output", {2}, {2, 2});
  Run(valid);
}

TEST_P(IndexingOpset29Test, ScatterElementsMeanInvalidGpuIndexDoesNotDiscardOtherUpdates) {
  if (!GetParam()) {
    GTEST_SKIP() << "CPU rejects invalid indices instead of skipping updates";
  }
  OpTester test("ScatterElements", 29);
  Prepare(test);
  test.AddAttribute("reduction", std::string("mean"));
  test.AddInput<float>("data", {2}, {1, 2});
  test.AddInput<int32_t>("indices", {4}, {0, 3, 0, -3});
  test.AddInput<float>("updates", {4}, {3, 100, 5, 100});
  test.AddOutput<float>("output", {2}, {3, 2});
  Run(test);
}

TEST_P(IndexingOpset29Test, ScatterNDNonMeanInvalidGpuIndexIsNotClamped) {
  if (!GetParam()) {
    GTEST_SKIP() << "CPU rejects invalid indices instead of skipping updates";
  }
  for (const char* reduction : {"none", "add"}) {
    for (int32_t index : {-3, 2}) {
      OpTester test("ScatterND", 29);
      Prepare(test);
      test.AddAttribute("reduction", std::string(reduction));
      test.AddInput<float>("data", {2}, {1, 2});
      test.AddInput<int32_t>("indices", {1, 1}, {index});
      test.AddInput<float>("updates", {1}, {3});
      test.AddOutput<float>("output", {2}, {1, 2});
      Run(test);
    }
  }
}

#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(CPUAndCUDA, IndexingOpset29Test, ::testing::Values(false, true));
#else
INSTANTIATE_TEST_SUITE_P(CPU, IndexingOpset29Test, ::testing::Values(false));
#endif

}  // namespace
}  // namespace onnxruntime::test
