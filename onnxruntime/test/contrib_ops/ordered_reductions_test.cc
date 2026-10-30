// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

using ONNX_NAMESPACE::TensorProto;

common::Status ResolveOrdered(
    bool matrix, const std::optional<std::vector<int64_t>>& dimensions,
    std::optional<int64_t> width = 128, int64_t normalize = 0, int type = TensorProto::FLOAT,
    const std::function<void(const NodeArg&)>& verify = nullptr) {
  Model model("ordered_reduction", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto input_type;
  auto* tensor = input_type.mutable_tensor_type();
  tensor->set_elem_type(type);
  if (dimensions) {
    auto* shape = tensor->mutable_shape();
    for (int64_t value : *dimensions) {
      auto* dim = shape->add_dim();
      if (value == -1) {
        dim->set_dim_param("batch");
      } else {
        dim->set_dim_value(value);
      }
    }
  }
  auto& input = graph.GetOrCreateNodeArg("X", &input_type);
  auto& output = graph.GetOrCreateNodeArg("Y", nullptr);
  auto& node = graph.AddNode("ordered", matrix ? "OrderedMatrixReduce128" : "OrderedRowReduce", "",
                             {&input}, {&output}, nullptr, kMSDomain);
  if (!matrix) {
    if (width) node.AddAttribute("width", *width);
    node.AddAttribute("normalize", normalize);
  }
  graph.SetOutputs({&output});
  auto status = graph.Resolve();
  if (status.IsOK() && verify) verify(output);
  return status;
}

TEST(OrderedReductionsShapeTest, RowShapes) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{128}, {2, 3, 128}, {0, 128}, {-1, 0, 128}}) {
    ASSERT_STATUS_OK(ResolveOrdered(false, shape, 128, 0, TensorProto::FLOAT, [&](const NodeArg& output) {
      ASSERT_NE(output.Shape(), nullptr);
      ASSERT_EQ(output.Shape()->dim_size(), 2);
      EXPECT_EQ(output.Shape()->dim(0).dim_value(), shape.size() == 1 ? 1 : (shape[0] == 2 ? 6 : 0));
      EXPECT_EQ(output.Shape()->dim(1).dim_value(), 1);
      EXPECT_EQ(output.TypeAsProto()->tensor_type().elem_type(), TensorProto::FLOAT);
    }));
  }
  ASSERT_STATUS_OK(ResolveOrdered(false, std::vector<int64_t>{-1, 128}, 128, 1, TensorProto::FLOAT,
                                  [](const NodeArg& output) {
                                    EXPECT_EQ(output.Shape()->dim(0).dim_param(), "batch");
                                  }));
  // Graph canonicalizes negative dimension values to unknown before inference.
  for (const auto& shape : {std::optional<std::vector<int64_t>>{}, {{2, -1, 128}}, {{-2, 128}}}) {
    ASSERT_STATUS_OK(ResolveOrdered(false, shape, 128, 0, TensorProto::FLOAT, [](const NodeArg& output) {
      ASSERT_EQ(output.Shape()->dim_size(), 2);
      EXPECT_FALSE(output.Shape()->dim(0).has_dim_value());
      EXPECT_EQ(output.Shape()->dim(1).dim_value(), 1);
    }));
  }
}

TEST(OrderedReductionsShapeTest, MatrixShapes) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{128, 128}, {2, 3, 128, 128}, {0, 128, 128}}) {
    ASSERT_STATUS_OK(ResolveOrdered(true, shape, 128, 0, TensorProto::FLOAT, [&](const NodeArg& output) {
      ASSERT_EQ(output.Shape()->dim_size(), shape.size() - 1);
      for (int i = 0; i < output.Shape()->dim_size(); ++i) {
        EXPECT_EQ(output.Shape()->dim(i).dim_value(), shape[i]);
      }
    }));
  }
  ASSERT_STATUS_OK(ResolveOrdered(true, std::vector<int64_t>{-1, 128, 128}, 128, 0, TensorProto::FLOAT,
                                  [](const NodeArg& output) {
                                    EXPECT_EQ(output.Shape()->dim(0).dim_param(), "batch");
                                    EXPECT_EQ(output.Shape()->dim(1).dim_value(), 128);
                                  }));
  ASSERT_STATUS_OK(ResolveOrdered(true, std::nullopt, 128, 0, TensorProto::FLOAT, [](const NodeArg& output) {
    EXPECT_EQ(output.Shape(), nullptr);
  }));
}

TEST(OrderedReductionsShapeTest, InvalidAttributesShapesAndTypes) {
  for (int64_t width : {-32, 0, 16, 33, 8192}) {
    EXPECT_FALSE(ResolveOrdered(false, std::nullopt, width).IsOK());
  }
  EXPECT_FALSE(ResolveOrdered(false, std::nullopt, std::nullopt).IsOK());
  for (int64_t normalize : {-1, 2}) {
    EXPECT_FALSE(ResolveOrdered(false, std::nullopt, 128, normalize).IsOK());
  }
  const std::vector<std::vector<int64_t>> bad_rows{
      {}, {64}, {0, 64}, {std::numeric_limits<int64_t>::max(), 2, 128}};
  for (const auto& shape : bad_rows) {
    SCOPED_TRACE(::testing::PrintToString(shape));
    EXPECT_FALSE(ResolveOrdered(false, shape).IsOK());
  }
  for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {128}, {128, 64}, {64, 128}, {0, 128, 64}}) {
    EXPECT_FALSE(ResolveOrdered(true, shape).IsOK());
  }
  for (int type : {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::DOUBLE, TensorProto::INT32}) {
    EXPECT_FALSE(ResolveOrdered(false, std::vector<int64_t>{128}, 128, 0, type).IsOK());
    EXPECT_FALSE(ResolveOrdered(true, std::vector<int64_t>{128, 128}, 128, 0, type).IsOK());
  }
}

#ifdef USE_CUDA
// Round every reference operation to FLOAT independently of host contraction.
float Add(float a, float b) {
  volatile float result = a + b;
  return result;
}

std::vector<float> RowReference(const std::vector<float>& input, size_t rows, int width, bool normalize) {
  const int vector = width > 128 ? 4 : 1;
  int threads = std::min(width / vector, 512);
  for (int cutoff : {2, 4, 8, 16}) {
    if (rows >= static_cast<size_t>(cutoff)) threads = std::min(width / vector, 512 / cutoff);
  }
  std::vector<float> output(rows);
  for (size_t row = 0; row < rows; ++row) {
    std::array<float, 512> partial{};
    for (int thread = 0; thread < threads; ++thread) {
      std::array<float, 4> lanes{};
      for (int lane = 0; lane < vector; ++lane) {
        lanes[lane] = input[row * width + thread * vector + lane];
        for (int offset = threads * vector; offset < width; offset += threads * vector) {
          lanes[lane] = Add(lanes[lane], input[row * width + offset + thread * vector + lane]);
        }
      }
      partial[thread] = lanes[0];
      for (int lane = 1; lane < vector; ++lane) partial[thread] = Add(partial[thread], lanes[lane]);
    }
    for (int count = threads; count > 32; count /= 2) {
      for (int i = 0; i < count / 2; ++i) partial[i] = Add(partial[i], partial[i + count / 2]);
    }
    for (int offset = 1; offset < 32; offset *= 2) {
      const auto previous = partial;
      for (int i = 0; i < 32 - offset; ++i) partial[i] = Add(previous[i], previous[i + offset]);
    }
    volatile float result = normalize ? partial[0] * (1.0f / width) : partial[0];
    output[row] = result;
  }
  return output;
}

std::vector<float> MatrixReference(const std::vector<float>& input, size_t matrices) {
  std::vector<float> output(matrices * 128);
  for (size_t matrix = 0; matrix < matrices; ++matrix) {
    for (int column = 0; column < 128; ++column) {
      std::array<float, 4> partial{};
      for (int lane = 0; lane < 4; ++lane) {
        std::array<float, 4> sums{};
        for (int accumulator = 0; accumulator < 4; ++accumulator) {
          for (int row = accumulator * 4 + lane; row < 128; row += 16) {
            sums[accumulator] = Add(sums[accumulator], input[matrix * 16384 + row * 128 + column]);
          }
        }
        partial[lane] = Add(Add(Add(sums[0], sums[1]), sums[2]), sums[3]);
      }
      output[matrix * 128 + column] = Add(Add(partial[0], partial[2]), Add(partial[1], partial[3]));
    }
  }
  return output;
}

std::vector<float> MakeInput(size_t count, int kind) {
  std::vector<float> result(count);
  uint32_t state = 7832;
  for (size_t i = 0; i < count; ++i) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    if (kind == 2) {
      const uint32_t bits = state & 0x807fffffu;  // Signed subnormals and zeros.
      std::memcpy(&result[i], &bits, sizeof(float));
    } else {
      const float value = std::ldexp(static_cast<float>(static_cast<int>(state % 65537) - 32768),
                                     static_cast<int>((state >> 16) % 25) - 24);
      result[i] = kind == 1 ? value * value : value;
    }
  }
  return result;
}

void RunExact(OpTester& test, const std::vector<float>& expected) {
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& values, const std::string& provider_type) {
    EXPECT_EQ(provider_type, kCudaExecutionProvider);
    ASSERT_EQ(values.size(), 1);
    const auto& tensor = values[0].Get<Tensor>();
    ASSERT_EQ(tensor.Shape().Size(), expected.size());
    if (!expected.empty()) {
      EXPECT_EQ(std::memcmp(tensor.Data<float>(), expected.data(), expected.size() * sizeof(float)), 0);
    }
  });
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(OrderedReductionsCudaTest, RowsExactTreesThresholdsAndEmpty) {
  for (int width : {32, 64, 128, 256, 512, 1024, 2048, 4096}) {
    for (int64_t rows : {0, 1, 2, 3, 4, 7, 8, 15, 16, 30}) {
      for (int kind = 0; kind < 3; ++kind) {
        const auto input = MakeInput(static_cast<size_t>(rows) * width, kind);
        for (int64_t normalize : {0, 1}) {
          SCOPED_TRACE(MakeString(width, ":", rows, ":", kind, ":", normalize));
          const auto expected = RowReference(input, static_cast<size_t>(rows), width, normalize != 0);
          OpTester test("OrderedRowReduce", 1, kMSDomain);
          test.AddAttribute("width", int64_t{width});
          test.AddAttribute("normalize", normalize);
          test.AddInput<float>("X", {rows, width}, input);
          test.AddOutput<float>("Y", {rows, 1}, expected);
          RunExact(test, expected);
        }
      }
    }
  }
}

TEST(OrderedReductionsCudaTest, RowGridStrideAndRank) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{32}, {2, 3, 32}, {65536, 32}}) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    const auto input = MakeInput(static_cast<size_t>(size), 0);
    const auto expected = RowReference(input, static_cast<size_t>(size / 32), 32, false);
    OpTester test("OrderedRowReduce", 1, kMSDomain);
    test.AddAttribute("width", int64_t{32});  // Also checks normalize's default.
    test.AddInput<float>("X", shape, input);
    test.AddOutput<float>("Y", {size / 32, 1}, expected);
    RunExact(test, expected);
  }
}

TEST(OrderedReductionsCudaTest, MatrixExactTreeAndEmpty) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{128, 128}, {2, 3, 128, 128}, {0, 128, 128}}) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    for (int kind = 0; kind < 3; ++kind) {
      const auto input = MakeInput(static_cast<size_t>(size), kind);
      const auto expected = MatrixReference(input, static_cast<size_t>(size / 16384));
      OpTester test("OrderedMatrixReduce128", 1, kMSDomain);
      test.AddInput<float>("X", shape, input);
      test.AddOutput<float>("Y", std::vector<int64_t>(shape.begin(), shape.end() - 1), expected);
      RunExact(test, expected);
    }
  }
}

TEST(OrderedReductionsCudaTest, SignedZeroAndInfinity) {
  for (bool matrix : {false, true}) {
    for (float value : {-0.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
      const std::vector<float> input(matrix ? 16384 : 128, value);
      const auto expected = matrix ? MatrixReference(input, 1) : RowReference(input, 1, 128, false);
      OpTester test(matrix ? "OrderedMatrixReduce128" : "OrderedRowReduce", 1, kMSDomain);
      if (!matrix) test.AddAttribute("width", int64_t{128});
      test.AddInput<float>("X", matrix ? std::vector<int64_t>{128, 128} : std::vector<int64_t>{128}, input);
      test.AddOutput<float>("Y", matrix ? std::vector<int64_t>{128} : std::vector<int64_t>{1, 1}, expected);
      RunExact(test, expected);
    }
  }
}

TEST(OrderedReductionsCudaTest, RejectDynamicInvalidShapes) {
  for (bool matrix : {false, true}) {
    for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {64}, {0, 64}, {128, 64}}) {
      int64_t size = 1;
      for (auto dim : shape) size *= dim;
      OpTester test(matrix ? "OrderedMatrixReduce128" : "OrderedRowReduce", 1, kMSDomain);
      test.AddShapeToTensorData(false);
      if (!matrix) test.AddAttribute("width", int64_t{128});
      test.AddInput<float>("X", shape, MakeInput(static_cast<size_t>(size), 0));
      test.AddOutput<float>("Y", {0, 1}, {});
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      const std::string error = matrix ? "must end in [128,128]"
                                       : (shape.empty() ? "rank must be positive" : "Last dimension must match width");
      test.Run(OpTester::ExpectResult::kExpectFailure, error, {}, nullptr, &providers);
    }
  }
}
#endif  // USE_CUDA

}  // namespace
}  // namespace onnxruntime::test
