// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <future>
#include <random>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_provider_options.h"
#endif

namespace onnxruntime::test {
namespace {

ONNX_NAMESPACE::TypeProto TensorType(std::initializer_list<int64_t> dimensions,
                                     int type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  ONNX_NAMESPACE::TypeProto result;
  auto* tensor = result.mutable_tensor_type();
  tensor->set_elem_type(type);
  for (const auto value : dimensions) {
    auto* dim = tensor->mutable_shape()->add_dim();
    if (value >= 0) dim->set_dim_value(value);
  }
  return result;
}

void CheckInference(const ONNX_NAMESPACE::TypeProto& a_type,
                    const ONNX_NAMESPACE::TypeProto& b_type,
                    const std::string& error = "") {
  Model model("unit lower solve shape inference", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto& a = graph.GetOrCreateNodeArg("A", &a_type);
  auto& b = graph.GetOrCreateNodeArg("B", &b_type);
  auto& x = graph.GetOrCreateNodeArg("X", nullptr);
  graph.AddNode("solve", "UnitLowerTriangularSolve", "", {&a, &b}, {&x}, nullptr, kMSDomain);
  graph.SetInputs({&a, &b});
  graph.SetOutputs({&x});
  const auto status = graph.Resolve();
  if (!error.empty()) {
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(error), std::string::npos) << status.ErrorMessage();
    return;
  }
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_NE(x.TypeAsProto(), nullptr);
  EXPECT_EQ(x.TypeAsProto()->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ASSERT_NE(x.Shape(), nullptr);
  ASSERT_NE(b.Shape(), nullptr);
  EXPECT_EQ(x.Shape()->SerializeAsString(), b.Shape()->SerializeAsString());
}

}  // namespace

TEST(UnitLowerTriangularSolveSchemaTest, ConcreteAndEmptyShapes) {
  CheckInference(TensorType({1, 32, 1, 64, 64}), TensorType({1, 32, 1, 64, 128}));
  CheckInference(TensorType({2, 0, 0}), TensorType({2, 0, 7}));
  CheckInference(TensorType({0, 3, 3}), TensorType({0, 3, 2}));
  CheckInference(TensorType({3, 3}), TensorType({3, 0}));
}

TEST(UnitLowerTriangularSolveSchemaTest, SymbolicAndUnknownDimensions) {
  auto a = TensorType({-1, 3, 3});
  auto b = TensorType({-1, 3, -1});
  a.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_param("batch");
  b.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_param("batch");
  b.mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_param("rhs");
  CheckInference(a, b);
  a.mutable_tensor_type()->clear_shape();
  CheckInference(a, b);
  CheckInference(TensorType({-1, -1}), TensorType({-1, -1}));
}

TEST(UnitLowerTriangularSolveSchemaTest, RejectsInvalidShapesAndTypes) {
  CheckInference(TensorType({3}), TensorType({3}), "rank >= 2");
  CheckInference(TensorType({2, 3, 3}), TensorType({3, 1}), "equal rank");
  CheckInference(TensorType({3, 4}), TensorType({4, 1}), "square");
  CheckInference(TensorType({3, 3}), TensorType({4, 1}), "row counts");
  CheckInference(TensorType({2, 3, 3}), TensorType({1, 3, 1}), "no broadcasting");
  CheckInference(TensorType({3, 3}, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE),
                 TensorType({3, 1}, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE), "invalid");
}

#if defined(USE_CUDA) && CUDART_VERSION >= 13000
namespace {

void RunSolve(std::initializer_list<int64_t> a_shape, std::initializer_list<int64_t> b_shape,
              std::initializer_list<float> a, std::initializer_list<float> b,
              std::initializer_list<float> expected, cudaStream_t stream = nullptr) {
  OrtCUDAProviderOptionsV2 options{};
  options.has_user_compute_stream = stream != nullptr;
  options.user_compute_stream = stream;
  auto cuda = stream ? CudaExecutionProviderWithOptions(&options) : DefaultCudaExecutionProvider();
  if (!cuda) GTEST_SKIP() << "CUDA EP is unavailable";
  OpTester test("UnitLowerTriangularSolve", 1, kMSDomain);
  test.AddInput<float>("A", a_shape, a);
  test.AddInput<float>("B", b_shape, b);
  test.AddOutput<float>("X", b_shape, expected);
  test.SetOutputAbsErr("X", 0);
  test.SetOutputRelErr("X", 0);
  test.SetNumRunCalls(10);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

TEST(UnitLowerTriangularSolveTest, IgnoresNonfiniteDiagonalAndUpperTriangle) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  RunSolve({3, 3}, {3, 2},
           {nan, inf, nan, 2, -inf, nan, 3, 4, nan},
           {1, 2, 5, 8, 20, 28}, {1, 2, 3, 4, 5, 6});
}

TEST(UnitLowerTriangularSolveTest, MultipleLeadingBatchDimensions) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  RunSolve({2, 1, 2, 2, 2}, {2, 1, 2, 2, 1},
           {nan, nan, 2, nan, nan, nan, -3, nan,
            nan, nan, 0, nan, nan, nan, 0.5f, nan},
           {1, 5, 2, -2, 3, 4, 2, 5},
           {1, 3, 2, 4, 3, 4, 2, 4});
}

TEST(UnitLowerTriangularSolveTest, EmptyMatricesBatchAndRightHandSide) {
  RunSolve({2, 0, 0}, {2, 0, 3}, {}, {}, {});
  RunSolve({0, 3, 3}, {0, 3, 2}, {}, {}, {});
  RunSolve({3, 3}, {3, 0}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {}, {});
}

TEST(UnitLowerTriangularSolveTest, ScalarUnitMatrix) {
  RunSolve({1, 1}, {1, 3}, {0}, {-7, 0, 11}, {-7, 0, 11});
}

TEST(UnitLowerTriangularSolveTest, SeededBatchedRectangularRightHandSides) {
  std::mt19937 generator(128041);
  for (const int64_t rows : {64, 65}) {
    auto cuda = DefaultCudaExecutionProvider();
    if (!cuda) GTEST_SKIP() << "CUDA EP is unavailable";
    constexpr int64_t batch = 2;
    constexpr int64_t columns = 17;
    std::vector<float> a(batch * rows * rows, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> expected(batch * rows * columns);
    std::vector<float> b(expected.size());
    for (auto& value : expected) value = static_cast<float>(generator() % 256) / 16;
    for (int64_t matrix = 0; matrix < batch; ++matrix) {
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < row; ++col) {
          a[matrix * rows * rows + row * rows + col] = col + 1 == row ? -0.5f : 0;
        }
        for (int64_t col = 0; col < columns; ++col) {
          const int64_t i = matrix * rows * columns + row * columns + col;
          b[i] = expected[i] - (row == 0 ? 0 : 0.5f * expected[i - columns]);
        }
      }
    }
    OpTester test("UnitLowerTriangularSolve", 1, kMSDomain);
    test.AddInput<float>("A", {batch, rows, rows}, a);
    test.AddInput<float>("B", {batch, rows, columns}, b);
    test.AddOutput<float>("X", {batch, rows, columns}, expected);
    test.SetOutputAbsErr("X", 0);
    test.SetOutputRelErr("X", 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(cuda));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(UnitLowerTriangularSolveTest, RejectsInvalidRuntimeShapes) {
  struct Case {
    std::vector<int64_t> a;
    std::vector<int64_t> b;
    const char* error;
  };
  const int64_t too_large = int64_t{std::numeric_limits<int>::max()} + 1;
  const Case cases[] = {
      {{3}, {3}, "equal rank >= 2"},
      {{2, 3, 3}, {3, 1}, "equal rank >= 2"},
      {{3, 4}, {4, 1}, "A must be square"},
      {{3, 3}, {4, 1}, "B row count"},
      {{2, 3, 3}, {1, 3, 1}, "no broadcasting"},
      {{0, too_large, too_large}, {0, too_large, 1}, "must fit int32"},
      {{too_large, 0, 0}, {too_large, 0, 1}, "must fit int32"},
  };
  for (const auto& item : cases) {
    auto cuda = DefaultCudaExecutionProvider();
    if (!cuda) GTEST_SKIP() << "CUDA EP is unavailable";
    OpTester test("UnitLowerTriangularSolve", 1, kMSDomain);
    test.AddShapeToTensorData(false);
    const auto a_count = static_cast<size_t>(TensorShape(item.a).Size());
    const auto b_count = static_cast<size_t>(TensorShape(item.b).Size());
    test.AddInput<float>("A", item.a, std::vector<float>(a_count, 1));
    test.AddInput<float>("B", item.b, std::vector<float>(b_count, 1));
    test.AddOutput<float>("X", item.b, std::vector<float>(b_count, 0));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(cuda));
    test.Run(OpTester::ExpectResult::kExpectFailure, item.error, {}, nullptr, &providers);
  }
}

TEST(UnitLowerTriangularSolveTest, ConcurrentNondefaultAndPerThreadStreams) {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  struct OwnedStream {
    OwnedStream() {
      ORT_ENFORCE(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) == cudaSuccess);
    }
    ~OwnedStream() { EXPECT_EQ(cudaStreamDestroy(value), cudaSuccess); }
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OwnedStream);
    cudaStream_t value = nullptr;
  };
  OwnedStream first;
  OwnedStream second;
  for (const int mode : {0, 1, 2}) {
    std::promise<void> start;
    auto ready = start.get_future().share();
    auto run = [&](bool other) {
      ready.wait();
      const auto stream = mode == 2 ? cudaStreamPerThread : (other && mode == 0 ? second.value : first.value);
      RunSolve({2, 2}, {2, 2}, {0, 0, 2, 0},
               {1, 2, 5, 8}, {1, 2, 3, 4}, stream);
    };
    auto a = std::async(std::launch::async, run, false);
    auto b = std::async(std::launch::async, run, true);
    start.set_value();
    a.get();
    b.get();
  }
}
#endif

}  // namespace onnxruntime::test
