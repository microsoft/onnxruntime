// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/common/optional.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "test/common/random_generator.h"
#include "test/util/include/asserts.h"
// To avoid conflict of LogRuntimeError, we direct include the cc file directly.
#include "test/util/test_random_seed.cc"

using onnxruntime::test::RandomValueGenerator;

namespace onnxruntime {
namespace cuda {
namespace test {

namespace {
struct DeviceMemoryDeleter {
  template <typename T>
  void operator()(T* p) {
    cudaFree(p);
  }
};

template <typename T>
std::unique_ptr<T, DeviceMemoryDeleter> AllocateDeviceMemory(size_t n = 1) {
  T* p{};
  cudaMalloc(&p, n * sizeof(T));
  return std::unique_ptr<T, DeviceMemoryDeleter>(p);
}

template <typename T>
void CheckDeviceValues(size_t n, const T* d_actual, const T* expected, float relative_error_tolerance) {
  std::vector<T> actual(n);
  cudaMemcpy(actual.data(), d_actual, n * sizeof(T), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_LE(std::abs(actual[i] - expected[i]) / expected[i], relative_error_tolerance)
        << "i: " << i << ", actual[i]: " << actual[i] << ", expected[i]: " << expected[i];
  }
}

void TestReduceRowToScalarApis(int size, float relative_error_tolerance = 1e-4f) {
  SCOPED_TRACE(MakeString("size: ", size));

  float expected_output_sum = 0;
  float expected_output_square_sum = 0;
  float expected_output_mean = 0;
  const std::vector<int64_t> shape = {size};
  RandomValueGenerator random_value_generator{};
  const auto input = random_value_generator.Uniform<float>(shape, 0.1f, 1.0f);
  for (const auto input_value : input) {
    expected_output_sum += input_value;
    expected_output_square_sum += input_value * input_value;
    expected_output_mean += input_value / float(size);
  }
  const auto buffer_size_in_bytes =
      compute_reduction_buffer_size<float>(size);

  auto device_input = AllocateDeviceMemory<float>(size);
  auto device_output_sum = AllocateDeviceMemory<float>();
  auto device_output_square_sum = AllocateDeviceMemory<float>();
  auto device_output_mean = AllocateDeviceMemory<float>();
  auto buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  cudaMemcpy(device_input.get(), input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum(
      0,
      device_input.get(),
      device_output_sum.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));
  ASSERT_STATUS_OK(reduce_square_sum(
      0,
      device_input.get(),
      device_output_square_sum.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));
  ASSERT_STATUS_OK(reduce_mean(
      0,
      device_input.get(),
      device_output_mean.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

  CheckDeviceValues(1, device_output_sum.get(), &expected_output_sum, relative_error_tolerance);
  CheckDeviceValues(1, device_output_square_sum.get(), &expected_output_square_sum, relative_error_tolerance);
  CheckDeviceValues(1, device_output_mean.get(), &expected_output_mean, relative_error_tolerance);
}

void TestReduceRowsToRow(int m, int n, bool reset_initial_output, float relative_error_tolerance = 1e-4f) {
  SCOPED_TRACE(MakeString("m: ", m, ", n:", n, ", reset_initial_output: ", reset_initial_output));

  const TensorShape shape{m, n};
  RandomValueGenerator random{};
  const auto values = random.Uniform<float>(shape.GetDims(), 1.0f, 10.0f);
  const auto initial_value = reset_initial_output ? 0.0f : 5.0f;
  const std::vector<float> expected_row =
      [m, n, &values, initial_value]() {
        std::vector<float> row(n, initial_value);
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            row[j] += values[i * n + j];
          }
        }
        return row;
      }();

  auto d_in = AllocateDeviceMemory<float>(m * n);
  auto d_out = AllocateDeviceMemory<float>(n);

  cudaMemcpy(d_in.get(), values.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

  if (!reset_initial_output) {
    // manually initialize output data
    Fill(0, d_out.get(), initial_value, n);
  }

  ASSERT_STATUS_OK(reduce_matrix_rows(
      0, d_in.get(), d_out.get(),
      m, n,
      reset_initial_output));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

  CheckDeviceValues(n, d_out.get(), expected_row.data(), relative_error_tolerance);
}

template <typename T>
std::vector<T> ExpectedReduceMatrixColumnsOutput(
    int m, int n, const std::vector<T>& values) {
  std::vector<T> column(m);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      column[i] += values[i * n + j];
    }
  }
  return column;
}

void TestReduceColumnsToColumn(int m, int n, float relative_error_tolerance = 1e-4f) {
  SCOPED_TRACE(MakeString("m: ", m, ", n:", n));

  const TensorShape shape{m, n};
  RandomValueGenerator random{};
  const auto values = random.Uniform<float>(shape.GetDims(), 1.0f, 10.0f);
  const auto expected_column = ExpectedReduceMatrixColumnsOutput(m, n, values);

  auto d_in = AllocateDeviceMemory<float>(m * n);
  auto d_out = AllocateDeviceMemory<float>(m);

  cudaMemcpy(d_in.get(), values.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

  size_t buffer_size_in_bytes =
      compute_reduce_matrix_columns_buffer_size<float>(m, n);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  ASSERT_STATUS_OK(reduce_matrix_columns(
      0,
      d_in.get(), d_out.get(),
      m, n,
      d_buffer.get(), buffer_size_in_bytes));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

  CheckDeviceValues(m, d_out.get(), expected_column.data(), relative_error_tolerance);
}

void TestReduceColumnsToColumnRepeated(int m, int n, int iterations, float relative_error_tolerance = 1e-4f) {
  SCOPED_TRACE(MakeString("m: ", m, ", n:", n, ", iterations: ", iterations));

  const TensorShape shape{m, n};
  RandomValueGenerator random{};
  const auto values = random.Uniform<float>(shape.GetDims(), 1.0f, 10.0f);
  const auto expected_column = ExpectedReduceMatrixColumnsOutput(m, n, values);

  auto d_in = AllocateDeviceMemory<float>(m * n);
  auto d_out = AllocateDeviceMemory<float>(m);

  cudaMemcpy(d_in.get(), values.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

  size_t buffer_size_in_bytes =
      compute_reduce_matrix_columns_buffer_size<float>(m, n);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  for (int i = 0; i < iterations; ++i) {
    ASSERT_STATUS_OK(reduce_matrix_columns(
        0,
        d_in.get(), d_out.get(),
        m, n,
        d_buffer.get(), buffer_size_in_bytes));

    ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
    CheckDeviceValues(m, d_out.get(), expected_column.data(), relative_error_tolerance);
  }
}
}  // namespace

TEST(ReductionFunctionsTest, ReduceRowToScalar) {
  TestReduceRowToScalarApis(3);
  TestReduceRowToScalarApis(19);
  TestReduceRowToScalarApis(123);
  TestReduceRowToScalarApis(1128);
  TestReduceRowToScalarApis(5566);
  TestReduceRowToScalarApis(941736, 2e-4f);
}

TEST(ReductionFunctionsTest, ReduceRowsToRow) {
  for (int m : {3, 193, 2945}) {
    for (int n : {3, 193, 2945}) {
      TestReduceRowsToRow(m, n, true);
      TestReduceRowsToRow(m, n, false);
    }
  }
}

TEST(ReductionFunctionsTest, ReduceColumnsToColumn) {
  for (int m : {3, 193, 2945}) {
    for (int n : {3, 193, 2945}) {
      TestReduceColumnsToColumn(m, n);
    }
  }
}

TEST(ReductionFunctionsTest, ReduceColumnsToColumnRepeated) {
  TestReduceColumnsToColumnRepeated(17, 8192, 100, 2e-4f);
}

TEST(ReductionFunctionsTest, ReduceSumNdMiddleAndMultipleAxes) {
  const std::vector<int64_t> dims{2, 3, 4, 2};
  const std::vector<int64_t> axes{1, 3};
  std::vector<float> input(48);
  std::iota(input.begin(), input.end(), 1.0f);
  std::vector<float> expected(8, 0.0f);
  for (int64_t d0 = 0; d0 < dims[0]; ++d0) {
    for (int64_t d1 = 0; d1 < dims[1]; ++d1) {
      for (int64_t d2 = 0; d2 < dims[2]; ++d2) {
        for (int64_t d3 = 0; d3 < dims[3]; ++d3) {
          expected[d0 * dims[2] + d2] += input[((d0 * dims[1] + d1) * dims[2] + d2) * dims[3] + d3];
        }
      }
    }
  }

  auto d_input = AllocateDeviceMemory<float>(input.size());
  auto d_output = AllocateDeviceMemory<float>(expected.size());
  cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum_nd(0, d_input.get(), d_output.get(), dims, axes));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  CheckDeviceValues(expected.size(), d_output.get(), expected.data(), 1e-6f);
}

TEST(ReductionFunctionsTest, ReduceSumNdIntegerSaturation) {
  const std::vector<int64_t> dims{2, 3, 2};
  const std::vector<int64_t> axes{1};
  const int32_t big = 1'100'000'000;
  const std::vector<int32_t> input(12, big);
  const std::vector<int32_t> expected(4, std::numeric_limits<int32_t>::max());

  auto d_input = AllocateDeviceMemory<int32_t>(input.size());
  auto d_output = AllocateDeviceMemory<int32_t>(expected.size());
  cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum_nd(0, d_input.get(), d_output.get(), dims, axes));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  std::vector<int32_t> actual(expected.size());
  cudaMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
  EXPECT_EQ(actual, expected);
}

TEST(ReductionFunctionsTest, ReduceSumNdLargeReductionSmallOutput) {
  const std::vector<int64_t> dims{2, 131072, 3};
  const std::vector<int64_t> axes{1};
  std::vector<float> input(TensorShape(dims).Size(), 1.0f);
  const std::vector<float> expected(6, 131072.0f);

  auto d_input = AllocateDeviceMemory<float>(input.size());
  auto d_output = AllocateDeviceMemory<float>(expected.size());
  cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum_nd(0, d_input.get(), d_output.get(), dims, axes));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  CheckDeviceValues(expected.size(), d_output.get(), expected.data(), 0.0f);
}

TEST(ReductionFunctionsTest, ReduceSumNdInt64Cancellation) {
  const std::vector<int64_t> dims{1, 3, 1};
  const std::vector<int64_t> axes{1};
  const int64_t large = int64_t{1} << 53;
  const std::vector<int64_t> input{large, 1, -large};
  const std::vector<int64_t> expected{1};

  auto d_input = AllocateDeviceMemory<int64_t>(input.size());
  auto d_output = AllocateDeviceMemory<int64_t>(expected.size());
  cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum_nd(0, d_input.get(), d_output.get(), dims, axes));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  std::vector<int64_t> actual(expected.size());
  cudaMemcpy(actual.data(), d_output.get(), actual.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
  EXPECT_EQ(actual, expected);
}

TEST(ReductionFunctionsTest, ReduceSumNdRank9) {
  const std::vector<int64_t> dims{2, 2, 2, 2, 2, 2, 2, 2, 2};
  const std::vector<int64_t> axes{1, 3, 5, 7};
  std::vector<float> input(TensorShape(dims).Size(), 1.0f);
  const std::vector<float> expected(32, 16.0f);

  auto d_input = AllocateDeviceMemory<float>(input.size());
  auto d_output = AllocateDeviceMemory<float>(expected.size());
  cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(reduce_sum_nd(0, d_input.get(), d_output.get(), dims, axes));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  CheckDeviceValues(expected.size(), d_output.get(), expected.data(), 0.0f);
}

TEST(ReductionFunctionsTest, BufferOffsets) {
  const int m = 2048;
  const int n = 1024;
  const TensorShape shape{m, n};

  const size_t max_buffer_offset = 15;

  const size_t buffer_size_in_bytes =
      compute_reduce_matrix_columns_buffer_size<double>(m, n) + max_buffer_offset;

  auto d_input = AllocateDeviceMemory<double>(m * n);
  auto d_output = AllocateDeviceMemory<double>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  RandomValueGenerator random{};
  const float relative_error_tolerance = 1e-4f;

  for (size_t buffer_offset = 1; buffer_offset <= max_buffer_offset; ++buffer_offset) {
    SCOPED_TRACE(MakeString("buffer offset: ", buffer_offset));

    const auto input = random.Uniform<double>(shape.GetDims(), 1.0, 10.0);
    cudaMemcpy(d_input.get(), input.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    ASSERT_STATUS_OK(reduce_matrix_columns(
        0,
        d_input.get(), d_output.get(),
        m, n,
        d_buffer.get() + buffer_offset,
        buffer_size_in_bytes - buffer_offset));

    const auto expected_column = ExpectedReduceMatrixColumnsOutput(m, n, input);
    CheckDeviceValues(m, d_output.get(), expected_column.data(), relative_error_tolerance);
  }
}

TEST(ReductionFunctionsTest, InvalidBufferSize) {
  const int m = 2048;
  const int n = 1024;
  const TensorShape shape{m, n};

  // this should be too small
  const size_t buffer_size_in_bytes =
      compute_reduce_matrix_columns_buffer_size<float>(m, n) / 10;

  auto d_input = AllocateDeviceMemory<float>(m * n);
  auto d_output = AllocateDeviceMemory<float>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  RandomValueGenerator random{};
  const auto input = random.Uniform<float>(shape.GetDims(), 1.0, 10.0);
  cudaMemcpy(d_input.get(), input.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

  const auto status =
      reduce_matrix_columns(0, d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes);
  ASSERT_FALSE(status.IsOK());
}

// ---------------------------------------------------------------------------
// arg_min_max_last_axis
// ---------------------------------------------------------------------------
namespace {
template <typename T>
struct ArgTestTraits {
  static T FromFloat(float value) { return static_cast<T>(value); }
  static double ToDouble(T value) { return static_cast<double>(value); }
};

template <>
struct ArgTestTraits<half> {
  static half FromFloat(float value) { return __float2half(value); }
  static double ToDouble(half value) { return static_cast<double>(__half2float(value)); }
};

// Mirrors the sequential semantics of the reduction: seed with the first
// element, keep the first occurrence of the extreme value and never select a
// NaN (every comparison against NaN is false, so a leading NaN wins the row).
template <typename T, bool IsArgMax>
std::vector<int64_t> ExpectedArgMinMaxIndices(const std::vector<T>& values, int m, int n) {
  std::vector<int64_t> expected(m);
  for (int row = 0; row < m; ++row) {
    double best = ArgTestTraits<T>::ToDouble(values[static_cast<size_t>(row) * n]);
    int64_t best_index = 0;
    for (int i = 1; i < n; ++i) {
      const double value = ArgTestTraits<T>::ToDouble(values[static_cast<size_t>(row) * n + i]);
      if (IsArgMax ? (value > best) : (value < best)) {
        best = value;
        best_index = i;
      }
    }
    expected[row] = best_index;
  }
  return expected;
}

template <typename T, bool IsArgMax>
void CheckArgMinMaxLastAxis(const std::vector<T>& values, int m, int n) {
  SCOPED_TRACE(MakeString("m: ", m, ", n: ", n, ", is_arg_max: ", IsArgMax));

  auto d_input = AllocateDeviceMemory<T>(static_cast<size_t>(m) * n);
  auto d_output = AllocateDeviceMemory<int64_t>(m);
  cudaMemcpy(d_input.get(), values.data(), static_cast<size_t>(m) * n * sizeof(T), cudaMemcpyHostToDevice);
  // Poison the output so a kernel that never writes a row is detected.
  cudaMemset(d_output.get(), 0xff, static_cast<size_t>(m) * sizeof(int64_t));

  const size_t buffer_size_in_bytes = compute_arg_min_max_last_axis_buffer_size<T>(m, n);
  auto d_buffer = AllocateDeviceMemory<char>(std::max<size_t>(buffer_size_in_bytes, 1));
  // Dirty the intermediate buffer: the implementation must not assume it is zeroed.
  if (buffer_size_in_bytes > 0) {
    cudaMemset(d_buffer.get(), 0x5a, buffer_size_in_bytes);
  }

  ASSERT_STATUS_OK((arg_min_max_last_axis<T, IsArgMax>(
      0, d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes)));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

  std::vector<int64_t> actual(m);
  cudaMemcpy(actual.data(), d_output.get(), static_cast<size_t>(m) * sizeof(int64_t), cudaMemcpyDeviceToHost);
  const auto expected = ExpectedArgMinMaxIndices<T, IsArgMax>(values, m, n);
  for (int row = 0; row < m; ++row) {
    ASSERT_EQ(actual[row], expected[row]) << "row: " << row;
  }
}

template <typename T>
std::vector<T> MakeRandomValues(int m, int n, RandomValueGenerator::RandomSeedType seed) {
  RandomValueGenerator random{seed};
  const TensorShape shape{m, n};
  const auto float_values = random.Uniform<float>(shape.GetDims(), -100.0f, 100.0f);
  std::vector<T> values(float_values.size());
  for (size_t i = 0; i < float_values.size(); ++i) {
    values[i] = ArgTestTraits<T>::FromFloat(float_values[i]);
  }
  return values;
}

template <typename T>
void TestArgMinMaxLastAxisRandom(int m, int n, RandomValueGenerator::RandomSeedType seed = 0) {
  const auto values = MakeRandomValues<T>(m, n, seed);
  CheckArgMinMaxLastAxis<T, true>(values, m, n);
  CheckArgMinMaxLastAxis<T, false>(values, m, n);
}

// Fills rows with duplicates, infinities and NaNs. The row count is fixed, the
// width is not, so every dispatch path can be exercised with the same rows.
template <typename T>
std::vector<T> MakeSpecialValueRows(int n) {
  constexpr int kNumSpecialRows = 8;
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float lowest_finite = std::is_same<T, half>::value ? -65504.0f : -3.0e38f;
  const float highest_finite = -lowest_finite;

  std::vector<T> values(static_cast<size_t>(kNumSpecialRows) * n);
  for (int row = 0; row < kNumSpecialRows; ++row) {
    for (int i = 0; i < n; ++i) {
      values[static_cast<size_t>(row) * n + i] = ArgTestTraits<T>::FromFloat(static_cast<float>((i * 37) % 11) - 5.0f);
    }
  }
  auto set = [&](int row, int index, float value) {
    values[static_cast<size_t>(row) * n + index] = ArgTestTraits<T>::FromFloat(value);
  };
  auto fill = [&](int row, float value) {
    for (int i = 0; i < n; ++i) set(row, i, value);
  };

  // row 0: leading NaN.
  set(0, 0, nan);
  // row 1: NaN in the middle plus a unique maximum after it.
  set(1, n / 2, nan);
  set(1, n - 1, 42.0f);
  // row 2: every element is -infinity.
  fill(2, -infinity);
  // row 3: -infinity everywhere except one finite element that must win ArgMax.
  fill(3, -infinity);
  set(3, n - 2, lowest_finite);
  // row 4: all elements equal, ties must resolve to the lowest index.
  fill(4, 1.0f);
  // row 5: duplicated +infinity maximum and duplicated -infinity minimum.
  set(5, 1, infinity);
  set(5, n - 1, infinity);
  set(5, 2, -infinity);
  set(5, n - 2, -infinity);
  // row 6: every element is NaN.
  fill(6, nan);
  // row 7: finite extremes at the ends.
  set(7, n - 1, highest_finite);
  set(7, n / 3, lowest_finite);

  return values;
}

template <typename T>
void TestArgMinMaxLastAxisSpecialValues(int n) {
  constexpr int kNumSpecialRows = 8;
  const auto values = MakeSpecialValueRows<T>(n);
  CheckArgMinMaxLastAxis<T, true>(values, kNumSpecialRows, n);
  CheckArgMinMaxLastAxis<T, false>(values, kNumSpecialRows, n);
}
}  // namespace

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisNarrowRows) {
  // Below the cooperative threshold: one thread per row.
  for (int n : {1, 2, 3, 7, 31, 32, 33, 64, 127}) {
    for (int m : {1, 3, 1024}) {
      TestArgMinMaxLastAxisRandom<float>(m, n, n);
    }
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisSingleBlockRows) {
  // Wide enough for the cooperative kernel, narrow enough for a single block.
  for (int n : {128, 129, 255, 256, 511, 1024, 4095}) {
    for (int m : {1, 3, 1024}) {
      TestArgMinMaxLastAxisRandom<float>(m, n, n);
    }
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisMultiBlockRows) {
  // Few, very wide rows: several blocks cooperate on each row.
  for (int n : {8192, 32000, 65535, 65536, 202048, 262144}) {
    for (int m : {1, 2, 4, 8}) {
      TestArgMinMaxLastAxisRandom<float>(m, n, n);
    }
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisManyRows) {
  // More rows than grid rows, so blocks loop over rows.
  TestArgMinMaxLastAxisRandom<float>(40000, 128);
  TestArgMinMaxLastAxisRandom<float>(40000, 1024);
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisTypes) {
  for (int n : {64, 1024, 65536}) {
    TestArgMinMaxLastAxisRandom<half>(3, n, n);
    TestArgMinMaxLastAxisRandom<float>(3, n, n);
    TestArgMinMaxLastAxisRandom<double>(3, n, n);
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisSpecialValues) {
  // 64 -> one thread per row, 1024 -> single block per row, 200000 -> many blocks per row.
  for (int n : {64, 1024, 200000}) {
    TestArgMinMaxLastAxisSpecialValues<half>(n);
    TestArgMinMaxLastAxisSpecialValues<float>(n);
    TestArgMinMaxLastAxisSpecialValues<double>(n);
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisScanStepBoundary) {
  // A single row is scanned by at most 256 blocks x 256 threads, so the widest
  // per iteration step is 4 * 65536 = 262144 elements. Straddle it to cover the
  // loop termination arithmetic.
  for (int n : {262143, 262144, 262145, 524287, 524288, 524289}) {
    TestArgMinMaxLastAxisRandom<float>(1, n, static_cast<RandomValueGenerator::RandomSeedType>(n));
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisBufferSizeExtremes) {
  // Sizing must stay well defined and small for the widest admitted rows.
  const int int_max = std::numeric_limits<int>::max();
  for (int n : {1 << 20, 1 << 30, int_max - 1, int_max}) {
    const size_t float_bytes = compute_arg_min_max_last_axis_buffer_size<float>(1, n);
    const size_t double_bytes = compute_arg_min_max_last_axis_buffer_size<double>(1, n);
    EXPECT_GT(float_bytes, 0u) << "n: " << n;
    EXPECT_LT(float_bytes, size_t{1} << 20) << "n: " << n;
    EXPECT_GT(double_bytes, float_bytes) << "n: " << n;
    EXPECT_LT(double_bytes, size_t{1} << 20) << "n: " << n;
  }
  // Many narrow rows are handled by the one thread per row kernel, which needs no buffer.
  EXPECT_EQ(compute_arg_min_max_last_axis_buffer_size<float>(1 << 20, 64), 0u);
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisHugeWidth) {
  // A row whose width approaches INT_MAX is what breaks 32 bit position arithmetic in
  // the cooperative scan, so exercise the widest row the dispatch admits. fp16 keeps
  // the allocation at ~4 GiB; the test is skipped when the device cannot hold it.
  const int num_cols = std::numeric_limits<int>::max();
  const size_t input_bytes = static_cast<size_t>(num_cols) * sizeof(half);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  ASSERT_TRUE(CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes)).IsOK());
  if (free_bytes < input_bytes + (size_t{256} << 20)) {
    GTEST_SKIP() << "needs " << (input_bytes >> 20) << " MiB of free device memory, have "
                 << (free_bytes >> 20) << " MiB";
  }

  auto d_input = AllocateDeviceMemory<half>(static_cast<size_t>(num_cols));
  ASSERT_NE(d_input.get(), nullptr) << "allocation of " << (input_bytes >> 20) << " MiB failed";
  auto d_output = AllocateDeviceMemory<int64_t>(1);

  // Zero fill, then place duplicated extremes: the first occurrence must win, and the
  // extremes sit in different scan iterations, including the very last one.
  ASSERT_TRUE(CUDA_CALL(cudaMemset(d_input.get(), 0, input_bytes)).IsOK());
  const int64_t expected_max_index = num_cols - 3;
  const int64_t expected_min_index = 1234567890;
  const half positive = __float2half(1.0f);
  const half negative = __float2half(-1.0f);
  for (int64_t index : {expected_max_index, static_cast<int64_t>(num_cols - 1)}) {
    ASSERT_TRUE(CUDA_CALL(cudaMemcpy(d_input.get() + index, &positive, sizeof(half), cudaMemcpyHostToDevice)).IsOK());
  }
  for (int64_t index : {expected_min_index, static_cast<int64_t>(num_cols - 2)}) {
    ASSERT_TRUE(CUDA_CALL(cudaMemcpy(d_input.get() + index, &negative, sizeof(half), cudaMemcpyHostToDevice)).IsOK());
  }

  const size_t buffer_size_in_bytes = compute_arg_min_max_last_axis_buffer_size<half>(1, num_cols);
  ASSERT_GT(buffer_size_in_bytes, 0u);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  int64_t actual = -1;
  ASSERT_STATUS_OK((arg_min_max_last_axis<half, true>(0, d_input.get(), d_output.get(), 1, num_cols,
                                                      d_buffer.get(), buffer_size_in_bytes)));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  cudaMemcpy(&actual, d_output.get(), sizeof(int64_t), cudaMemcpyDeviceToHost);
  ASSERT_EQ(actual, expected_max_index);

  ASSERT_STATUS_OK((arg_min_max_last_axis<half, false>(0, d_input.get(), d_output.get(), 1, num_cols,
                                                       d_buffer.get(), buffer_size_in_bytes)));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  cudaMemcpy(&actual, d_output.get(), sizeof(int64_t), cudaMemcpyDeviceToHost);
  ASSERT_EQ(actual, expected_min_index);
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisEmptyRows) {
  auto d_input = AllocateDeviceMemory<float>(1);
  auto d_output = AllocateDeviceMemory<int64_t>(1);
  // No rows: nothing to do, and no buffer is required.
  ASSERT_STATUS_OK((arg_min_max_last_axis<float, true>(0, d_input.get(), d_output.get(), 0, 1024, nullptr, 0)));
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());
  ASSERT_EQ(compute_arg_min_max_last_axis_buffer_size<float>(0, 1024), 0u);
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisInvalidBufferSize) {
  const int m = 4;
  const int n = 202048;
  const size_t buffer_size_in_bytes = compute_arg_min_max_last_axis_buffer_size<float>(m, n);
  ASSERT_GT(buffer_size_in_bytes, 0u);

  auto d_input = AllocateDeviceMemory<float>(static_cast<size_t>(m) * n);
  auto d_output = AllocateDeviceMemory<int64_t>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  const auto status = arg_min_max_last_axis<float, true>(
      0, d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes / 10);
  ASSERT_FALSE(status.IsOK());
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisBufferOffsets) {
  const int m = 4;
  const int n = 202048;
  const size_t max_buffer_offset = 15;
  const size_t buffer_size_in_bytes =
      compute_arg_min_max_last_axis_buffer_size<double>(m, n) + max_buffer_offset;

  const auto values = MakeRandomValues<double>(m, n, 7);
  auto d_input = AllocateDeviceMemory<double>(static_cast<size_t>(m) * n);
  auto d_output = AllocateDeviceMemory<int64_t>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);
  cudaMemcpy(d_input.get(), values.data(), static_cast<size_t>(m) * n * sizeof(double), cudaMemcpyHostToDevice);
  const auto expected = ExpectedArgMinMaxIndices<double, true>(values, m, n);

  for (size_t buffer_offset = 1; buffer_offset <= max_buffer_offset; ++buffer_offset) {
    SCOPED_TRACE(MakeString("buffer offset: ", buffer_offset));
    ASSERT_STATUS_OK((arg_min_max_last_axis<double, true>(
        0, d_input.get(), d_output.get(), m, n, d_buffer.get() + buffer_offset,
        buffer_size_in_bytes - buffer_offset)));
    ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

    std::vector<int64_t> actual(m);
    cudaMemcpy(actual.data(), d_output.get(), static_cast<size_t>(m) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int row = 0; row < m; ++row) {
      ASSERT_EQ(actual[row], expected[row]) << "row: " << row;
    }
  }
}

TEST(ReductionFunctionsTest, ArgMinMaxLastAxisCudaGraphCaptureAndReplay) {
  const int m = 2;
  const int n = 202048;
  const auto values = MakeRandomValues<float>(m, n, 11);
  const auto expected = ExpectedArgMinMaxIndices<float, true>(values, m, n);

  auto d_input = AllocateDeviceMemory<float>(static_cast<size_t>(m) * n);
  auto d_output = AllocateDeviceMemory<int64_t>(m);
  const size_t buffer_size_in_bytes = compute_arg_min_max_last_axis_buffer_size<float>(m, n);
  ASSERT_GT(buffer_size_in_bytes, 0u);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);
  cudaMemcpy(d_input.get(), values.data(), static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t stream = nullptr;
  ASSERT_TRUE(CUDA_CALL(cudaStreamCreate(&stream)).IsOK());

  // Warm up outside of the capture, the same way the CUDA EP does.
  ASSERT_STATUS_OK((arg_min_max_last_axis<float, true>(
      stream, d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes)));
  ASSERT_TRUE(CUDA_CALL(cudaStreamSynchronize(stream)).IsOK());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_TRUE(CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)).IsOK());
  ASSERT_STATUS_OK((arg_min_max_last_axis<float, true>(
      stream, d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes)));
  ASSERT_TRUE(CUDA_CALL(cudaStreamEndCapture(stream, &graph)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0)).IsOK());

  // Replaying must give the same answer every time: the block done counters are
  // reset by the captured memset, not by the host.
  for (int replay = 0; replay < 3; ++replay) {
    cudaMemsetAsync(d_output.get(), 0xff, static_cast<size_t>(m) * sizeof(int64_t), stream);
    ASSERT_TRUE(CUDA_CALL(cudaGraphLaunch(graph_exec, stream)).IsOK());
    ASSERT_TRUE(CUDA_CALL(cudaStreamSynchronize(stream)).IsOK());

    std::vector<int64_t> actual(m);
    cudaMemcpy(actual.data(), d_output.get(), static_cast<size_t>(m) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int row = 0; row < m; ++row) {
      ASSERT_EQ(actual[row], expected[row]) << "replay: " << replay << ", row: " << row;
    }
  }

  ASSERT_TRUE(CUDA_CALL(cudaGraphExecDestroy(graph_exec)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaGraphDestroy(graph)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaStreamDestroy(stream)).IsOK());
}

// Not run by default: prints a timing table for the shapes that matter for the
// last axis ArgMax/ArgMin dispatch. Run it with
// "GTEST_FILTER=ReductionFunctionsTest.DISABLED_ArgMinMaxLastAxisPerf
// onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
// --gtest_also_run_disabled_tests".
TEST(ReductionFunctionsTest, DISABLED_ArgMinMaxLastAxisPerf) {
  struct Shape {
    int m;
    int n;
  };
  const std::vector<Shape> shapes = {
      // few rows, very wide: sampling / classification over a large vocabulary
      {1, 32000},
      {1, 50257},
      {1, 128256},
      {1, 151936},
      {1, 200000},
      {1, 202048},
      {1, 262144},
      {2, 202048},
      {4, 202048},
      {8, 202048},
      {16, 202048},
      {32, 202048},
      // many rows
      {4096, 128},
      {4096, 1024},
      {4096, 4096},
      {65536, 256},
      {65536, 1024},
      // narrow rows, handled by the one thread per row kernel
      {4096, 8},
      {4096, 32},
      {4096, 64},
      {65536, 64},
  };

  constexpr int kWarmupIterations = 20;
  constexpr int kTimedIterations = 100;

  cudaStream_t stream = nullptr;
  ASSERT_TRUE(CUDA_CALL(cudaStreamCreate(&stream)).IsOK());
  cudaEvent_t start, stop;
  ASSERT_TRUE(CUDA_CALL(cudaEventCreate(&start)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaEventCreate(&stop)).IsOK());

  std::cout << "rows,cols,scratch_bytes,microseconds\n";
  for (const auto& shape : shapes) {
    const auto values = MakeRandomValues<float>(shape.m, shape.n, static_cast<uint32_t>(shape.n));
    auto d_input = AllocateDeviceMemory<float>(static_cast<size_t>(shape.m) * shape.n);
    auto d_output = AllocateDeviceMemory<int64_t>(shape.m);
    cudaMemcpy(d_input.get(), values.data(), static_cast<size_t>(shape.m) * shape.n * sizeof(float),
               cudaMemcpyHostToDevice);
    const size_t buffer_size_in_bytes = compute_arg_min_max_last_axis_buffer_size<float>(shape.m, shape.n);
    auto d_buffer = AllocateDeviceMemory<char>(std::max<size_t>(buffer_size_in_bytes, 1));

    for (int i = 0; i < kWarmupIterations; ++i) {
      ASSERT_STATUS_OK((arg_min_max_last_axis<float, true>(stream, d_input.get(), d_output.get(), shape.m,
                                                           shape.n, d_buffer.get(), buffer_size_in_bytes)));
    }
    ASSERT_TRUE(CUDA_CALL(cudaStreamSynchronize(stream)).IsOK());

    ASSERT_TRUE(CUDA_CALL(cudaEventRecord(start, stream)).IsOK());
    for (int i = 0; i < kTimedIterations; ++i) {
      ASSERT_STATUS_OK((arg_min_max_last_axis<float, true>(stream, d_input.get(), d_output.get(), shape.m,
                                                           shape.n, d_buffer.get(), buffer_size_in_bytes)));
    }
    ASSERT_TRUE(CUDA_CALL(cudaEventRecord(stop, stream)).IsOK());
    ASSERT_TRUE(CUDA_CALL(cudaEventSynchronize(stop)).IsOK());

    float elapsed_ms = 0.0f;
    ASSERT_TRUE(CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, stop)).IsOK());
    std::cout << shape.m << "," << shape.n << "," << buffer_size_in_bytes << ","
              << (elapsed_ms * 1000.0f / kTimedIterations) << "\n";
  }

  ASSERT_TRUE(CUDA_CALL(cudaEventDestroy(start)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaEventDestroy(stop)).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaStreamDestroy(stream)).IsOK());
}

TEST(ReductionFunctionsTest, GetApplicableMatrixReduction) {
  auto test_get_applicable_matrix_reduction =
      [](cudnnReduceTensorOp_t cudnn_op,
         const std::vector<int64_t>& dims, const std::vector<int64_t>& axes,
         ApplicableMatrixReduction expected_reduction,
         const optional<int>& expected_m = nullopt,
         const optional<int>& expected_n = nullopt) {
        SCOPED_TRACE(MakeString(
            "cudnn_op: ", cudnn_op,
            ", dims: ", TensorShape::FromExistingBuffer(dims),
            ", axes: ", TensorShape::FromExistingBuffer(axes)));
        int m{}, n{};
        EXPECT_EQ(
            static_cast<int>(get_applicable_matrix_reduction(cudnn_op, dims, axes, m, n)),
            static_cast<int>(expected_reduction));
        if (expected_m) {
          EXPECT_EQ(m, *expected_m);
        }
        if (expected_n) {
          EXPECT_EQ(n, *expected_n);
        }
      };

  const cudnnReduceTensorOp_t valid_op_type = CUDNN_REDUCE_TENSOR_ADD;

  // contiguous axes from beginning
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {0, 1},
      ApplicableMatrixReduction::Rows, 2 * 4, 8 * 16);

  // contiguous axes to end
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {1, 2, 3},
      ApplicableMatrixReduction::Columns, 2, 4 * 8 * 16);

  // single axis
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {3},
      ApplicableMatrixReduction::Columns, 2 * 4 * 8, 16);

  // empty axes
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {},
      ApplicableMatrixReduction::Rows, 2 * 4 * 8 * 16, 1);

  // all axes
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {0, 1, 2, 3},
      ApplicableMatrixReduction::Rows, 2 * 4 * 8 * 16, 1);

  // handle ones
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {0},
      ApplicableMatrixReduction::Columns, 2 * 4 * 8, 1);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {1},
      ApplicableMatrixReduction::Rows, 2, 4 * 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {1, 3},
      ApplicableMatrixReduction::Rows, 2, 4 * 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {1, 3, 4},
      ApplicableMatrixReduction::Rows, 2 * 4, 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {1, 3, 4, 6},
      ApplicableMatrixReduction::Rows, 2 * 4 * 8, 1);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {3, 4, 6},
      ApplicableMatrixReduction::Columns, 2, 4 * 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {4, 6},
      ApplicableMatrixReduction::Columns, 2, 4 * 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {6},
      ApplicableMatrixReduction::Columns, 2 * 4, 8);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {7},
      ApplicableMatrixReduction::Columns, 2 * 4 * 8, 1);

  // unsupported axes
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16, 32, 64}, {0, 1, 3, 4},
      ApplicableMatrixReduction::None);
  test_get_applicable_matrix_reduction(
      valid_op_type, {2, 4, 8, 16}, {1, 2},
      ApplicableMatrixReduction::None);
  test_get_applicable_matrix_reduction(
      valid_op_type, {1, 2, 1, 1, 4, 1, 8, 1}, {3, 6},
      ApplicableMatrixReduction::Columns, 2 * 4, 8);

  // invalid op type
  test_get_applicable_matrix_reduction(
      CUDNN_REDUCE_TENSOR_MAX, {2, 4, 8, 16}, {0, 1},
      ApplicableMatrixReduction::None);
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
