// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#include <memory>

#include "gtest/gtest.h"

#include "core/providers/cuda/reduction/reduction_functions.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
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
      cuda::compute_reduction_buffer_size<float>(size);

  auto device_input = AllocateDeviceMemory<float>(size);
  auto device_output_sum = AllocateDeviceMemory<float>();
  auto device_output_square_sum = AllocateDeviceMemory<float>();
  auto device_output_mean = AllocateDeviceMemory<float>();
  auto buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  cudaMemcpy(device_input.get(), input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  ASSERT_STATUS_OK(cuda::reduce_sum(
      device_input.get(),
      device_output_sum.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));
  ASSERT_STATUS_OK(cuda::reduce_square_sum(
      device_input.get(),
      device_output_square_sum.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));
  ASSERT_STATUS_OK(cuda::reduce_mean(
      device_input.get(),
      device_output_mean.get(),
      size,
      buffer.get(),
      buffer_size_in_bytes));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()));

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
    cuda::Fill(d_out.get(), initial_value, n);
  }

  ASSERT_STATUS_OK(cuda::reduce_matrix_rows(
      d_in.get(), d_out.get(),
      m, n,
      reset_initial_output));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()));

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
      cuda::compute_reduce_matrix_columns_buffer_size<float>(m, n);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  ASSERT_STATUS_OK(cuda::reduce_matrix_columns(
      d_in.get(), d_out.get(),
      m, n,
      d_buffer.get(), buffer_size_in_bytes));

  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()));

  CheckDeviceValues(m, d_out.get(), expected_column.data(), relative_error_tolerance);
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

TEST(ReductionFunctionsTest, BufferOffsets) {
  const int m = 2048;
  const int n = 1024;

  const size_t max_buffer_offset = 15;

  const size_t buffer_size_in_bytes =
      cuda::compute_reduce_matrix_columns_buffer_size<double>(m, n) + max_buffer_offset;

  auto d_input = AllocateDeviceMemory<double>(m * n);
  auto d_output = AllocateDeviceMemory<double>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  RandomValueGenerator random{};
  const float relative_error_tolerance = 1e-4f;

  for (size_t buffer_offset = 1; buffer_offset <= max_buffer_offset; ++buffer_offset) {
    SCOPED_TRACE(MakeString("buffer offset: ", buffer_offset));

    const auto input = random.Uniform<double>({m, n}, 1.0, 10.0);
    cudaMemcpy(d_input.get(), input.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    ASSERT_STATUS_OK(cuda::reduce_matrix_columns(
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

  // this should be too small
  const size_t buffer_size_in_bytes =
      cuda::compute_reduce_matrix_columns_buffer_size<float>(m, n) / 10;

  auto d_input = AllocateDeviceMemory<float>(m * n);
  auto d_output = AllocateDeviceMemory<float>(m);
  auto d_buffer = AllocateDeviceMemory<char>(buffer_size_in_bytes);

  RandomValueGenerator random{};
  const auto input = random.Uniform<float>({m, n}, 1.0, 10.0);
  cudaMemcpy(d_input.get(), input.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

  const auto status =
      cuda::reduce_matrix_columns(d_input.get(), d_output.get(), m, n, d_buffer.get(), buffer_size_in_bytes);
  ASSERT_FALSE(status.IsOK());
}

TEST(ReductionFunctionsTest, GetApplicableMatrixReduction) {
  const cudnnReduceTensorOp_t valid_op_type = CUDNN_REDUCE_TENSOR_ADD;
  int m{}, n{};

  // contiguous axes from beginning
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {0, 1}, m, n),
      cuda::ApplicableMatrixReduction::Rows);
  EXPECT_EQ(m, 2 * 4);
  EXPECT_EQ(n, 8 * 16);

  // contiguous axes to end
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {1, 2, 3}, m, n),
      cuda::ApplicableMatrixReduction::Columns);
  EXPECT_EQ(m, 2);
  EXPECT_EQ(n, 4 * 8 * 16);

  // single axis
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {3}, m, n),
      cuda::ApplicableMatrixReduction::Columns);
  EXPECT_EQ(m, 2 * 4 * 8);
  EXPECT_EQ(n, 16);

  // unsupported axes
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {0, 1, 2, 3}, m, n),
      cuda::ApplicableMatrixReduction::None);
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {}, m, n),
      cuda::ApplicableMatrixReduction::None);
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16, 32, 64}, {0, 1, 3, 4}, m, n),
      cuda::ApplicableMatrixReduction::None);
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          valid_op_type, {2, 4, 8, 16}, {1, 2}, m, n),
      cuda::ApplicableMatrixReduction::None);

  // invalid op type
  EXPECT_EQ(
      cuda::get_applicable_matrix_reduction(
          CUDNN_REDUCE_TENSOR_MAX, {2, 4, 8, 16}, {0, 1}, m, n),
      cuda::ApplicableMatrixReduction::None);
}

}  // namespace test
}  // namespace onnxruntime

#endif
