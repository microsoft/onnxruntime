// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Refer https://github.com/microsoft/onnxruntime/blob/e94153e82197bcd38a602a91831bc6835dac48af/onnxruntime/core/providers/cpu/math/matmul_helper.h#L27
Status ComputeMatMulOutputShape(const TensorShape& orig_left_shape, const TensorShape& orig_right_shape, TensorShape& output_shape,
                                int64_t& M, int64_t& K, int64_t& N) {
  // Following numpy.matmul for shape inference:
  // https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
  // The behavior depends on the arguments in the following way.
  // * If both arguments are 2 - D they are multiplied like conventional matrices.
  // * If either argument is N - D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

  size_t left_num_dims = orig_left_shape.NumDimensions();
  size_t right_num_dims = orig_right_shape.NumDimensions();

  // Special cases below for right_shape being 2D and left_shape > 2D by flattening left_shape to 2D
  // Note that padding 1s in front of the right_shape can be flattened too
  // A: [M1, M2, ... K], B: [K, N]
  // A: [M1, M2, ... K], B: [1, ..., 1, K, N]
  if (left_num_dims >= 2 && right_num_dims >= 2 && left_num_dims >= right_num_dims &&
      orig_right_shape.SizeToDimension(right_num_dims - 1) == orig_right_shape[right_num_dims - 2]) {
    M = orig_left_shape.SizeToDimension(left_num_dims - 1);
    K = orig_left_shape[left_num_dims - 1];
    N = orig_right_shape[right_num_dims - 1];
    output_shape = orig_left_shape;
    output_shape[left_num_dims - 1] = N;
    ORT_RETURN_IF_NOT((K == orig_right_shape[right_num_dims - 2]), "MatMul dimension mismatch");
    return Status::OK();
  }

  std::vector<int64_t> dims_left(left_num_dims);
  std::vector<int64_t> dims_right(right_num_dims);
  orig_left_shape.CopyDims(&dims_left[0], left_num_dims);
  orig_right_shape.CopyDims(&dims_right[0], right_num_dims);

  TensorShape left_shape(dims_left);
  TensorShape right_shape(dims_right);

  bool has_1D_input = (left_num_dims == 1 || right_num_dims == 1);

  size_t num_input_dims = std::max(left_num_dims, right_num_dims);

  // use padded dims to compute matrix offsets, right 1D would be padded
  size_t num_dims_with_pad = num_input_dims;

  // output shape would squeeze the reduced 1D dimension
  size_t num_output_dims = num_input_dims - (has_1D_input ? 1 : 0);

  auto left_padded_dims = std::vector<int64_t>(num_dims_with_pad, 1);
  auto right_padded_dims = std::vector<int64_t>(num_dims_with_pad, 1);

  // pad 1 in the front for left
  left_shape.CopyDims(&left_padded_dims[num_dims_with_pad - left_num_dims], left_num_dims);
  // pad 1 in the front for right
  right_shape.CopyDims(&right_padded_dims[num_dims_with_pad - right_num_dims], right_num_dims);

  // validate input shape and generate output shape
  std::vector<int64_t> output_dims(num_output_dims);

  // broadcasting for all output dims except last two
  for (size_t idx_dim = 0; idx_dim < num_dims_with_pad - 2; ++idx_dim) {
    output_dims[idx_dim] = std::max(left_padded_dims[idx_dim], right_padded_dims[idx_dim]);
    if (left_padded_dims[idx_dim] != output_dims[idx_dim])
      ORT_RETURN_IF_NOT(left_padded_dims[idx_dim] == 1, "left operand cannot broadcast on dim ", idx_dim);
    if (right_padded_dims[idx_dim] != output_dims[idx_dim])
      ORT_RETURN_IF_NOT(right_padded_dims[idx_dim] == 1, "right operand cannot broadcast on dim ", idx_dim);
  }

  M = has_1D_input ? 1 : left_shape[left_num_dims - 2];
  K = left_shape[left_num_dims - 1];
  N = right_shape[right_num_dims - 1];

  ORT_RETURN_IF_NOT(K == right_shape[right_num_dims - 2], "MatMul dimension mismatch");
  // left (...M x K), right (...K x N), output (...M x N)
  ORT_RETURN_IF_NOT(num_dims_with_pad == num_output_dims, "num_dims_with_pad != num_output_dims");
  output_dims[num_output_dims - 2] = M;
  output_dims[num_output_dims - 1] = N;

  // assign shape
  output_shape = TensorShape(output_dims);

  return Status::OK();
}

Status GetExpectedResult(const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                         std::vector<float>& expected_vals,
                         const TensorShape& a_shape, const TensorShape& b_shape,
                         const TensorShape& output_shape) {
  int64_t N = output_shape[output_shape.NumDimensions() - 1];
  int64_t K = a_shape[a_shape.NumDimensions() - 1];
  int64_t M = output_shape[output_shape.NumDimensions() - 2];
  int64_t batch_1 = output_shape.NumDimensions() > 2 ? output_shape[output_shape.NumDimensions() - 3] : 1;
  int64_t batch_0 = output_shape.NumDimensions() > 3 ? output_shape[output_shape.NumDimensions() - 4] : 1;
  int64_t batch_1_stride = M * N;
  int64_t batch_0_stride = batch_1 * batch_1_stride;
  int64_t a_batch_1_stride = M * K;
  int64_t a_batch_0_stride = (a_shape.NumDimensions() > 3 ? a_shape[a_shape.NumDimensions() - 3] : 1) * a_batch_1_stride;
  int64_t b_batch_1_stride = K * N;
  int64_t b_batch_0_stride = (b_shape.NumDimensions() > 3 ? b_shape[b_shape.NumDimensions() - 3] : 1) * b_batch_1_stride;
  for (int64_t i = 0; i < batch_0; i++) {
    int64_t a_batch_0_offset = a_batch_0_stride * ((a_shape.NumDimensions() < 4 || (a_shape[a_shape.NumDimensions() - 4] == 1)) ? 0 : i);
    int64_t b_batch_0_offset = b_batch_0_stride * ((b_shape.NumDimensions() < 4 || (b_shape[b_shape.NumDimensions() - 4] == 1)) ? 0 : i);
    for (int64_t j = 0; j < batch_1; j++) {
      int64_t a_batch_1_offset = a_batch_1_stride * ((a_shape.NumDimensions() < 3 || (a_shape[a_shape.NumDimensions() - 3] == 1)) ? 0 : j);
      int64_t b_batch_1_offset = b_batch_1_stride * ((b_shape.NumDimensions() < 3 || (b_shape[b_shape.NumDimensions() - 3] == 1)) ? 0 : j);
      for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; k++) {
            sum += a_vals[a_batch_0_offset + a_batch_1_offset + m * K + k] * b_vals[b_batch_0_offset + b_batch_1_offset + k * N + n];
          }
          expected_vals[i * batch_0_stride + j * batch_1_stride + m * N + n] = sum;
        }
      }
    }
  }

  return Status::OK();
}

template <typename T1, int version>
void RunTestTyped(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims) {
  ASSERT_TRUE(a_dims.size() < 5 && b_dims.size() < 5, "max supported tensor dim is 4-D.");
  ASSERT_TRUE(a_dims.size() > 1 && b_dims.size() > 1, "cannot support 1-D tensor.");
  static_assert(std::is_same_v<T1, float> || std::is_same_v<T1, MLFloat16>, "unexpected type for T1");

  int64_t M = 0;
  int64_t K = 0;
  int64_t N = 0;
  TensorShape a_shape = TensorShape(a_dims);
  TensorShape b_shape = TensorShape(b_dims);
  TensorShape output_shape{};
  auto status = ComputeMatMulOutputShape(a_shape, b_shape, output_shape, M, K, N);
  ASSERT_TRUE(status.IsOK());

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  std::vector<float> expected_vals(output_shape.Size());
  GetExpectedResult(a_vals, b_vals, expected_vals, a_shape, b_shape, output_shape);

  std::vector<int64_t> output_dims(output_shape.NumDimensions());
  output_shape.CopyDims(output_dims.data(), output_shape.NumDimensions());
  OpTester test("MatMul", version);
  if constexpr (std::is_same_v<T1, float>) {
    test.AddInput<T1>("A", a_dims, a_vals);
    test.AddInput<T1>("B", b_dims, b_vals);
    test.AddOutput<T1>("Y", output_dims, expected_vals);
  } else if constexpr (std::is_same<T1, MLFloat16>::value) {
    test.AddInput<T1>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T1>("B", b_dims, FloatsToMLFloat16s(b_vals));
    test.AddOutput<T1>("Y", output_dims, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  test.RunWithConfig();
}

TEST(MatMul_Large, Float32) {
  RunTestTyped<float, 13>({512, 1024}, {1024, 1024});
  RunTestTyped<float, 13>({511, 1024}, {1024, 1024});
  RunTestTyped<float, 13>({511, 1024}, {1024, 1023});
  RunTestTyped<float, 13>({1, 512, 1024}, {1024, 1024});
  RunTestTyped<float, 13>({2, 512, 1024}, {1024, 1024});
  RunTestTyped<float, 13>({2, 512, 1024}, {2, 1024, 1024});
  RunTestTyped<float, 13>({2, 2, 512, 1024}, {2, 1024, 1024});
}

TEST(MatMul_Large, Float16) {
  RunTestTyped<MLFloat16, 13>({512, 1024}, {1024, 1024});
  RunTestTyped<MLFloat16, 13>({511, 1024}, {1024, 1024});
  RunTestTyped<MLFloat16, 13>({511, 1024}, {1024, 1023});
  RunTestTyped<MLFloat16, 13>({1, 512, 1024}, {1024, 1024});
  RunTestTyped<MLFloat16, 13>({2, 512, 1024}, {1024, 1024});
  RunTestTyped<MLFloat16, 13>({2, 512, 1024}, {2, 1024, 1024});
  RunTestTyped<MLFloat16, 13>({2, 2, 512, 1024}, {2, 1024, 1024});
}

}  // namespace test
}  // namespace onnxruntime
