// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD
#if (defined(MLAS_TARGET_AMD64_IX86) && !defined(USE_DML) && !defined(USE_WEBGPU) && !defined(USE_COREML)) || defined(USE_CUDA)

#include <optional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

namespace {

constexpr int QBits = 4;

struct TestOptions8Bits {
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  int64_t block_size{32};
  int64_t accuracy_level{0};

  bool has_zero_point{false};
  bool has_g_idx{false};
  bool has_bias{false};
  bool is_zero_point_scale_same_type{true};
  bool load_from_file{false};

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

enum class DType {
    Float32,
    Float16,
    UInt8,
    Int8
};

struct TensorInfo {
    DType dtype;
    std::vector<int> shape;

    std::vector<int64_t> dims() const {
        std::vector<int64_t> dims(shape.size());
        std::transform(shape.begin(), shape.end(), dims.begin(), [](int dim) { return static_cast<int64_t>(dim); });
        return dims;
    }

    size_t size() const {return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());}
};

template <typename T, typename T2>
std::vector<T2> parse_tensor_data(const std::string& filename, TensorInfo& info_out) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file");

    std::string dtype_line, shape_line;
    std::getline(file, dtype_line);
    std::getline(file, shape_line);

    // Parse dtype
    if (dtype_line == "float32") info_out.dtype = DType::Float32;
    else if (dtype_line == "float16") info_out.dtype = DType::Float16;
    else if (dtype_line == "uint8") info_out.dtype = DType::UInt8;
    else if (dtype_line == "int8") info_out.dtype = DType::Int8;
    else throw std::runtime_error("Unsupported dtype");

    // Parse shape
    std::istringstream shape_stream(shape_line);
    int ndim;
    shape_stream >> ndim;
    info_out.shape.resize(ndim);
    for (int i = 0; i < ndim; ++i)
        shape_stream >> info_out.shape[i];

    // Read data
    std::vector<T> values;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        T val;
        while (line_stream >> val) {
            values.push_back(val);
        }
    }

    if (values.size() != info_out.size())
        throw std::runtime_error("Data size does not match shape");

    if constexpr (std::is_same<T, T2>::value) {
        return values;
    } else {
      std::vector<T2> values_output;
      values_output.resize(values.size());
      std::transform(values.begin(), values.end(), values_output.begin(), [](T val) { return static_cast<T2>(val); });

      return values_output;
    }
}


[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const TestOptions8Bits& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", block_size:" << opts.block_size
            << ", accuracy_level:" << opts.accuracy_level
            << ", has_zero_point:" << opts.has_zero_point
            << ", has_g_idx:" << opts.has_g_idx
            << ", has_bias:" << opts.has_bias;
}

template <typename T1>
void RunTest8BitsFromFile(const TestOptions8Bits& opts) {
  const int64_t M = opts.M,
                K = opts.K,
                N = opts.N;

  TensorInfo tensor_info_activation;
  std::vector<float> activation = parse_tensor_data<float, float>("woq_2_64_128_64_activation.txt", tensor_info_activation);

  TensorInfo tensor_info_ref_weight;
  std::vector<float> ref_weight = parse_tensor_data<float, float>("woq_2_64_128_64_ref_weight.txt", tensor_info_ref_weight);

  TensorInfo tensor_info_matmulnbits_weight;
  std::vector<uint8_t> matmulnbits_weight = parse_tensor_data<int, uint8_t>("woq_2_64_128_64_matmulnbits_weight.txt", tensor_info_matmulnbits_weight);

  TensorInfo tensor_info_matmulnbits_scale;
  std::vector<float> matmulnbits_scale = parse_tensor_data<float, float>("woq_2_64_128_64_matmulnbits_scale.txt", tensor_info_matmulnbits_scale);
  std::vector<MLFloat16> matmulnbits_scale_fp16 = FloatsToMLFloat16s(matmulnbits_scale);

  TensorInfo tensor_info_matmulnbits_zp;
  std::vector<float> matmulnbits_zp = parse_tensor_data<float, float>("woq_2_64_128_64_matmulnbits_zp.txt", tensor_info_matmulnbits_zp);

  std::vector<MLFloat16> matmulnbits_zp_fp16 = FloatsToMLFloat16s(matmulnbits_zp);

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<float, QBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N),
                                            q_rows, q_cols);

  size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
  MlasBlockwiseQuantizedBufferSizes<QBits>(static_cast<int>(opts.block_size), /* columnwise */ true,
                                           static_cast<int>(K), static_cast<int>(N),
                                           q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

  assert(q_data_size_in_bytes == tensor_info_matmulnbits_weight.size());
  assert(tensor_info_matmulnbits_scale.size() == tensor_info_matmulnbits_zp.size());
  assert(q_scale_size == tensor_info_matmulnbits_scale.size());

  RandomValueGenerator random{1234};

  const std::vector<int64_t> bias_shape = {N};
  const auto bias = [&]() -> std::optional<std::vector<float>> {
    if (opts.has_bias) {
      return random.Uniform(bias_shape, 1.0f, 5.0f);
    }
    return std::nullopt;
  }();

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += activation[m * K + k] * ref_weight[k * N + n];
      }
      expected_vals[m * N + n] = sum + (bias.has_value() ? (*bias)[n] : 0.0f);
    }
  }

  printf("Expected:\n");
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      printf("%f\t", expected_vals[m * N + n]);
    }
    printf("\n");
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", opts.block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  test.AddAttribute<int64_t>("accuracy_level", opts.accuracy_level);

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("A", tensor_info_activation.dims(), activation, false);
  } else {
    test.AddInput<T1>("A", tensor_info_activation.dims(), FloatsToMLFloat16s(activation), false);
  }

  test.AddInput<uint8_t>("B", tensor_info_matmulnbits_weight.dims(), matmulnbits_weight, true);

  if constexpr (std::is_same<T1, float>::value) {
    test.AddInput<T1>("scales", tensor_info_matmulnbits_scale.dims(), matmulnbits_scale, true);
  } else {
    test.AddInput<T1>("scales", tensor_info_matmulnbits_scale.dims(), FloatsToMLFloat16s(matmulnbits_scale), true);
  }

  if (opts.has_zero_point) {
    if constexpr (std::is_same<T1, float>::value) {
      test.AddInput<T1>("zero_points", tensor_info_matmulnbits_zp.dims(), matmulnbits_zp, true);
    } else {
      test.AddInput<T1>("zero_points", tensor_info_matmulnbits_zp.dims(), FloatsToMLFloat16s(matmulnbits_zp), true);
    }
  } else { // zero point is optional
    test.AddOptionalInputEdge<uint8_t>();
  }

  if (bias.has_value()) {
    if constexpr (std::is_same<T1, float>::value) {
      test.AddInput<T1>("bias", bias_shape, *bias, true);
    } else {
      test.AddInput<T1>("bias", bias_shape, FloatsToMLFloat16s(*bias), true);
    }
  } else {
    test.AddOptionalInputEdge<T1>();
  }

  if constexpr (std::is_same<T1, float>::value) {
    test.AddOutput<T1>("Y", {M, N}, expected_vals);
  } else {
    test.AddOutput<T1>("Y", {M, N}, FloatsToMLFloat16s(expected_vals));
  }

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.emplace_back(DefaultCudaExecutionProvider());
  test.ConfigEps(std::move(execution_providers));
  test.RunWithConfig();
  execution_providers.clear();
#else
  if constexpr (std::is_same<T1, float>::value) {
    if (MlasIsQNBitGemmAvailable(8, 32, SQNBIT_CompInt8)) {
      execution_providers.emplace_back(DefaultCpuExecutionProvider());
      test.ConfigEps(std::move(execution_providers));
      test.RunWithConfig();
    }
  }
#endif
}


template <typename AType, int M, int N, int K, int block_size, int accuracy_level>
void TestMatMul8BitsTyped() {
  TestOptions8Bits base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;
  base_opts.block_size = block_size;
  base_opts.accuracy_level = accuracy_level;

  if (base_opts.accuracy_level == 4) {
    base_opts.output_abs_error = 0.1f;
    base_opts.output_rel_error = 0.02f;
  } else if constexpr (std::is_same<AType, MLFloat16>::value) {
    base_opts.output_abs_error = 0.055f;
    base_opts.output_rel_error = 0.02f;
  }

  TestOptions8Bits opts = base_opts;
  opts.has_zero_point = true;
  opts.is_zero_point_scale_same_type = true;
  opts.load_from_file = true;
  RunTest8BitsFromFile<AType>(opts);
}
}  // namespace


TEST(MatMulNBits, Fp16_int4_gemm) {
  // TestMatMul8BitsTyped<MLFloat16, 2, 4, 32, 16, 4>();
  // TestMatMul8BitsTyped<MLFloat16, 199, 40, 576, 32, 4>();
  
  TestMatMul8BitsTyped<MLFloat16, 2, 64, 128, 64, 4>();

  // TestMatMul8BitsTyped<MLFloat16, 2, 64, 64, 64, 4>();
}

}  // namespace test
}  // namespace onnxruntime

#endif
#endif  // ORT_MINIMAL_BUILD
