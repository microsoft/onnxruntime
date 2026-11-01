// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests of the tensor-core INT8 GEMV (fpA_intB_gemv/int8_tc_gemv.cuh) through fpA_intB_gemv::kernel_launcher:
//  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
#if USE_FPA_INTB_GEMM
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace wo = onnxruntime::llm::kernels::fpA_intB_gemv;

namespace {

using bf16 = __nv_bfloat16;

// The weight layout of the SM80 packing, which MatMulNBits uses for SM 9.0 devices as well (the kernel arch passed to
// kernel_launcher).
constexpr int kPackingArch = 80;

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CheckCuda(cudaMalloc(&data_, bytes));
  }
  ~DeviceBuffer() {
    cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  void Upload(const void* src) { CheckCuda(cudaMemcpy(data_, src, bytes_, cudaMemcpyHostToDevice)); }

 private:
  void* data_ = nullptr;
  size_t bytes_;
};

// A BF16 x symmetric INT8 group-wise problem. The weight bytes are random: the scalar kernel defines what a byte of the
// interleaved layout means, and the tensor-core kernel has to read the same weights.
struct Problem {
  Problem(int m_, int n_, int k_, int group_size_, uint32_t seed)
      : m(m_),
        n(n_),
        k(k_),
        group_size(group_size_),
        act(static_cast<size_t>(m_) * k_),
        weight(static_cast<size_t>(n_) * k_),
        scales(static_cast<size_t>(k_ / group_size_) * n_),
        d_act(act.size() * sizeof(bf16)),
        d_weight(weight.size()),
        d_scales(scales.size() * sizeof(bf16)),
        d_out(static_cast<size_t>(m_) * n_ * sizeof(bf16)) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> act_dist(-1.f, 1.f);
    std::uniform_real_distribution<float> scale_dist(0.05f, 1.5f);
    for (bf16& v : act) {
      v = __float2bfloat16(act_dist(gen));
    }
    for (uint8_t& v : weight) {
      v = static_cast<uint8_t>(gen() & 0xFF);
    }
    for (bf16& s : scales) {
      s = __float2bfloat16(((gen() & 1) != 0 ? 1.f : -1.f) * scale_dist(gen));
    }
    d_weight.Upload(weight.data());
    d_scales.Upload(scales.data());
    UploadActivations();
  }

  void UploadActivations() { d_act.Upload(act.data()); }

  // Row t becomes `value` at K position first + t (zero elsewhere), for all rows.
  void SetOneHotActivations(int first, float value = 1.f) {
    std::fill(act.begin(), act.end(), __float2bfloat16(0.f));
    for (int t = 0; t < m; ++t) {
      act[static_cast<size_t>(t) * k + first + t] = __float2bfloat16(value);
    }
    UploadActivations();
  }

  int m, n, k, group_size;
  std::vector<bf16> act;        // [m][k]
  std::vector<uint8_t> weight;  // interleaved, n * k bytes
  std::vector<bf16> scales;     // [k / group_size][n]
  DeviceBuffer d_act, d_weight, d_scales, d_out;
};

// Scales of random sign and 7-bit mantissa with exponents spread over 2^-133 .. 2^100 (including bf16 subnormals), so
// that (q - 128) * scale rounds to bf16 in many different ways. Columns 0..7 hold +-2^64, +-2^-80, +-2^-127 and
// +-2^-130 in every group.
void SetWideRangeScales(Problem& p, uint32_t seed) {
  std::mt19937 gen(seed);
  for (bf16& s : p.scales) {
    const float magnitude =
        std::ldexp(1.f + static_cast<float>(gen() % 128) / 128.f, static_cast<int>(gen() % 234) - 133);
    s = __float2bfloat16(((gen() & 1) != 0 ? 1.f : -1.f) * magnitude);
  }
  constexpr float kExtremes[] = {0x1p64f, -0x1p64f, 0x1p-80f, -0x1p-80f, 0x1p-127f, -0x1p-127f, 0x1p-130f, -0x1p-130f};
  for (int g = 0; g < p.k / p.group_size; ++g) {
    for (int c = 0; c < 8; ++c) {
      p.scales[static_cast<size_t>(g) * p.n + c] = __float2bfloat16(kExtremes[c]);
    }
  }
  p.d_scales.Upload(p.scales.data());
}

class Int8TensorCoreGemvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    CheckCuda(cudaGetDeviceProperties(&prop_, device));
    if (prop_.major != 9 || prop_.minor != 0) {
      GTEST_SKIP() << "The tensor-core INT8 GEMV is enabled on SM 9.0 only";
    }
    CheckCuda(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  static std::vector<int> GroupSizes() {
#if USE_COMPACT_FPA_INTB_GEMM
    return {32};
#else
    return {32, 64, 128};
#endif
  }

  // Launches rows [first_row, first_row + rows) of the problem and returns their [rows][n] output. `reported_sm` and
  // `sm_count` are what MatMulNBits reports to the dispatcher. A non-null `log` receives the debug line of the launch.
  std::vector<bf16> Run(Problem& p, int rows, int first_row, bool tensor_core, int sm_count,
                        std::string* log = nullptr, int reported_sm = 90, float alpha = 1.f) {
    wo::Params params(static_cast<const bf16*>(p.d_act.data()) + static_cast<size_t>(first_row) * p.k, nullptr,
                      p.d_weight.data(), p.d_scales.data(), nullptr, nullptr, p.d_out.data(), alpha, rows, p.n, p.k,
                      p.group_size, wo::KernelType::BF16Int8Groupwise);
    params.device_sm = reported_sm;
    params.device_sm_count = sm_count;
    params.tensor_core = tensor_core;
    params.debug = log != nullptr;
    // Start from a NaN pattern so that an output that is never written cannot match by accident.
    CheckCuda(cudaMemsetAsync(p.d_out.data(), 0xFF, p.d_out.bytes(), stream_));
    if (log != nullptr) {
      testing::internal::CaptureStdout();
    }
    try {
      wo::kernel_launcher(kPackingArch, params, stream_);
      CheckCuda(cudaGetLastError());
      CheckCuda(cudaStreamSynchronize(stream_));
    } catch (...) {
      if (log != nullptr) {
        testing::internal::GetCapturedStdout();
      }
      throw;
    }
    if (log != nullptr) {
      *log = testing::internal::GetCapturedStdout();
    }
    std::vector<bf16> out(static_cast<size_t>(rows) * p.n);
    CheckCuda(cudaMemcpy(out.data(), p.d_out.data(), out.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    return out;
  }

  // The dequantized weights [n][k] as the scalar kernel reads the weight bytes. A one-hot activation row makes an output
  // the single product bf16((q - 128) * scale), so probing every K position recovers the whole matrix.
  std::vector<float> ProbeWeights(Problem& p) {
    EXPECT_EQ(p.m, 8);
    const std::vector<bf16> saved = p.act;
    std::vector<float> w(static_cast<size_t>(p.n) * p.k);
    for (int first = 0; first < p.k; first += 8) {
      p.SetOneHotActivations(first);
      const std::vector<bf16> out = Run(p, 8, 0, false, prop_.multiProcessorCount);
      for (int t = 0; t < 8; ++t) {
        for (int c = 0; c < p.n; ++c) {
          w[static_cast<size_t>(c) * p.k + first + t] = __bfloat162float(out[static_cast<size_t>(t) * p.n + c]);
        }
      }
    }
    p.act = saved;
    p.UploadActivations();
    return w;
  }

  // Every output within the bf16 rounding of the double-precision result, plus the fp32 accumulation error.
  static void ExpectMatchesReference(const Problem& p, const std::vector<float>& w, const std::vector<bf16>& out,
                                     int rows) {
    int bad = 0;
    for (int t = 0; t < rows; ++t) {
      const bf16* a = &p.act[static_cast<size_t>(t) * p.k];
      for (int c = 0; c < p.n; ++c) {
        const float* wc = &w[static_cast<size_t>(c) * p.k];
        double ref = 0.0;
        double abs_sum = 0.0;
        for (int i = 0; i < p.k; ++i) {
          const double term = static_cast<double>(wc[i]) * static_cast<double>(__bfloat162float(a[i]));
          ref += term;
          abs_sum += std::abs(term);
        }
        const double got = static_cast<double>(__bfloat162float(out[static_cast<size_t>(t) * p.n + c]));
        if (!(std::abs(got - ref) <= 0.0042 * std::abs(ref) + 1e-6 * abs_sum) && bad++ < 5) {
          ADD_FAILURE() << "row " << t << " column " << c << ": got " << got << ", reference " << ref
                        << " (n=" << p.n << " k=" << p.k << " group=" << p.group_size << " rows=" << rows << ")";
        }
      }
    }
    EXPECT_EQ(bad, 0);
  }

  static bool BitwiseEqual(const std::vector<bf16>& a, const bf16* b) {
    return std::memcmp(a.data(), b, a.size() * sizeof(bf16)) == 0;
  }

  // One-hot activation rows at every K position (8 per launch): the tensor-core output has to equal the scalar
  // kernel's under each of the four launch configurations that the reported SM counts select for N = 480 (30 blocks).
  void ExpectOneHotMatchesScalarKernel(Problem& p) {
    ASSERT_EQ(p.n, 480);
    struct Expected {
      int sm_count;
      const char* config;
    };
    const Expected expected[] = {{16, "KWarps=8 Depth=4 BlocksPerSm=2"},
                                 {11, "KWarps=8 Depth=2 BlocksPerSm=3"},
                                 {5, "KWarps=4 Depth=2 BlocksPerSm=7"},
                                 {4, "KWarps=4 Depth=2 BlocksPerSm=8"}};
    for (int first = 0; first < p.k; first += 8) {
      p.SetOneHotActivations(first);
      const std::vector<bf16> scalar = Run(p, 8, 0, false, prop_.multiProcessorCount);
      for (const Expected& e : expected) {
        // Capturing stdout is slow, so only the first launch of a configuration reports it.
        std::string log;
        const std::vector<bf16> tc = Run(p, 8, 0, true, e.sm_count, first == 0 ? &log : nullptr);
        if (first == 0) {
          ASSERT_NE(log.find(e.config), std::string::npos) << log;
        }
        int mismatches = 0;
        for (size_t i = 0; i < tc.size(); ++i) {
          if (__bfloat162float(tc[i]) != __bfloat162float(scalar[i]) && mismatches++ < 3) {
            ADD_FAILURE() << "K positions " << first << ".." << first + 7 << ", row " << i / p.n << ", column "
                          << i % p.n << ": tensor core " << __bfloat162float(tc[i]) << ", scalar "
                          << __bfloat162float(scalar[i]) << ", " << e.config;
          }
        }
        ASSERT_EQ(mismatches, 0);
      }
    }
  }

  cudaDeviceProp prop_{};
  cudaStream_t stream_ = nullptr;
};

}  // namespace

// A one-hot activation row makes the output the dequantized weight at one K position. The scalar kernel and the
// tensor-core kernel round (q - 128) * scale to bf16 once, so their outputs have to be equal as values (a zero result
// may have either sign); probing every K position checks the weight layout (the K order inside a 64-wide tile, the
// column pairs and the scale rows) of every launch configuration. N = 480 is 30 blocks, which the reported SM counts
// 16 / 11 / 5 / 4 spread over the four configurations.
TEST_F(Int8TensorCoreGemvTest, OneHotActivationsMatchScalarKernel) {
  for (int group_size : GroupSizes()) {
    std::vector<int> ks = {128, 384, 640};
    if (group_size == 32) {
      ks.push_back(2560);
    }
    for (int k : ks) {
      SCOPED_TRACE("group=" + std::to_string(group_size) + " k=" + std::to_string(k));
      Problem p(8, 480, k, group_size, 1);
      ASSERT_NO_FATAL_FAILURE(ExpectOneHotMatchesScalarKernel(p));
    }
  }
}

// The same probe with group scales over most of the bf16 range, so the products round in many different ways.
TEST_F(Int8TensorCoreGemvTest, WideScaleRangeMatchesScalarKernel) {
  for (int group_size : GroupSizes()) {
    SCOPED_TRACE("group=" + std::to_string(group_size));
    Problem p(8, 480, 384, group_size, 6);
    SetWideRangeScales(p, 7);
    ASSERT_NO_FATAL_FAILURE(ExpectOneHotMatchesScalarKernel(p));
  }
}

// The kernel handles every finite bf16 scale (including subnormal scales below 2^-119 down to 2^-133) whose products
// stay finite. A fixed power-of-two pre-scaling of the weight codes would overflow for code 129 with scale 2^64 and
// flush the weight to zero for scale 2^-80, while dividing the weight by 128 before rounding to bf16 would lose or
// distort tiny weights below 2^-119 (such as code 129 with scale 2^-130 or code 131 with scale 2^-127). Uniform codes
// at the exponent extremes are compared against the bf16 rounding of the exact product and the scalar kernel.
TEST_F(Int8TensorCoreGemvTest, ExtremeScalesMatchReferenceWeights) {
  constexpr float kMagnitudes[] = {0x1p100f, 0x1p64f, 0x1p-80f, 0x1p-110f, 0x1p-119f, 0x1p-120f, 0x1p-126f,
                                   0x1p-127f, 0x1p-130f, 0x1p-133f};
  for (int group_size : GroupSizes()) {
    Problem p(8, 480, 128, group_size, 1);
    for (float act_val : {1.f, 0x1p120f}) {
      p.SetOneHotActivations(0, act_val);
      for (int code : {0, 1, 125, 127, 128, 129, 131, 254, 255}) {
        std::fill(p.weight.begin(), p.weight.end(), static_cast<uint8_t>(code));
        p.d_weight.Upload(p.weight.data());
        for (float magnitude : kMagnitudes) {
          if (act_val > 1.f && magnitude > 0x1p-119f) {
            continue;
          }
          for (float scale : {magnitude, -magnitude}) {
            SCOPED_TRACE(testing::Message()
                         << "group=" << group_size << " act=" << act_val << " code=" << code << " scale=" << scale);
            std::fill(p.scales.begin(), p.scales.end(), __float2bfloat16(scale));
            p.d_scales.Upload(p.scales.data());
            const float weight_bf16 =
                __bfloat162float(__float2bfloat16(static_cast<float>(code - 128) * __bfloat162float(__float2bfloat16(scale))));
            const float expected = __bfloat162float(__float2bfloat16(weight_bf16 * act_val));
            const std::vector<bf16> scalar = Run(p, 8, 0, false, prop_.multiProcessorCount);
            for (size_t i = 0; i < scalar.size(); ++i) {
              ASSERT_EQ(__bfloat162float(scalar[i]), expected) << "scalar kernel, output " << i;
            }
            for (int rows : {1, 8}) {
              if (rows == 1 && (code != 129 || magnitude != 0x1p-130f) && (code != 131 || magnitude != 0x1p-127f)) {
                continue;
              }
              for (int sm_count : {16, 11, 5, 4}) {
                std::string log;
                const std::vector<bf16> tc = Run(p, rows, 0, true, sm_count, &log);
                ASSERT_NE(log.find("tensor_core=1"), std::string::npos) << log;
                for (size_t i = 0; i < tc.size(); ++i) {
                  ASSERT_EQ(__bfloat162float(tc[i]), expected)
                      << "rows=" << rows << " sm_count=" << sm_count << ", output " << i;
                }
              }
            }
          }
        }
      }
    }
  }
}

// Random activations against a double-precision reference of the probed weights, for the Qwen3.8 dense projection
// shapes (in_proj N = 16480 / K = 2560, out_proj N = 2560 / K = 6144), other launch configurations, every row count
// and all group sizes, with this device's SM count.
TEST_F(Int8TensorCoreGemvTest, RandomActivationsMatchReference) {
  struct Shape {
    int n, k;
    bool all_groups;
    std::vector<int> rows;
  };
  const Shape shapes[] = {{480, 640, true, {1, 2, 3, 4, 5, 6, 7, 8}},
                          {1536, 2560, true, {1, 3, 8}},
                          {5120, 2560, true, {1, 5, 8}},
                          {8192, 1280, true, {8}},
                          {16480, 2560, false, {8}},
                          {2560, 6144, false, {1, 8}}};
  for (const Shape& shape : shapes) {
    for (int group_size : GroupSizes()) {
      if (group_size != 32 && !shape.all_groups) {
        continue;
      }
      SCOPED_TRACE("n=" + std::to_string(shape.n) + " k=" + std::to_string(shape.k) +
                   " group=" + std::to_string(group_size));
      Problem p(8, shape.n, shape.k, group_size, 2);
      const std::vector<float> w = ProbeWeights(p);
      for (int rows : shape.rows) {
        std::string log;
        const std::vector<bf16> out = Run(p, rows, 0, true, prop_.multiProcessorCount, &log);
        ASSERT_NE(log.find("tensor_core=1"), std::string::npos) << log;
        ExpectMatchesReference(p, w, out, rows);
      }
    }
  }
}

// A row's result does not depend on how many other rows are in the launch (every MMA output is contracted on its own
// and the launch configuration depends on N only), so one M = 8 launch equals launches of fewer rows and single rows.
TEST_F(Int8TensorCoreGemvTest, RowResultsDoNotDependOnRowCount) {
  for (int group_size : GroupSizes()) {
    SCOPED_TRACE("group=" + std::to_string(group_size));
    Problem p(8, 5120, 2560, group_size, 3);
    const std::vector<bf16> all = Run(p, 8, 0, true, prop_.multiProcessorCount);
    for (int rows : {1, 2, 3, 5, 7}) {
      EXPECT_TRUE(BitwiseEqual(Run(p, rows, 0, true, prop_.multiProcessorCount), all.data())) << "rows=" << rows;
    }
    for (int row = 1; row < 8; ++row) {
      EXPECT_TRUE(BitwiseEqual(Run(p, 1, row, true, prop_.multiProcessorCount),
                               all.data() + static_cast<size_t>(row) * p.n))
          << "row=" << row;
    }
    EXPECT_TRUE(BitwiseEqual(Run(p, 4, 4, true, prop_.multiProcessorCount), all.data() + 4 * static_cast<size_t>(p.n)));
  }
}

// The launch configuration follows the grid size (N / 16 blocks) against the SM count.
TEST_F(Int8TensorCoreGemvTest, LaunchConfigurationFollowsGridSize) {
  constexpr int kSms = 132;
  constexpr const char* kD = "KWarps=8 Depth=4 BlocksPerSm=2";
  constexpr const char* kC = "KWarps=8 Depth=2 BlocksPerSm=3";
  constexpr const char* kB = "KWarps=4 Depth=2 BlocksPerSm=7";
  constexpr const char* kA = "KWarps=4 Depth=2 BlocksPerSm=8";
  struct Case {
    int n;
    const char* config;
  };
  // 2 * 132 = 264, 3 * 132 = 396, 7 * 132 = 924 and 8 * 132 = 1056 blocks; beyond 7 * 132 the configuration with fewer
  // waves wins and the 7-block one takes ties.
  const Case cases[] = {{2560, kD}, {4224, kD}, {4240, kC}, {6336, kC}, {6352, kB}, {13952, kB}, {14784, kB}, {14800, kA}, {16480, kA}, {16896, kA}, {16912, kB}, {20480, kB}, {25344, kB}, {33792, kA}};
  for (const Case& c : cases) {
    Problem p(8, c.n, 64, 32, 4);
    std::string log;
    Run(p, 8, 0, true, kSms, &log);
    EXPECT_NE(log.find(c.config), std::string::npos) << "n=" << c.n << " expected " << c.config << ", got " << log;
  }
}

// Launches the kernel cannot take stay on the scalar kernel: the flag off, another device, an unreported SM count,
// more than 8 rows and an alpha other than 1.
TEST_F(Int8TensorCoreGemvTest, UnsupportedLaunchesUseScalarKernel) {
  constexpr int kSms = 132;
  Problem p(9, 5120, 64, 32, 5);
  std::string log;
  Run(p, 8, 0, true, kSms, &log);
  EXPECT_NE(log.find("tensor_core=1"), std::string::npos) << log;
  Run(p, 9, 0, true, kSms, &log);
  EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << "M = 9: " << log;
  Run(p, 8, 0, false, kSms, &log);
  EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << "tensor_core = false: " << log;
  Run(p, 8, 0, true, kSms, &log, 80);
  EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << "device SM 8.0: " << log;
  Run(p, 8, 0, true, 0, &log);
  EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << "unreported SM count: " << log;
  Run(p, 8, 0, true, kSms, &log, 90, 2.f);
  EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << "alpha = 2: " << log;
}

// The tensor-core epilogue has no alpha multiplier, so a launch with alpha != 1 has to run the scalar kernel, which
// scales the accumulated sum. Scaling by a power of two is exact, so the output is the scalar kernel's alpha = 1 output
// times alpha.
TEST_F(Int8TensorCoreGemvTest, NonUnitAlphaScalesTheOutput) {
  Problem p(8, 480, 640, 32, 6);
  const std::vector<bf16> unscaled = Run(p, 8, 0, false, prop_.multiProcessorCount);
  for (float alpha : {2.f, 0.5f}) {
    SCOPED_TRACE("alpha=" + std::to_string(alpha));
    std::string log;
    const std::vector<bf16> scaled = Run(p, 8, 0, true, prop_.multiProcessorCount, &log, 90, alpha);
    EXPECT_EQ(log.find("tensor_core=1"), std::string::npos) << log;
    int mismatches = 0;
    for (size_t i = 0; i < scaled.size(); ++i) {
      if (__bfloat162float(scaled[i]) != alpha * __bfloat162float(unscaled[i]) && mismatches++ < 3) {
        ADD_FAILURE() << "element " << i << ": got " << __bfloat162float(scaled[i]) << ", expected "
                      << alpha * __bfloat162float(unscaled[i]);
      }
    }
    EXPECT_EQ(mismatches, 0);
  }
}

#endif  // USE_FPA_INTB_GEMM
