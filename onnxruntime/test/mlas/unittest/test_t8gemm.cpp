// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <random>

#include "test_util.h"
#include "core/mlas/inc/mlas_t8.h"
#include "core/mlas/lib/mlasi.h"

//
// Checks MlasT8Gemm and MlasT8GemmPacked against a double-precision reference
// over shapes that cover padded output units, odd group counts (the x86
// kernels decode several groups at once) and both row tiles. On x86 each
// available SIMD kernel is also called directly, so the AVX2 kernel is tested
// on AVX-512 hosts too.
//

class MlasT8GemmTest : public MlasTestBase {
 private:
  struct Problem {
    size_t M, N, K;
    std::vector<float> A;
    std::vector<uint32_t> Planes;
    std::vector<MLAS_FP16> Scales;
    std::vector<double> Reference;
    std::vector<double> Magnitude;  // sum of |a * w|, bounds the rounding error
  };

  std::mt19937 Rng{1234};

  Problem Generate(size_t M, size_t N, size_t K) {
    Problem p{M, N, K, {}, {}, {}, {}, {}};
    const size_t units = (N + 4) / 5, sets = K / 32, groups = K / 128;
    std::uniform_int_distribution<int> trit_dist(0, 2);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

    std::vector<int> trits(units * 5 * K, 0);  // [unit*5 + j][k], padded rows stay 0
    for (size_t row = 0; row < N; ++row) {
      for (size_t k = 0; k < K; ++k) trits[row * K + k] = trit_dist(Rng);
    }
    p.Planes.assign(units * 8 * sets, 0);
    for (size_t q = 0; q < units; ++q) {
      for (size_t k = 0; k < K; ++k) {
        uint32_t byte = 0;
        for (size_t j = 5; j-- > 0;) byte = byte * 3 + static_cast<uint32_t>(trits[(q * 5 + j) * K + k]);
        for (size_t b = 0; b < 8; ++b) p.Planes[(q * 8 + b) * sets + k / 32] |= ((byte >> b) & 1u) << (k % 32);
      }
    }
    p.Scales.resize(units * groups * 5);
    for (auto& s : p.Scales) s = MLAS_FP16(scale_dist(Rng));
    p.A.resize(M * K);
    for (auto& a : p.A) a = a_dist(Rng);

    p.Reference.assign(M * N, 0.0);
    p.Magnitude.assign(M * N, 0.0);
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
          const int t = trits[n * K + k];
          const double w = (t == 1 ? 1.0 : t == 2 ? -1.0
                                                  : 0.0) *
                           static_cast<double>(p.Scales[((n / 5) * groups + k / 128) * 5 + n % 5].ToFloat());
          p.Reference[m * N + n] += static_cast<double>(p.A[m * K + k]) * w;
          p.Magnitude[m * N + n] += std::abs(static_cast<double>(p.A[m * K + k]) * w);
        }
      }
    }
    return p;
  }

  static void Check(const Problem& p, const std::vector<float>& C, const char* what) {
    for (size_t i = 0; i < p.M * p.N; ++i) {
      const double tolerance = 1e-5 * p.Magnitude[i] + 1e-6;
      ASSERT_NEAR(C[i], p.Reference[i], tolerance)
          << what << " M=" << p.M << " N=" << p.N << " K=" << p.K << " index=" << i;
    }
  }

  void Test(size_t M, size_t N, size_t K) {
    const Problem p = Generate(M, N, K);
    TestDispatched(p);
#if defined(MLAS_TARGET_AMD64)
    // Clearing the platform kernel routes the public entry points to the
    // portable fallback.
    MLAS_T8_GEMM_KERNEL* kernel = GetMlasPlatform().T8GemmKernel;
    if (kernel != nullptr) {
      GetMlasPlatform().T8GemmKernel = nullptr;
      TestDispatched(p);
      GetMlasPlatform().T8GemmKernel = kernel;
    }
    TestKernels(p);
#endif
  }

  void TestDispatched(const Problem& p) {
    const size_t M = p.M, N = p.N, K = p.K;
    const size_t units = (N + 4) / 5, groups = K / 128;
    std::vector<float> workspace(M * K);
    std::vector<float> c(M * N);

    MlasT8Gemm(M, N, K, p.A.data(), p.Planes.data(), p.Scales.data(), c.data(), workspace.data(), nullptr);
    Check(p, c, "MlasT8Gemm");

    std::vector<uint32_t> tiled(units * groups * 32), masks(units * groups * 40);
    MlasT8Pack(N, K, p.Planes.data(), tiled.data(), MLAS_T8_PACKED_LAYOUT::Tiled);
    MlasT8Pack(N, K, p.Planes.data(), masks.data(), MLAS_T8_PACKED_LAYOUT::Masks);
    for (size_t row_tile : {size_t{1}, size_t{2}}) {
      std::fill(c.begin(), c.end(), 0.0f);
      MlasT8GemmPacked(M, N, K, p.A.data(), tiled.data(), p.Scales.data(), c.data(), workspace.data(), nullptr,
                       MLAS_T8_PACKED_LAYOUT::Tiled, row_tile);
      Check(p, c, row_tile == 1 ? "Tiled/1" : "Tiled/2");
      std::fill(c.begin(), c.end(), 0.0f);
      MlasT8GemmPacked(M, N, K, p.A.data(), masks.data(), p.Scales.data(), c.data(), workspace.data(), nullptr,
                       MLAS_T8_PACKED_LAYOUT::Masks, row_tile);
      Check(p, c, row_tile == 1 ? "Masks/1" : "Masks/2");
    }
  }

#if defined(MLAS_TARGET_AMD64)
  void TestKernels(const Problem& p) {
    const size_t M = p.M, N = p.N, K = p.K;
    const size_t units = (N + 4) / 5, groups = K / 128;
    std::vector<float> c(M * N);
    std::vector<uint32_t> tiled(units * groups * 32), masks(units * groups * 40);
    MlasT8Pack(N, K, p.Planes.data(), tiled.data(), MLAS_T8_PACKED_LAYOUT::Tiled);
    MlasT8Pack(N, K, p.Planes.data(), masks.data(), MLAS_T8_PACKED_LAYOUT::Masks);

    // Each SIMD kernel must give the same bits for every layout and row tile.
    std::vector<std::pair<MLAS_T8_GEMM_KERNEL*, const char*>> kernels;
    if (GetMlasPlatform().T8GemmKernel != nullptr) kernels.emplace_back(MlasT8GemmKernelAvx2, "Avx2");
    if (GetMlasPlatform().T8GemmKernel == MlasT8GemmKernelAvx512F) kernels.emplace_back(MlasT8GemmKernelAvx512F, "Avx512F");
    for (const auto& [kernel, name] : kernels) {
      std::vector<float> first;
      for (const auto layout : {MLAS_T8_WEIGHT_LAYOUT::Planes, MLAS_T8_WEIGHT_LAYOUT::Tiled, MLAS_T8_WEIGHT_LAYOUT::Masks}) {
        const uint32_t* b = layout == MLAS_T8_WEIGHT_LAYOUT::Planes  ? p.Planes.data()
                            : layout == MLAS_T8_WEIGHT_LAYOUT::Tiled ? tiled.data()
                                                                     : masks.data();
        for (size_t rows : {size_t{1}, size_t{2}}) {
          std::fill(c.begin(), c.end(), 0.0f);
          for (size_t m = 0; m < M; m += rows) {
            kernel(std::min(rows, M - m), N, K, 0, units, p.A.data() + m * K, b, p.Scales.data(), c.data() + m * N, layout);
          }
          Check(p, c, name);
          if (first.empty()) {
            first = c;
          } else {
            ASSERT_EQ(std::memcmp(first.data(), c.data(), c.size() * sizeof(float)), 0)
                << name << " layout=" << static_cast<int>(layout) << " rows=" << rows << " M=" << M << " N=" << N << " K=" << K;
          }
        }
      }
    }
  }
#endif

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("T8Gemm");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t M : {size_t{1}, size_t{2}, size_t{3}, size_t{5}}) {
      for (size_t N : {size_t{1}, size_t{5}, size_t{7}, size_t{23}, size_t{84}}) {
        for (size_t K : {size_t{128}, size_t{256}, size_t{384}, size_t{640}, size_t{1152}}) {
          Test(M, N, K);
        }
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasT8GemmTest>::RegisterShortExecute() : 0;
});
