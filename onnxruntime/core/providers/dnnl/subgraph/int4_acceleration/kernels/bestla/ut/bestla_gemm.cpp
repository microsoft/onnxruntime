#include "bestla_gemm.h"
#include "bestla_utils.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;

template <int NTILE>
void ref_bf16(utils::bf16* matA, utils::bf16* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride,
              int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::bf16);
  int ldb = _bstride / sizeof(utils::bf16);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(utils::bf16);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::bf16, float>(utils::bf16{matA[i * lda + k + ik]});
            auto tmpB = utils::cast<utils::bf16, float>(utils::bf16{matB[k * NTILE + ij * KPack + ik + j * ldb]});
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp16(utils::fp16* matA, utils::fp16* matB, utils::fp16* matC, int _m, int _n, int _k, int _astride,
              int _bstride, int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::fp16);
  int ldb = _bstride / sizeof(utils::fp16);
  int ldc = _cstride / sizeof(utils::fp16);
  int constexpr KPack = 1;
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        utils::fp16 tmp = utils::fp16(0.f);
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::fp16, float>(matA[i * lda + k + ik]);
            auto tmpB = utils::cast<utils::fp16, float>(matB[k * NTILE + ij * KPack + ik + j * ldb]);
            tmp = float(tmp) + tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp32(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(float);
  int ldb = _bstride / sizeof(float);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(float);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = matA[i * lda + k + ik];
            auto tmpB = matB[k * NTILE + ij * KPack + ik + j * ldb];
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, class T_A_, class T_B_>
void ref_int8(T_A_* matA, T_B_* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(T_A_);
  int ldb = _bstride / sizeof(T_B_);
  int ldc = _cstride / sizeof(int32_t);
  int constexpr KPack = 4 / sizeof(T_B_);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        int32_t tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<T_A_, int32_t>(matA[i * lda + k + ik]);
            auto tmpB = utils::cast<T_B_, int32_t>(matB[k * NTILE + ij * KPack + ik + j * ldb]);
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, typename T_SB_>
void ref_kblock_int8(uint8_t* matA, int8_t* matB, float* matC, uint8_t* zpA, float* scaleA, int _ldsa, T_SB_* scaleB,
                     float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride,
                     int _cstride, int kpos) {
  int lda = _astride / sizeof(matA[0]);
  int ldb = _bstride / sizeof(matB[0]);
  int ldc = _cstride / sizeof(matC[0]);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          break;
        }
        float tmpf = 0.f;
        for (int k = 0; k < _k; k += _kblock) {
          int tmp = 0;
          int zpval = int(zpA[i * _ldsa + k / _kblock]);
          for (int ik = 0; ik < _kblock; ik += 4) {
            if (k + ik >= _k) {
              break;
            }
            for (int ikk = 0; ikk < 4; ikk++) {
              tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
            }
          }
          tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * float(scaleB[j + ij + k / _kblock * _ldsb]);
          tmpf -= zpval * scaleA[i * _ldsa + k / _kblock] * reduceB[j + ij + k / _kblock * _ldsb];
        }
        matC[i * ldc + j + ij] = tmpf;
      }
    }
  }
}
namespace ut {
class UT_GEMM_AVX2 {
 public:
  UT_GEMM_AVX2() {
    UT_START();
    CheckISA(AVX2);
    ut_24(4, 24, 3);
    ut_24(4, 48, 3);

    ut_48(1, 48, 3);
    ut_48(1, 144, 3);

#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  void ut_24(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx2<24>;
    Core gemm;
    if (n % Core::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    if (m > Core::Code::MTILE) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  void ut_48(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx2<48>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    if (m > Core::Code::MTILE) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, float* A, float* B, float* C) {
    LOG_T log;
    CORE_T core;
    for (size_t i = 0; i < batch; i++) {
      log.start();
      for (size_t im = 0; im < m; im += CORE_T::MTILE) {
        auto im_re = remainsize(im, m, CORE_T::MTILE);
        core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 4, k * 4, n * 4, 0,
                     cache, CacheSize);
      }
      if (log.stop()) {
        printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
      }
    }
  }
  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch), B(k * n * batch), C(m * n * batch, 0.f), RefC(m * n * batch, 0.f);
    using LOG = timer_statistics_logger<500>;
    using Core8 = gemm::SCoreRowNAvx2<8>;
    using Core16 = gemm::SCoreRowNAvx2<16>;
    using Core24 = gemm::SCoreRowNAvx2<24>;
    using Core24_B = gemm::SCoreRowNAvx2<24, 4>;
    using Core32 = gemm::SCoreRowNAvx2<32>;
    using Core48 = gemm::SCoreRowNAvx2<48>;

    benchmark<Core24, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
    benchmark<Core24_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
    benchmark<Core8, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
    benchmark<Core16, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data());
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX2 sUT_GEMM_AVX2;
#endif

class UT_GEMM_AVX512F {
 public:
  UT_GEMM_AVX512F() {
    UT_START();
    CheckISA(AVX512F);
    ut_32(4, 32, 3);
    ut_32(4, 64, 3);

    ut_48(1, 48, 3);
    ut_48(1, 144, 3);
    ut(1024, 144, 154);

#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  void ut_32(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx512f<32>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  void ut_48(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::SCoreRowNAvx512f<48>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    fill_buffer_randn(A.data(), A.size(), -0.5f, 0.5f);
    fill_buffer_randn(B.data(), B.size(), -0.5f, 0.5f);
    ref_fp32<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * 4, k * 4, n * 4, 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * 4, k * 4, n * 4, 0, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, float* A, float* B, float* C, float timems) {
    LOG_T log;
    CORE_T core;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 4, k * 4, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void ut(size_t m, size_t n, size_t k) {
    printf("%s %d %d %d\n", __FUNCTION__, int(m), int(n), int(k));
    avector<float> A(m * k), B(k * n), C(m * n, 0.f), RefC(m * n, 0.f);
    using Core48_B = gemm::SCoreRowNAvx512f<48, 8>;
    ref_fp32<Core48_B::NTILE>(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                              n * sizeof(C[0]), 0);
    Core48_B core;
    for (size_t im = 0; im < m; im += Core48_B::MTILE) {
      auto im_re = remainsize(im, m, Core48_B::MTILE);
      core.forward(A.data(), B.data(), C.data(), im_re, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0,
                   cache, CacheSize);
    }
    buffer_error(RefC.data(), C.data(), RefC.size());
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch), B(k * n * batch), C(m * n * batch, 0.f), RefC(m * n * batch, 0.f);
    using LOG = timer_statistics_logger<100>;
    using Core16 = gemm::SCoreRowNAvx512f<16>;
    using Core32 = gemm::SCoreRowNAvx512f<32>;
    using Core48 = gemm::SCoreRowNAvx512f<48>;
    using Core48_B = gemm::SCoreRowNAvx512f<48, 8>;
    using Core64 = gemm::SCoreRowNAvx512f<64>;

    float testtime = 500.f;
    benchmark<Core16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512F sUT_GEMM_AVX512F;
#endif

class UT_GEMM_AVX512VNNI {
 public:
  UT_GEMM_AVX512VNNI() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut<32, 0>(4, 64, 12);
    ut<32, 12>(4, 64, 12);

    ut<48, 0>(4, 96, 12);
    ut<48, 8>(4, 96, 12);

#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx512vnni<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, uint8_t* A, int8_t* B, int32_t* C, float timems) {
    LOG_T log;
    CORE_T core;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 1, k * 1, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<uint8_t> A(m * k * batch);
    avector<int8_t> B(k * n * batch);
    avector<int> C(m * n * batch, 0), RefC(m * n * batch, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    using LOG = timer_statistics_logger<100>;
    using Core16 = gemm::ICoreRowNAvx512vnni<16>;
    using Core32 = gemm::ICoreRowNAvx512vnni<32>;
    using Core48 = gemm::ICoreRowNAvx512vnni<48>;
    using Core48_B = gemm::ICoreRowNAvx512vnni<48, 8>;
    using Core64 = gemm::ICoreRowNAvx512vnni<64>;
    float testtime = 500.f;
    benchmark<Core16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512VNNI sUT_GEMM_AVX512VNNI;
#endif

class UT_GEMM_AVX512VNNI_KBLOCK {
 public:
  UT_GEMM_AVX512VNNI_KBLOCK() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut<48, 4>(4, 96, 36, 36);
    ut<48, 4>(4, 144, 128, 32);
    ut<48, 4>(4, 144, 128, 128);
    ut<48, 4>(4, 144, 256, 128);
    ut_splitblock<48, 4>(4, 144, 128, 128, 64);
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k, int kblock) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx512vnniKBlock<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<float> C(m * n, 0), RefC(m * n, 0);
    int blk_num = utils::updiv(k, kblock);
    avector<float> scaleA(blk_num * m), scaleB(blk_num * n), reduceB(blk_num * n, 0.f);
    avector<uint8_t> zpA(m * blk_num);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(zpA.data(), zpA.size(), (uint8_t)0, (uint8_t)0);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.003f, 0.005f);
    fill_buffer_randn(scaleB.data(), scaleB.size(), 0.003f, 0.005f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i / kblock * n + j] += B[i * n + j];
      }
    }
    for (size_t i = 0; i < blk_num; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i * n + j] *= scaleB[i * n + j];
      }
    }

    ref_kblock_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), zpA.data(), scaleA.data(), blk_num,
                                       scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]),
                                       k * sizeof(B[0]), n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), zpA.data(), scaleA.data(), blk_num, scaleB.data(), reduceB.data(), n, m,
                 n, k, kblock, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, 1.f, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }

  template <int NTile, int MTile>
  void ut_splitblock(int m, int n, int k, int kblock, int kstep) {
    assert(k == kblock);  // for large kblock case
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvx512vnniKBlock<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<float> C(m * n, 0), RefC(m * n, 0);
    int blk_num = utils::updiv(k, kblock);
    avector<float> scaleA(blk_num * m), scaleB(blk_num * n), reduceB(blk_num * n, 0.f);
    avector<uint8_t> zpA(m * blk_num);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(zpA.data(), zpA.size(), (uint8_t)0, (uint8_t)0);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.003f, 0.005f);
    fill_buffer_randn(scaleB.data(), scaleB.size(), 0.003f, 0.005f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i / kblock * n + j] += B[i * n + j];
      }
    }
    for (size_t i = 0; i < blk_num; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i * n + j] *= scaleB[i * n + j];
      }
    }

    ref_kblock_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), zpA.data(), scaleA.data(), blk_num,
                                       scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]),
                                       k * sizeof(B[0]), n * sizeof(C[0]), 0);
    for (size_t i = 0; i < k; i += kstep) {
      auto k_re = remainsize(i, k, kstep);
      gemm.forward(A.data() + i, B.data() + i * Core::Code::NTILE, C.data(), zpA.data(), scaleA.data(), blk_num,
                   scaleB.data(), reduceB.data(), n, m, n, k_re, k_re, k * sizeof(A[0]), k * sizeof(B[0]),
                   n * sizeof(C[0]), i, k_re / float(k), cache, CacheSize);
    }

    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512VNNI_KBLOCK sUT_GEMM_AVX512VNNI_KBLOCK;
#endif

class UT_GEMM_AVXVNNI_KBLOCK {
 public:
  UT_GEMM_AVXVNNI_KBLOCK() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut<48, 1>(1, 96, 36, 36);
    ut<48, 1>(1, 144, 128, 32);
    ut<48, 1>(1, 144, 128, 128);
    ut<48, 1>(1, 144, 256, 128);
    ut<24, 2>(2, 96, 36, 36);
    ut<24, 2>(2, 144, 128, 32);
    ut<24, 2>(2, 144, 128, 128);
    ut<24, 2>(2, 144, 256, 128);
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k, int kblock) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvxvnniKBlock<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<float> C(m * n, 0), RefC(m * n, 0);
    int blk_num = utils::updiv(k, kblock);
    avector<float> scaleA(blk_num * m), scaleB(blk_num * n), reduceB(blk_num * n, 0.f);
    avector<uint8_t> zpA(m * blk_num);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(zpA.data(), zpA.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.003f, 0.005f);
    fill_buffer_randn(scaleB.data(), scaleB.size(), 0.003f, 0.005f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i / kblock * n + j] += B[i * n + j];
      }
    }
    for (size_t i = 0; i < blk_num; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i * n + j] *= scaleB[i * n + j];
      }
    }

    ref_kblock_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), zpA.data(), scaleA.data(), blk_num,
                                       scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]),
                                       k * sizeof(B[0]), n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), zpA.data(), scaleA.data(), blk_num, scaleB.data(), reduceB.data(), n, m,
                 n, k, kblock, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, 1.f, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVXVNNI_KBLOCK sUT_GEMM_AVXVNNI_KBLOCK;
#endif

class UT_GEMM_AMXINT8_KBLOCK {
 public:
  UT_GEMM_AMXINT8_KBLOCK() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut_splitblock<48, 16>(16, 144, 128, 64);
    ut_splitblock<48, 16>(16, 144, 128, 128);
    ut_splitblock<48, 16>(16, 144, 256, 128);
    ut_splitblock<48, 16>(16, 144, 256, 128);
  }

  template <int NTile, int MTile>
  void ut_splitblock(int m, int n, int k, int kblock) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAmxint8KBlock<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<float> C(m * n, 0), RefC(m * n, 0);
    int blk_num = utils::updiv(k, kblock);
    avector<float> scaleA(blk_num * m), scaleB(blk_num * n), reduceB(blk_num * n, 0.f);
    avector<uint8_t> zpA(m * blk_num);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(zpA.data(), zpA.size(), (uint8_t)0, (uint8_t)0);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    fill_buffer_randn(scaleA.data(), scaleA.size(), 0.003f, 0.005f);
    fill_buffer_randn(scaleB.data(), scaleB.size(), 0.003f, 0.005f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i / kblock * n + j] += B[i * n + j];
      }
    }
    for (size_t i = 0; i < blk_num; i++) {
      for (size_t j = 0; j < n; j++) {
        reduceB[i * n + j] *= scaleB[i * n + j];
      }
    }

    ref_kblock_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), zpA.data(), scaleA.data(), blk_num,
                                       scaleB.data(), reduceB.data(), n, m, n, k, kblock, k * sizeof(A[0]),
                                       k * sizeof(B[0]), n * sizeof(C[0]), 0);
    gemm.configure(16, 16, 16);
    gemm.forward(A.data(), B.data(), C.data(), zpA.data(), scaleA.data(), blk_num, scaleB.data(), reduceB.data(), n, m,
                 n, k, kblock, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, 1.f, cache, CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AMXINT8_KBLOCK sUT_GEMM_AMXINT8_KBLOCK;
#endif

class UT_GEMM_AVXVNNI {
 public:
  UT_GEMM_AVXVNNI() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut<24>(4, 48, 12);

    ut<48>(2, 96, 12);
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  template <int NTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAvxvnni<NTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(m * n, 0), RefC(m * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);

    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), RefC.size(), 1);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, uint8_t* A, int8_t* B, int32_t* C, float timems) {
    LOG_T log;
    CORE_T core;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 1, k * 1, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<uint8_t> A(m * k * batch);
    avector<int8_t> B(k * n * batch);
    avector<int> C(m * n * batch, 0), RefC(m * n * batch, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    using LOG = timer_statistics_logger<100>;
    utils::timer<std::chrono::milliseconds> tm;
    using Core16 = gemm::ICoreRowNAvxvnni<16>;
    using Core24 = gemm::ICoreRowNAvxvnni<24>;
    using Core24_B = gemm::ICoreRowNAvxvnni<24, 4>;
    using Core32 = gemm::ICoreRowNAvxvnni<32>;
    using Core48 = gemm::ICoreRowNAvxvnni<48>;
    float testtime = 500.f;
    benchmark<Core16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core24, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core24_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVXVNNI sUT_GEMM_AVXVNNI;
#endif

class UT_GEMM_AVX512FP16 {
 public:
  UT_GEMM_AVX512FP16() {
    UT_START();
    CheckISA(AVX512_FP16);
    ut<32, 0>(4, 64, 3);
    ut<64, 0>(4, 128, 3);

#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAvx512fp16<NTILE, MTILE>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::fp16> matAbf16(m * k), matBbf16(k * n), matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int reordered_bstride = k * 2;
    ref_fp16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, reordered_bstride, n * 2, 0);
    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(fp16), k * sizeof(fp16),
                 n * sizeof(fp16), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), fp16(0.00001f * k));
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, utils::fp16* A, utils::fp16* B, utils::fp16* C, float timems) {
    LOG_T log;
    CORE_T core;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k, k, n, 0, cache,
                       CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<utils::fp16> A(m * k * batch);
    avector<utils::fp16> B(k * n * batch);
    avector<utils::fp16> C(m * n * batch, utils::fp16(0.f)), RefC(m * n * batch, utils::fp16(0.f));
    fill_buffer_randn(A.data(), A.size(), (utils::fp16)-0.5f, (utils::fp16)0.5f);
    fill_buffer_randn(B.data(), B.size(), (utils::fp16)-0.5f, (utils::fp16)0.5f);
    using LOG = timer_statistics_logger<100>;
    using Core32 = gemm::HCoreRowNAvx512fp16<32>;
    using Core64 = gemm::HCoreRowNAvx512fp16<64>;
    using Core96 = gemm::HCoreRowNAvx512fp16<96>;
    using Core96_B = gemm::HCoreRowNAvx512fp16<96, 8>;
    using Core128 = gemm::HCoreRowNAvx512fp16<128>;

    float testtime = 500.f;

    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core96, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core96_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core128, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512FP16 sUT_GEMM_AVX512FP16;
#endif

class UT_GEMM_AVX512BF16 {
 public:
  UT_GEMM_AVX512BF16() {
    UT_START();
    CheckISA(AVX512_BF16);
    ut<48, 0>(4, 96, 6);
    ut<48, 8>(4, 96, 6);
    ut<64, 0>(4, 128, 6);

#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(388, 192, 512, 1024);
    benchmark_all(512, 192, 768, 1024);
    benchmark_all(512, 384, 512, 1024);
#endif
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAvx512bf16<NTILE, MTILE>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    avector<float> matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    ref_bf16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, k * 2, n * 4, 0);
    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(bf16), k * sizeof(bf16),
                 n * sizeof(float), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, utils::bf16* A, utils::bf16* B, float* C, float timems) {
    LOG_T log;
    CORE_T core;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 2, k * 2, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<utils::bf16> A(m * k * batch);
    avector<utils::bf16> B(k * n * batch);
    avector<float> C(m * n * batch, utils::bf16(0.f)), RefC(m * n * batch, utils::bf16(0.f));
    fill_buffer_randn(A.data(), A.size(), (utils::bf16)-0.5f, (utils::bf16)0.5f);
    fill_buffer_randn(B.data(), B.size(), (utils::bf16)-0.5f, (utils::bf16)0.5f);
    using LOG = timer_statistics_logger<100>;
    using Core32 = gemm::HCoreRowNAvx512bf16<32>;
    using Core48 = gemm::HCoreRowNAvx512bf16<48>;
    using Core48_B = gemm::HCoreRowNAvx512bf16<48, 8>;
    using Core64 = gemm::HCoreRowNAvx512bf16<64>;

    float testtime = 500.f;
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48_B, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512BF16 sUT_GEMM_AVX512BF16;
#endif

class UT_GEMM_AMXBF16 {
 public:
  UT_GEMM_AMXBF16() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut<32, 32>(32, 32, 64);
    ut<32, 32>(4, 96, 96);
    ut<48, 0>(4, 96, 96);
    ut<64, 16>(4, 128, 96);
#ifdef JBLAS_UT_BENCHMARK
    benchmakr_all(384, 192, 512, 1024);
    benchmakr_all(384, 192, 768, 1024);
    benchmakr_all(768, 384, 512, 1024);
#endif
  }

  template <int NTILE, int MTILE>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::HCoreRowNAmxbf16<NTILE, MTILE>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }

    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    avector<float> matC(Core::Code::MTILE * n), refC(Core::Code::MTILE * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    ref_bf16<Core::NTILE>(matAbf16.data(), matBbf16.data(), refC.data(), m, n, k, k * 2, k * 2, n * 4, 0);
    gemm.configure(m, n, k);

    gemm.forward(matAbf16.data(), matBbf16.data(), matC.data(), m, n, k, k * sizeof(bf16), k * sizeof(bf16),
                 n * sizeof(float), 0, cache, CacheSize);
    ut::buffer_error(refC.data(), matC.data(), m * n, 0.001f);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, utils::bf16* A, utils::bf16* B, float* C, float timems) {
    LOG_T log;
    CORE_T core;
    core.configure(m, n, k);

    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 2, k * 2, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmakr_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<utils::bf16> A(m * k * batch);
    avector<utils::bf16> B(k * n * batch);
    avector<float> C(m * n * batch, utils::bf16(0.f)), RefC(m * n * batch, utils::bf16(0.f));
    fill_buffer_randn(A.data(), A.size(), (utils::bf16)-0.5f, (utils::bf16)0.5f);
    fill_buffer_randn(B.data(), B.size(), (utils::bf16)-0.5f, (utils::bf16)0.5f);
    using LOG = timer_statistics_logger<100>;
    using Core32 = gemm::HCoreRowNAmxbf16<32, 32>;
    using Core48 = gemm::HCoreRowNAmxbf16<48, 16>;
    using Core64 = gemm::HCoreRowNAmxbf16<64, 16>;
    float testtime = 500.f;
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AMXBF16 sUT_GEMM_AMXBF16;
#endif

class UT_GEMM_AMXINT8 {
 public:
  UT_GEMM_AMXINT8() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut<32, 32>(32, 64, 64 * 3);
    ut<48, 16>(16, 96, 64 * 3);
    ut<32, 32>(4, 64, 64 * 3);
    ut<32, 32>(20, 64, 64 * 3);
    ut<32, 32>(32, 64, 64 * 3);

    ut<64, 16>(16, 128, 64 * 3);
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(384, 192, 512, 1024);
    benchmark_all(384, 192, 768, 1024);
    benchmark_all(768, 384, 512, 1024);
#endif
  }

  template <int NTile, int MTile>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d\n", m, n, k);
    using Core = gemm::ICoreRowNAmxint8<NTile, MTile>;
    Core gemm;
    if (n % Core::Code::NTILE != 0) {
      return;
    }
    if (k % Core::Code::KTILE != 0) {
      return;
    }
    avector<uint8_t> A(m * k);
    avector<int8_t> B(k * n);
    avector<int> C(Core::Code::MTILE * n, 0), RefC(Core::Code::MTILE * n, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    ref_int8<Core::Code::NTILE>(A.data(), B.data(), RefC.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]),
                                n * sizeof(C[0]), 0);
    gemm.configure(m, n, k);
    gemm.forward(A.data(), B.data(), C.data(), m, n, k, k * sizeof(A[0]), k * sizeof(B[0]), n * sizeof(C[0]), 0, cache,
                 CacheSize);
    ut::buffer_error(RefC.data(), C.data(), m * n, 1);
  }

  template <typename CORE_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, uint8_t* A, int8_t* B, int32_t* C, float timems) {
    LOG_T log;
    CORE_T core;
    core.configure(m, n, k);
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        for (int im = 0; im < m; im += CORE_T::MTILE) {
          auto im_re = remainsize(im, m, CORE_T::MTILE);
          core.forward(A + i * m * k + im * k, B + i * n * k, C + i * m * n + im * n, im_re, n, k, k * 1, k * 1, n * 4,
                       0, cache, CacheSize);
        }
        if (log.stop()) {
          printf("tar%d %d %s\n", CORE_T::NTILE, CORE_T::MTILE, log.get_log_str());
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<uint8_t> A(m * k * batch);
    avector<int8_t> B(k * n * batch);
    avector<int> C(m * n * batch, 0), RefC(m * n * batch, 0);
    fill_buffer_randn(A.data(), A.size(), (uint8_t)0, (uint8_t)255);
    fill_buffer_randn(B.data(), B.size(), (int8_t)-127, (int8_t)127);
    using LOG = timer_statistics_logger<100>;
    using Core32 = gemm::ICoreRowNAmxint8<32, 32>;
    using Core48 = gemm::ICoreRowNAmxint8<48, 16>;
    using Core64 = gemm::ICoreRowNAmxint8<64, 16>;

    float testtime = 500.f;
    benchmark<Core32, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core48, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
    benchmark<Core64, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime);
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AMXINT8 sUT_GEMM_AMXINT8;
#endif
}  // namespace ut
}  // namespace bestla
