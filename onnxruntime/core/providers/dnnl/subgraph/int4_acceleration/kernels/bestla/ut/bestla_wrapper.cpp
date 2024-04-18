#include "bestla_wrapper.h"
#include "bestla_ut.h"
namespace bestla {
using namespace utils;
namespace ut {
class UT_Fp32Fp32 {
 public:
  UT_Fp32Fp32() {
    UT_START();
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(1, 4096, 4096, 32);
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
#endif  // JBLAS_UT_BENCHMARK

    CheckISA(AVX2);
    ut<sAVX2>(1, 1, 1);
    ut<sAVX2>(8, 48, 2);
    ut<sAVX2>(8, 4096, 4096);
    ut<sAVX2>(384, 768, 768);
    ut<sAVX2>(1024, 1024, 1024);
    ut<sAVX2>(1024, 1536, 1536);

    CheckISA(AVX512F);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(384, 768, 768);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1024, 1024, 1024);
    ut<sAVX512F>(1, 1, 1);
    ut<sAVX512F>(8, 48, 2);
    ut<sAVX512F>(8, 4096, 4096);
    ut<sAVX512F>(384, 768, 768);
    ut<sAVX512F>(1024, 1024, 1024);
    ut<sAVX512F>(1024, 1536, 1536);
  }
  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    fill_buffer_randn(matB.data(), matB.size(), -0.5f, 0.5f);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matB.data(), n, &packw}, &DefaultThreading);
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matA.data(), k}, {matB.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    ut::buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("%s %s Flops:%.3f PerCoreFlops:%.3f\n ", corestr, log.get_log_str(), flops, flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch, 0), RefC(m * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;

    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512F()) {
      benchmark<sAVX512F, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<sAVX512F, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 56);
    }
    if (_cd->AVX2()) {
      benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 56);
    }
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_Fp32Fp32 sUT_Fp32Fp32;
#endif

class UT_U8S8S32 {
 public:
  UT_U8S8S32() {
    UT_START();
    GetCPUDevice();
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
#endif
    if (_cd->AVX512_VNNI()) {
      ut<sAVX512_VNNI>(4, 48, 4);
      ut<sAVX512_VNNI>(1, 1, 1);
      ut<sAVX512_VNNI>(8, 48, 2);
      ut<sAVX512_VNNI>(8, 4096, 4096);
      ut<sAVX512_VNNI>(384, 768, 768);
      ut<sAVX512_VNNI>(1024, 1024, 1024);
      ut<sAVX512_VNNI>(1024, 1536, 1536);
    }
    if (_cd->AVX_VNNI()) {
      ut<sAVX_VNNI>(1, 1, 1);
      ut<sAVX_VNNI>(8, 48, 2);
      ut<sAVX_VNNI>(8, 4096, 4096);
      ut<sAVX_VNNI>(384, 768, 768);
      ut<sAVX_VNNI>(1024, 1024, 1024);
      ut<sAVX_VNNI>(1024, 1536, 1536);
    }
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut<sAMX_INT8_US>(1, 1, 1);
      ut<sAMX_INT8_US>(8, 48, 2);
      ut<sAMX_INT8_US>(8, 4096, 4096);
      ut<sAMX_INT8_US>(384, 768, 768);
      ut<sAMX_INT8_US>(1024, 1024, 1024);
      ut<sAMX_INT8_US>(1024, 1536, 1536);
    }
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n);
    avector<uint8_t> matAu8(m * k), zpAu8(m);
    avector<int8_t> matBs8(k * n);
    avector<float> scaleAf32(m), scaleBf32(n);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    fill_buffer_randn(scaleBf32.data(), scaleBf32.size(), 0.001f, 0.005f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] = (float(matAu8[i * k + j]) - zpAu8[i]) * scaleAf32[i];
      }
    }
    avector<float> reduceB(n, 0);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(matBs8[i * n + j])) * scaleBf32[j];
        reduceB[j] += matBf32[i * n + j];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                                 prologue_b::gemm::WeightPack, epilogue::gemm::ZpDequantInt32ToFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matBs8.data(), n, &packw}, &DefaultThreading);
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{
        gp,
        {matAu8.data(), k},
        {matBs8.data(), n, &packw},
        {matC.data(), n, 1, scaleAf32.data(), scaleBf32.data(), zpAu8.data(), reduceB.data()}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }

  using AType = uint8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackInt32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      benchmark<gemm::ICoreRowNAmxint8<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::ICoreRowNAmxint8<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::ICoreRowNAmxint8<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
    if (_cd->AVX512_VNNI()) {
      benchmark<gemm::ICoreRowNAvx512vnni<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
    if (_cd->AVX_VNNI()) {
      benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_U8S8S32 sUT_U8S8S32;
#endif

class UT_S8S8S32 {
 public:
  UT_S8S8S32() {
    UT_START();
    GetCPUDevice();
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
#endif
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut<sAMX_INT8_SS>(1, 1, 1);
      ut<sAMX_INT8_SS>(8, 48, 2);
      ut<sAMX_INT8_SS>(8, 4096, 4096);
      ut<sAMX_INT8_SS>(384, 768, 768);
      ut<sAMX_INT8_SS>(1024, 1024, 1024);
      ut<sAMX_INT8_SS>(1024, 1536, 1536);
    }
  }
  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n);
    avector<int8_t> matAu8(m * k);
    avector<int8_t> matBs8(k * n);
    avector<float> scaleAf32(m), scaleBf32(n);
    fill_buffer_randn(matAu8.data(), matAu8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    fill_buffer_randn(scaleBf32.data(), scaleBf32.size(), 0.001f, 0.005f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] = (float(matAu8[i * k + j])) * scaleAf32[i];
      }
    }
    avector<float> reduceB(n, 0);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(matBs8[i * n + j])) * scaleBf32[j];
        reduceB[j] += matBf32[i * n + j];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                                 prologue_b::gemm::WeightPack, epilogue::gemm::DequantInt32ToFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matBs8.data(), n, &packw}, &DefaultThreading);
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{
        gp, {matAu8.data(), k}, {matBs8.data(), n, &packw}, {matC.data(), n, 1, scaleAf32.data(), scaleBf32.data()}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }

  using AType = int8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackInt32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(AType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      benchmark<gemm::ICoreRowNAmxint8SS<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::ICoreRowNAmxint8SS<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::ICoreRowNAmxint8SS<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_S8S8S32 sUT_S8S8S32;
#endif

class UT_Bf16Bf16Fp32 {
 public:
  UT_Bf16Bf16Fp32() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
#endif
    ut<sAMX_BF16>(1, 1, 1);
    ut<sAMX_BF16>(8, 48, 2);
    ut<sAMX_BF16>(8, 4096, 4096);
    ut<sAMX_BF16>(384, 768, 768);
    ut<sAMX_BF16>(1024, 1024, 1024);
    ut<sAMX_BF16>(1024, 1536, 1536);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case %s: %d %d %d core:%s\n", __FUNCTION__, m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    avector<float> matC(m * n), refC(m * n);
    launcher.mProB.packWeight(n, k, {matBbf16.data(), n, &packw}, &DefaultThreading);
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matAbf16.data(), k}, {matBbf16.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    buffer_error(refC.data(), matC.data(), refC.size(), 0.05f);
  }

  using AType = utils::bf16;
  using BType = utils::bf16;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, BType(-0.5f), BType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_BF16()) {
      request_perm_xtile_data();
      benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
      benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_Bf16Bf16Fp32 sUT_Bf16Bf16Fp32;
#endif

class UT_Fp16Fp16Fp16 {
 public:
  UT_Fp16Fp16Fp16() {
    UT_START();
    CheckISA(AVX512_FP16);
#ifdef JBLAS_UT_BENCHMARK
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
#endif
    ut<sAVX512_FP16>(1, 1, 1);
    ut<sAVX512_FP16>(8, 48, 2);
    ut<sAVX512_FP16>(8, 4096, 4096);
    ut<sAVX512_FP16>(384, 768, 768);
    ut<sAVX512_FP16>(1024, 1024, 1024);
    ut<sAVX512_FP16>(1024, 1536, 1536);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case %s: %d %d %d core:%s\n", __FUNCTION__, m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp16>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    avector<utils::fp16> matAbf16(m * k), matBbf16(k * n), matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    launcher.mProB.packWeight(n, k, {matBbf16.data(), n, &packw}, &DefaultThreading);
    gemmref_fp16fp16fp16(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matAbf16.data(), k}, {matBbf16.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    buffer_error(refC.data(), matC.data(), refC.size(), utils::fp16(0.0002f * k));
  }

  using AType = utils::fp16;
  using BType = utils::fp16;
  using CType = utils::fp16;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp16>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, AType(-0.5f), AType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 56);
      benchmark<gemm::HCoreRowNAvx512fp16<64, 12>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 56);
      benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, 48);
    }
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_Fp16Fp16Fp16 sUT_Fp16Fp16Fp16;
#endif
}  // namespace ut
}  // namespace bestla
