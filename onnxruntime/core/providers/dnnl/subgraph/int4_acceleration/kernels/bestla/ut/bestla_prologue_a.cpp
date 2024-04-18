#include "bestla_prologue_a.h"
#include "bestla_ut.h"
#include "kernel_avx512f.h"

namespace bestla {
using namespace utils;
namespace ut {
class UT_ActivationBase {
 public:
  UT_ActivationBase() {
    UT_START();
    CheckISA(AVX512F);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(8, 3, 128);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(8, 48, 128);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(8, 3, 128);
  }
  template <typename _T>
  void ut(int m, int k, int lda) {
    int kpad = padto(k, _T::KTILE);
    using BType = typename _T::BType;
    printf("Test Case %dB NTile%d: %d %d %d %d \n", int(sizeof(BType)), _T::NTILE, m, k, lda, kpad);
    std::vector<BType> src(m * lda);
    std::vector<BType> dst(m * kpad), dstref(m * kpad);
    for (int i = 0; i < src.size(); i++) {
      src[i] = static_cast<BType>(i);
    }
    prologue_a::gemm::ActivationBase<_T, BTLA_ISA::NoSIMD> reorderref;
    prologue_a::gemm::ActivationBase<_T, BTLA_ISA::AVX512F> reorderavx512;
    auto dstrefptr = dstref.data();
    auto dstptr = dst.data();
    int dststride = 0;
    reorderref.getActivation(&dstrefptr, &dststride, {src.data(), lda}, m, k, 0, 0, cache, CacheSize);
    GetCPUDevice();
    if (_cd->AVX512F()) {
      reorderavx512.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0, cache, CacheSize);
      ut::buffer_error(dst.data(), dstref.data(), dst.size());
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_A
static UT_ActivationBase sUT_ActivationBase;
#endif

class UT_ActivationConverter {
 public:
  UT_ActivationConverter() {
    UT_START();
    CheckISA(AMX_BF16);
    ut<float, gemm::HCoreRowNAmxbf16<64, 16>>(8, 3, 128);
    ut<float, gemm::HCoreRowNAmxbf16<64, 16>>(8, 48, 128);
    ut<utils::bf16, gemm::SCoreRowNAvx512f<48, 8>>(8, 3, 128);
    ut<utils::bf16, gemm::SCoreRowNAvx512f<48, 8>>(8, 48, 128);
  }

  template <typename SRC_T, typename _T>
  void ut(int m, int k, int lda) {
    using SrcType = SRC_T;
    using AType = typename _T::AType;
    int kpad = padto(k, _T::KTILE);
    printf("Test Case %dB NTile%d: %d %d %d %d \n", int(sizeof(AType)), _T::NTILE, m, k, lda, kpad);
    std::vector<SrcType> src(m * lda);
    std::vector<AType> dst(m * kpad), dstref(m * kpad);
    for (int i = 0; i < src.size(); i++) {
      src[i] = static_cast<SrcType>(float(i));
    }
    prologue_a::gemm::ActivationConverter<_T, BTLA_ISA::NoSIMD, SRC_T> reorderref;
    prologue_a::gemm::ActivationConverter<_T, BTLA_ISA::AVX512F, SRC_T> reorderavx512;
    auto dstptr = dstref.data();
    int dststride = 0;
    auto ret = reorderref.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0, cache, CacheSize);
    assert(ret == BTLA_CODE::Success);
    dstptr = dst.data();
    ret = reorderavx512.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0, cache, CacheSize);
    assert(ret == BTLA_CODE::Success);
    ut::buffer_error(dst.data(), dstref.data(), dst.size(), AType{0});
    aligned_vector<SrcType> revert(dst.size());
    for (size_t i = 0; i < revert.size(); i++) {
      revert[i] = utils::cast<AType, SrcType>(dst[i]);
    }
    for (size_t i = 0; i < src.size(); i++) {
      auto tmp = utils::cast<SrcType, AType>(src[i]);
      src[i] = utils::cast<AType, SrcType>(tmp);
    }
    buffer_error_2d(src.data(), revert.data(), m, k, lda, kpad);
  }
};
#ifdef BTLA_UT_PROLOGUE_A
static UT_ActivationConverter sUT_ActivationConverter;
#endif

class UT_ActivationU8KBlockQuantize {
 public:
  UT_ActivationU8KBlockQuantize() {
    UT_START();
    CheckISA(AVX512F);
    ut<gemm::ICoreRowNAmxint8<48, 16>>(15, 63, 64, 2);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 4096, 4096, 32, true);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 4096, 4096, 4096, true);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 4096, 4096, 32, true);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 4096, 4096, 4096, true);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 4096, 4096, 4096);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 4096, 4096, 128);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 4096, 4096, 32);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(2, 11040, 11040, 32);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(1024, 4096, 4096, 32);
    ut<gemm::ICoreRowNAvx512vnni<48, 8>>(1024, 11040, 11040, 32);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 4096, 4096, 4096);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 4096, 4096, 128);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 4096, 4096, 32);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(2, 11040, 11040, 32);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(1024, 4096, 4096, 32);
    ut<gemm::ICoreRowNAvxvnni<48, 2>>(1024, 11040, 11040, 32);
  }
  template <typename _T>
  void ut(int m, int k, int lda, int kblock, bool hasreduce = false) {
    int kpad = padto(k, _T::KTILE);
    printf("Test Case core:%s: %d %d %d %d %d reduce:%d\n", gemm::CoreAttr::to_str(_T::ID), m, k, lda, kblock, kpad,
           hasreduce);
    int kcount = updiv(kpad, kblock);
    utils::aligned_vector<float> raw(m * lda), scales(m * kcount);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.5f, 0.5f);
    utils::aligned_vector<uint8_t> q, zp;
    q.resize(m * lda);
    zp.resize(m * kcount);
    avector<float> reduce(m * kcount);

    kernel::ref::quantize_fp_u8_colblock(m, k, raw.data(), lda, q.data(), lda, scales.data(), kcount, zp.data(), kblock,
                                         hasreduce ? reduce.data() : nullptr);
    prologue_a::gemm::ActivationF32KBlockQuantize<_T, _T::ISA> actA;
    auto quanAct = actA.createStorage(m, k, kblock, hasreduce);
    avector<int8_t> bufA(quanAct.mSize);
    quanAct.assign(bufA.data());
    actA.quantize({raw.data(), lda, &quanAct}, m, k, &DefaultThreading);

    ut::buffer_error(q.data(), quanAct.template APtr<uint8_t>(), q.size(), uint8_t(1));
    ut::buffer_error(zp.data(), quanAct.template ZPtr<uint8_t>(), zp.size(), uint8_t(1));
    if (hasreduce) {
      avector<float> redref(reduce.size(), 0.f), redqref(reduce.size(), 0.f);
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          redref[i * kcount + j / kblock] += raw[i * k + j];
          redqref[i * kcount + j / kblock] +=
              (float(q[i * k + j]) - zp[i * kcount + j / kblock]) * scales[i * kcount + j / kblock];
        }
      }
      buffer_error(redref.data(), reduce.data(), reduce.size(), INT8_ERR);
      buffer_error(redqref.data(), reduce.data(), reduce.size(), FP32_ERR);
      buffer_error(reduce.data(), quanAct.template RPtr<float>(), reduce.size(), FP32_ERR);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_A
static UT_ActivationU8KBlockQuantize sUT_ActivationU8KBlockQuantize;
#endif

class UT_ActivationS8KBlockQuantize {
 public:
  UT_ActivationS8KBlockQuantize() {
    UT_START();
    CheckISA(AVX512F);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(15, 63, 2, true);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2, 4096, 32, true);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2, 4096, 4096, true);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2, 11040, 32);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2, 4096, 128);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(2, 4096, 32);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(1024, 4096, 32);
    ut<gemm::ICoreRowNAmxint8SS<48, 16>>(1024, 11040, 32);
  }

  template <typename _T>
  void ut(int m, int k, int kblock, bool hasreduce = false) {
    int kpad = padto(k, _T::KTILE);
    int lda = kpad;
    printf("Test Case NTile%d: %d %d %d %d %d reduce:%d\n", _T::KTILE, m, k, lda, kblock, kpad, hasreduce);
    int kcount = updiv(kpad, kblock);
    utils::aligned_vector<float> raw(m * k), scales(m * kcount);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.1f, 0.1f);
    utils::aligned_vector<int8_t> q;
    avector<float> reduce(m * kcount);
    q.resize(m * lda);
    kernel::ref::quantize_fp_s8_colblock(m, k, raw.data(), k, q.data(), lda, scales.data(), kcount, kblock,
                                         hasreduce ? reduce.data() : nullptr);
    prologue_a::gemm::ActivationF32KBlockQuantize<_T, BTLA_ISA::AVX512F> actA;
    auto quanAct = actA.createStorage(m, k, kblock, hasreduce);
    avector<int8_t> bufA(quanAct.mSize);
    quanAct.assign(bufA.data());
    actA.quantize({raw.data(), k, &quanAct}, m, k, &DefaultThreading);
    ut::buffer_error(q.data(), quanAct.template APtr<int8_t>(), q.size(), int8_t(1));
    if (hasreduce) {
      avector<float> redref(reduce.size(), 0.f), redqref(reduce.size(), 0.f);
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          redref[i * kcount + j / kblock] += raw[i * k + j];
          redqref[i * kcount + j / kblock] += (float(q[i * lda + j])) * scales[i * kcount + j / kblock];
        }
      }
      buffer_error(redref.data(), reduce.data(), reduce.size(), INT8_ERR);
      buffer_error(redqref.data(), reduce.data(), reduce.size(), FP32_ERR);
      buffer_error(reduce.data(), quanAct.template RPtr<float>(), reduce.size(), FP32_ERR);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_A
static UT_ActivationS8KBlockQuantize sUT_ActivationS8KBlockQuantize;
#endif

class UT_ShuffleActivationKblock {
 public:
  UT_ShuffleActivationKblock() {
    UT_START();
    ut<float, gemm::SCoreRowNAvx2<48, 2>>(15, 63);
    ut<bf16, gemm::SCoreRowNAvx2<48, 2>>(15, 63);
    ut<float, gemm::HCoreRowNAmxbf16<64, 16>>(15, 63);
    ut<bf16, gemm::HCoreRowNAmxbf16<64, 16>>(15, 63);
    dynamic_ut(15, 63, 2, true);
  }
  template <typename _SRC_T, class GC>
  void ut(int m, int k) {
    int kpad = padto(k, GC::KTILE);
    using BType = typename GC::BType;
    printf("Test Case %s %dB NTile%d: %d %d %d \n", __FUNCTION__, int(sizeof(BType)), GC::NTILE, m, k, kpad);
    std::vector<_SRC_T> src(m * k);
    std::vector<BType> dst(m * kpad), dstref(m * kpad);
    std::vector<int> indices(k);
    for (int i = 0; i < indices.size(); i++) {
      indices[i] = i % 2 == 0 ? (i + 1) == indices.size() ? i : i + 1 : i - 1;
    }
    for (int i = 0; i < src.size(); i++) src[i] = static_cast<_SRC_T>(i);
    prologue_a::gemm::ShuffleActivationKBlockBase<GC, BTLA_ISA::NoSIMD, _SRC_T> kernel;
    auto dstrefptr = dstref.data();
    auto dstptr = dst.data();
    int dststride = 0;
    auto reordA = kernel.createReorderStorage(m, k, 32);
    avector<int8_t> bufA(reordA.mSize);
    reordA.assign(bufA.data());
    kernel.preprocess({src.data(), k, nullptr, indices.data(), &reordA}, m, k, 32, &DefaultThreading);

    kernel.getActivation(&dstptr, &dststride, {src.data(), k, nullptr, indices.data(), &reordA}, m, kpad, 0, 0, cache,
                         CacheSize);
    for (int i = 0; i < m; i++) {
      int j = 0;
      for (; j < k; j++) dstrefptr[i * kpad + j] = static_cast<BType>(src[i * k + indices[j]]);
      for (; j < kpad; j++) dstrefptr[i * kpad + j] = static_cast<BType>(0);
    }
    buffer_error(dstrefptr, dstptr, dst.size());
  }

  void dynamic_ut(int m, int k, int kblock, bool hasreduce = false) {
    using GC = gemm::ICoreRowNAmxint8SS<48, 16>;
    int kpad = padto(k, GC::KTILE);
    int lda = kpad;
    printf("Test Case %s NTile%d: %d %d %d %d %d reduce:%d\n", __FUNCTION__, GC::KTILE, m, k, lda, kblock, kpad,
           hasreduce);
    int kcount = updiv(kpad, kblock);
    utils::aligned_vector<float> raw(m * k, 0), raw_cp(m * k, 0), scales(m * kcount);
    ut::fill_buffer_randn(raw_cp.data(), raw_cp.size(), -0.1f, 0.1f);
    utils::aligned_vector<int8_t> q;
    avector<float> reduce(m * kcount);
    q.resize(m * lda);
    std::vector<int> indices(k);
    for (int i = 0; i < indices.size(); i++) {
      indices[i] = i % 2 == 0 ? (i + 1) == indices.size() ? i : i + 1 : i - 1;
    }
    kernel::ref::shuffle_activation(raw_cp.data(), raw.data(), m, k, 0, 0, indices.data(), k, k);
    kernel::ref::quantize_fp_s8_colblock(m, k, raw.data(), k, q.data(), lda, scales.data(), kcount, kblock,
                                         hasreduce ? reduce.data() : nullptr);
    prologue_a::gemm::ShuffleActivationKBlockQuantize<GC, BTLA_ISA::NoSIMD, float> actA;
    auto quanAct = actA.createQuantStorage(m, k, kblock, hasreduce);
    auto reordAct = actA.createReorderStorage(m, k, kblock);
    avector<int8_t> bufA(quanAct.mSize + reordAct.mSize);
    quanAct.assign(bufA.data());
    reordAct.assign(bufA.data() + quanAct.mSize);
    actA.quantize({raw_cp.data(), k, &quanAct, indices.data(), &reordAct}, m, k, &DefaultThreading);
    ut::buffer_error(quanAct.template APtr<int8_t>(), q.data(), q.size(), int8_t(1));
    if (hasreduce) {
      avector<float> redref(reduce.size(), 0.f), redqref(reduce.size(), 0.f);
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          redref[i * kcount + j / kblock] += raw[i * k + j];
          redqref[i * kcount + j / kblock] += (float(q[i * lda + j])) * scales[i * kcount + j / kblock];
        }
      }
      buffer_error(redref.data(), reduce.data(), reduce.size(), INT8_ERR);
      buffer_error(redqref.data(), reduce.data(), reduce.size(), FP32_ERR);
      buffer_error(reduce.data(), quanAct.template RPtr<float>(), reduce.size(), FP32_ERR);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_A
static UT_ShuffleActivationKblock sUT_ShuffleActivationKblock;
#endif
}  // namespace ut
}  // namespace bestla
