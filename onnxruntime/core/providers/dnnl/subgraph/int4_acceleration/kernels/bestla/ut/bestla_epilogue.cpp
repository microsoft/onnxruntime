#include "bestla_epilogue.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
class UT_AccumulatorWriteBack {
 public:
  UT_AccumulatorWriteBack() {
    UT_START();
    CheckISA(AVX2);
    fp32ut<BTLA_ISA::AVX2>(127, 255, 0, 0, 127, 255);
    fp32ut<BTLA_ISA::AVX2>(101, 237, 10, 63, 30, 33);
    fp32ut_with_custom_gelu<BTLA_ISA::AVX2>(15, 15, 0, 0, 15, 15);
    fp32ut_with_custom_swish<BTLA_ISA::AVX2>(15, 15, 0, 0, 15, 15);
    bf16ut<BTLA_ISA::AVX2>(127, 255, 0, 0, 127, 255);
    bf16ut<BTLA_ISA::AVX2>(101, 237, 10, 63, 30, 33);
    bf16fp32ut<BTLA_ISA::AVX2>(101, 237, 10, 63, 30, 33);
    bf16fp32ut<BTLA_ISA::AVX2>(127, 255, 0, 0, 127, 255);
    CheckISA(AVX512F);
    fp32ut<BTLA_ISA::AVX512F>(127, 255, 0, 0, 127, 255);
    fp32ut<BTLA_ISA::AVX512F>(101, 237, 10, 63, 30, 33);
    bf16ut<BTLA_ISA::AVX512F>(127, 255, 0, 0, 127, 255);
    bf16ut<BTLA_ISA::AVX512F>(101, 237, 10, 63, 30, 33);
    fp32ut_with_custom_gelu<BTLA_ISA::AVX512F>(15, 15, 0, 0, 15, 15);
    fp32ut_with_custom_swish<BTLA_ISA::AVX512F>(15, 15, 0, 0, 15, 15);

    bf16fp32ut<BTLA_ISA::AVX512F>(101, 237, 10, 63, 30, 33);
    bf16fp32ut<BTLA_ISA::AVX512F>(127, 255, 0, 0, 127, 255);
  }
  template <BTLA_ISA _RT_ISA_T>
  void bf16fp32ut(int _M, int _N, int _M_offset, int _N_offset, int _cpy_M, int _cpy_N) {
    printf("Test Case %s %d %d %d %d %d %d\n", __FUNCTION__, _M, _N, _M_offset, _N_offset, _cpy_M, _cpy_N);
    std::vector<bf16> src(_M * _N);
    for (int i = 0; i < _M * _N; i++) src[i].fromfloat(i);
    std::vector<float> dstref(_M * _N, 0), dstker(_M * _N, 0);
    epilogue::gemm::AccumulatorWriteBackBf16Fp32<_RT_ISA_T> ker;
    epilogue::gemm::AccumulatorWriteBackBf16Fp32<BTLA_ISA::NoSIMD> kerref;

    kerref.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstref.data(), _N}, cache, CacheSize);
    ker.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstker.data(), _N}, cache, CacheSize);
    ut::buffer_error(dstref.data(), dstker.data(), dstref.size());
  }
  template <BTLA_ISA _RT_ISA_T>
  void bf16ut(int _M, int _N, int _M_offset, int _N_offset, int _cpy_M, int _cpy_N) {
    printf("Test Case %s %d %d %d %d %d %d\n", __FUNCTION__, _M, _N, _M_offset, _N_offset, _cpy_M, _cpy_N);
    std::vector<float> src(_M * _N);
    for (int i = 0; i < _M * _N; i++) src[i] = float(i);
    std::vector<uint16_t> dstref(_M * _N, 0), dstker(_M * _N, 0);
    epilogue::gemm::AccumulatorWriteBackFp32Bf16<_RT_ISA_T> ker;
    epilogue::gemm::AccumulatorWriteBackFp32Bf16<BTLA_ISA::NoSIMD> kerref;

    kerref.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {reinterpret_cast<bf16*>(dstref.data()), _N},
                   cache, CacheSize);
    ker.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {reinterpret_cast<bf16*>(dstker.data()), _N},
                cache, CacheSize);
    ut::buffer_error<uint16_t>(dstref.data(), dstker.data(), dstref.size());
  }
  template <BTLA_ISA _RT_ISA_T>
  void fp32ut(int _M, int _N, int _M_offset, int _N_offset, int _cpy_M, int _cpy_N) {
    printf("Test Case %s %d %d %d %d %d %d\n", __FUNCTION__, _M, _N, _M_offset, _N_offset, _cpy_M, _cpy_N);
    std::vector<float> src(_M * _N);
    for (int i = 0; i < _M * _N; i++) src[i] = float(i);
    std::vector<float> dstref(_M * _N, 0), dstker(_M * _N, 0);
    epilogue::gemm::AccumulatorWriteBackFp32<_RT_ISA_T> ker;
    epilogue::gemm::AccumulatorWriteBackFp32<BTLA_ISA::NoSIMD> kerref;

    kerref.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstref.data(), _N}, cache, CacheSize);
    ker.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstker.data(), _N}, cache, CacheSize);
    ut::buffer_error<float>(dstref.data(), dstker.data(), dstref.size());
  }
  template <BTLA_ISA _RT_ISA_T>
  void fp32ut_with_custom_gelu(int _M, int _N, int _M_offset, int _N_offset, int _cpy_M, int _cpy_N) {
    printf("Test Case %s %d %d %d %d %d %d\n", __FUNCTION__, _N, _M, _M_offset, _N_offset, _cpy_M, _cpy_N);
    std::vector<float> src(_M * _N);
    for (int i = 0; i < _M * _N; i++) src[i] = float(i);
    std::vector<float> dstref(_M * _N, 0), dstker(_M * _N, 0);
    epilogue::gemm::AccumulatorWriteBackWithGeluFp32<_RT_ISA_T> ker;
    ker.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstker.data(), _N}, cache, CacheSize);
    auto gelu = [&](float x) {
      return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
    };
    for (int i = 0; i < _M * _N; i++) src[i] = gelu(src[i]);
    ut::buffer_error<float>(src.data(), dstker.data(), dstker.size(), 0.000001f);
  }

  template <BTLA_ISA _RT_ISA_T>
  void fp32ut_with_custom_swish(int _M, int _N, int _M_offset, int _N_offset, int _cpy_M, int _cpy_N) {
    printf("Test Case %s %d %d %d %d %d %d\n", __FUNCTION__, _N, _M, _M_offset, _N_offset, _cpy_M, _cpy_N);
    std::vector<float> src(_M * _N);
    for (int i = 0; i < _M * _N; i++) src[i] = float(i);
    std::vector<float> dstref(_M * _N, 0), dstker(_M * _N, 0);
    float elt_const_v[] = {-1.0f};
    epilogue::gemm::AccumulatorWriteBackWithSwishFp32<_RT_ISA_T> ker;
    ker.forward(src.data(), _N, _M_offset, _N_offset, _cpy_M, _cpy_N, {dstker.data(), _N, elt_const_v}, cache,
                CacheSize);
    auto swish = [&](float x) { return x / (1 + exp(-x)); };
    for (int i = 0; i < _M * _N; i++) src[i] = swish(src[i]);
    ut::buffer_error<float>(src.data(), dstker.data(), dstker.size(), 0.2f);  // swish use low lprecision exp
  }
};

class UT_AlphaBetaProcessFp32 {
 public:
  UT_AlphaBetaProcessFp32() {
    UT_START();
    // ut(45, 8, padto(45, 48), 0, 45, 1.f, 1.f);
    // ut(45, 8, padto(45, 48), 0, 45, 1.f, 0.f);
    // ut(45, 8, padto(45, 48), 45, 45, 1.f, 0.f);
    CheckISA(AVX512F);
    ut(3, 8, padto(3, 48), 3, 3, 1.f, 1.f);
    ut(3, 8, padto(3, 48), 0, 3, 1.f, 1.f);
    ut(3, 8, padto(3, 48), 3, 3, 1.f, 0.f);
  }

  void ut(int _N, int _M, int _srcstep, int _src1step, int _dststep, float alpha, float beta) {
    printf("Test Case %d %d %d %d %d %f %f\n", _N, _M, _srcstep, _src1step, _dststep, alpha, beta);
    std::vector<float> src(_M * _srcstep), src1, dst(_M * _dststep, 0.f), dstref(_M * _dststep, 0.f);
    if (_src1step == 0) {
      src1.resize(_N, 10.f);
    } else {
      src1.resize(_M * _src1step, 10.f);
    }
    for (int i = 0; i < src.size(); i++) {
      src[i] = float(i);
    }
    epilogue::gemm::AlphaBetaProcessFp32<BTLA_ISA::NoSIMD> kernref;
    epilogue::gemm::AlphaBetaProcessFp32<BTLA_ISA::AVX512F> kern0;
    kernref.forward(src.data(), _srcstep, 0, 0, _M, _N, {dstref.data(), src1.data(), _dststep, _src1step, alpha, beta},
                    cache, CacheSize);
    kern0.forward(src.data(), _srcstep, 0, 0, _M, _N, {dst.data(), src1.data(), _dststep, _src1step, alpha, beta},
                  cache, CacheSize);
    ut::buffer_error<float>(dstref.data(), dst.data(), dstref.size());
  }
};
#ifdef BTLA_UT_EPILOGUE
static UT_AccumulatorWriteBack sUT_AccumulatorWriteBack;
static UT_AlphaBetaProcessFp32 sUT_AlphaBetaProcessFp32;
#endif
}  // namespace ut
}  // namespace bestla
