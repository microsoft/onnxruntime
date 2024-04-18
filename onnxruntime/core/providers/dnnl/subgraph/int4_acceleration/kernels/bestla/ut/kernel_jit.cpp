#include "kernel_jit.h"
#include "kernel_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
class UT_Memcpy2D_AVX512F {
 public:
  UT_Memcpy2D_AVX512F() {
    UT_START();
    CheckISA(AVX512F);
    ut(512, 432, 432, 432);
    ut(4, 432 * 1024, 432 * 2048, 432 * 1024);
    ut(16, 432 * 1024, 432 * 1024, 432 * 1024);
  }
  void ut(int row, int col, int srcstep, int dststep) {
    printf("Test Case: %d %d %d %d\n", row, col, srcstep, dststep);
    std::vector<float> src(row * srcstep), dst(row * dststep), dstref(row * dststep);
    for (int i = 0; i < src.size(); i++) {
      src[i] = float(i);
    }
    utils::timer<utils::microseconds> tm;
    size_t tsize = (size_t)row * col;

    int constexpr TestLoop = 1000;
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::JitMemcpy2DAvx512f::forward<float, float>(src.data(), dst.data(), row, col, srcstep, dststep);
    }
    tm.start();
    parallel::Scheduler2D para({DefaultThreading.num_threads(), row, col, 4, 64});
    for (size_t i = 0; i < TestLoop; i++) {
      DefaultThreading.parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        para.getIndex(thdp);
        if (thdp.valid) {
          kernel::jit::JitMemcpy2DAvx512f::forward<float, float>(  //
              src.data() + thdp.loc[0] * srcstep + thdp.loc[1],    //
              dst.data() + thdp.loc[0] * dststep + thdp.loc[1],    //
              thdp.size[0], thdp.size[1], srcstep, dststep);
        }
      });
    }

    auto tper = tm.stop() / TestLoop;
    printf("Kernel Time: %f us\n", tper);
    printf("Bandwidth: %f GB/s\n", tsize / tper / 1000);

    tm.start();
    for (size_t i = 0; i < TestLoop; i++) {
      DefaultThreading.parallel_for([&](int tidx) {
        parallel::ThreadProblem2D thdp{tidx};
        para.getIndex(thdp);
        if (thdp.valid) {
          kernel::ref::memcpy2d(                                    //
              src.data() + thdp.loc[0] * srcstep + thdp.loc[1],     //
              dstref.data() + thdp.loc[0] * dststep + thdp.loc[1],  //
              thdp.size[0], thdp.size[1] * sizeof(float), srcstep * sizeof(float), dststep * sizeof(float));
        }
      });
    }
    tper = tm.stop() / TestLoop;
    printf("Ref Time: %f us\n", tper);
    printf("Bandwidth: %f GB/s\n", tsize / tper / 1000);
    ut::buffer_error<float>(dstref.data(), dst.data(), dstref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_Memcpy2D_AVX512F sUT_Memcpy2D_AVX512F;
#endif

class UT_Memcpy2D_AVX2 {
 public:
  UT_Memcpy2D_AVX2() {
    UT_START();
    CheckISA(AVX2);
    ut(1, 1, 1, 4);
    ut(2, 2, 2, 4);
  }
  void ut(int row, int col, int srcstep, int dststep) {
    printf("Test Case: %d %d %d %d\n", row, col, srcstep, dststep);
    std::vector<float> src(row * srcstep), dst(row * dststep), dstref(row * dststep);
    for (int i = 0; i < src.size(); i++) {
      src[i] = float(i + 128);
    }

    kernel::jit::JitMemcpy2DAvx2::forward<float, float>(src.data(), dst.data(), row, col, srcstep, dststep);
    kernel::ref::memcpy2d(  //
        src.data(),         //
        dstref.data(),      //
        row, col * sizeof(src[0]), srcstep * sizeof(src[0]), dststep * sizeof(src[0]));
    ut::buffer_error<float>(dstref.data(), dst.data(), dstref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_Memcpy2D_AVX2 sUT_Memcpy2D_AVX2;
#endif

class UT_PaddngInterleaveCvt {
 public:
  UT_PaddngInterleaveCvt() {
    UT_START();
    CheckISA(AVX512_BF16);
    ut<48, float, bf16>(77, 256, 96);
    ut<48, float, bf16>(512, 432, 512);
    ut<48, float, bf16>(4, 432 * 1024, 4);
    ut<48, float, bf16>(16, 432 * 1024, 16);
  }
  template <int NTile, typename T_SRC, typename T_DST = T_SRC, int RowPack = 4 / sizeof(T_DST)>
  void ut(int rows, int cols, int rows_pad) {
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("row_%d col_%d row_pad_%d\n", rows, cols, rows_pad);
    const auto cols_pad = utils::padto(cols, NTile);
    const auto src_step = cols;
    const auto dst_step = rows_pad;
    std::vector<T_SRC> src(rows * cols);
    for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<float>(i);
    std::vector<T_DST> dst(cols_pad * dst_step), dst_ref(cols_pad * dst_step);

    utils::timer<utils::microseconds> tm;
    constexpr int TestLoop = 100;
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingInterleaveCvt::forward<NTile, T_SRC, T_DST, RowPack>(  //
          src.data(), dst.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    tm.start();
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingInterleaveCvt::forward<NTile, T_SRC, T_DST, RowPack>(  //
          src.data(), dst.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    const auto data_size = sizeof(T_DST) * rows_pad * cols_pad;
    const auto t_kern = tm.stop() / TestLoop;
    printf("Kernel Time: %f us\n", t_kern);
    printf("Bandwidth: %f GB/s\n", data_size / t_kern / 1000);

    tm.start();
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingInterleaveCvt::reference<NTile, T_SRC, T_DST, RowPack>(  //
          src.data(), dst_ref.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    const auto t_ref = tm.stop() / TestLoop;
    printf("Ref Time: %f us\n", t_ref);
    printf("Bandwidth: %f GB/s\n", data_size / t_ref / 1000);
    ut::buffer_error<T_DST>(dst_ref.data(), dst.data(), dst_ref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_PaddngInterleaveCvt sUT_Pading_InterleaveCvt;
#endif

class UT_PaddingTransInterleaveCvt {
 public:
  UT_PaddingTransInterleaveCvt() {
    UT_START();
    CheckISA(AVX512_BF16);
    ut<48, float, bf16>(48, 32, 32);
    ut<48, float, bf16>(77, 77, 96);
    ut<48, float, bf16>(77, 256, 256);
  }
  template <int MTile, typename T_SRC, typename T_DST = T_SRC, int ColPack = 4 / sizeof(T_DST)>
  void ut(int rows, int cols, int cols_pad) {
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("rows_%d cols_%d cols_pad_%d\n", rows, cols, cols_pad);
    const auto rows_pad = utils::padto(rows, MTile);
    const auto src_step = cols;
    const auto dst_step = cols_pad;
    std::vector<T_SRC> src(rows * cols);
    for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<float>(i);
    std::vector<T_DST> dst(rows_pad * dst_step), dst_ref(rows_pad * dst_step);

    utils::timer<utils::microseconds> tm;
    constexpr int TestLoop = 100;
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingTransInterleaveCvt::forward<MTile, T_SRC, T_DST, ColPack>(  //
          src.data(), dst.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    tm.start();
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingTransInterleaveCvt::forward<MTile, T_SRC, T_DST, ColPack>(  //
          src.data(), dst.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    const auto data_size = sizeof(T_DST) * cols_pad * rows_pad;
    const auto t_kern = tm.stop() / TestLoop;
    printf("Kernel Time: %f us\n", t_kern);
    printf("Bandwidth: %f GB/s\n", data_size / t_kern / 1000);

    tm.start();
    for (int i = 0; i < TestLoop; i++) {
      kernel::jit::PaddingTransInterleaveCvt::reference<MTile, T_SRC, T_DST, ColPack>(  //
          src.data(), dst_ref.data(), rows, cols, rows_pad, cols_pad, src_step, dst_step);
    }
    const auto t_ref = tm.stop() / TestLoop;
    printf("Ref Time: %f us\n", t_ref);
    printf("Bandwidth: %f GB/s\n", data_size / t_ref / 1000);
    ut::buffer_error<T_DST>(dst_ref.data(), dst.data(), dst_ref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_PaddingTransInterleaveCvt sUT_PaddingTransInterleaveCvt;
#endif

class UT_CScaleInterleavedBF16FP16 {
 public:
  UT_CScaleInterleavedBF16FP16() {
    UT_START();
    CheckISA(AVX512_BF16);
    CheckISA(AVX512_FP16);
    ut<48, 2>(32, 96, 0);
    ut<48, 2>(32, 96, 4);
  }
  template <int NTile, int RowPack = 2>
  void ut(int rows, int cols, int n_offset) {
    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("rows_%d cols_%d n_offset_%d\n", rows, cols, n_offset);
    const auto src_step = rows;
    std::vector<utils::bf16> data(rows * cols);
    std::vector<utils::bf16> data_ref(data.size());
    for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<utils::bf16>(static_cast<float>(i % 523 - 300));
    std::copy_n(data.cbegin(), data.size(), data_ref.begin());

    std::vector<utils::fp16> scale(rows);
    for (size_t i = 0; i < scale.size(); ++i)
      scale[i] = static_cast<utils::fp16>(static_cast<float>(i % 523 - 300) / 10.f);

    kernel::jit::CScaleInterleavedBF16FP16::forward<NTile, RowPack>(  //
        data.data(), scale.data(), rows, cols, src_step, n_offset);
    kernel::jit::CScaleInterleavedBF16FP16::reference<NTile, RowPack>(  //
        data_ref.data(), scale.data(), rows, cols, src_step, n_offset);

    ut::buffer_error<utils::bf16>(data.data(), data_ref.data(), data.size());
    printf("\n");
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_CScaleInterleavedBF16FP16 sUT_CScaleInterleavedBF16FP16;
#endif

class UT_DeQuant {
 public:
  UT_DeQuant() {
    UT_START();
    ut<float>(512, 48);
    ut<float>(512, 64);
    ut<float>(512, 32);
    ut<bf16>(512, 48);
    ut<bf16>(512, 64);
    ut<bf16>(512, 512);
  }
  template <typename DST_T>
  void ut(int row, int col) {
    int srcstride = col;
    int dststride = col * 4;
    printf("Test Case : %d %d %d %d\n", row, col, srcstride, dststride);
    CheckISA(AVX512F);
    ut::UT_vector_s8 test;
    test.resize(row * col);
    test.fill_rand(-127, 127);

    test.rand_scale(col, -0.05f, 0.05f);
    utils::aligned_vector<DST_T> ref, tar;
    ref.resize(row * col);
    tar.resize(row * col);
    int constexpr PACK_ROW = std::is_same_v<DST_T, float> ? 1 : 2;
    kernel::ref::decompress_kblock_s8_fp<DST_T, PACK_ROW>(test.data(), ref.data(), row, col, col, col,
                                                          test.scales.data(), nullptr, 0, row * 2, col);
    kernel::jit::DequanS8FP::forward_avx512f<PACK_ROW>(test.data(), tar.data(), row, col, col, col, test.scales.data(),
                                                       nullptr);
    ut::buffer_error<DST_T>(ref.data(), tar.data(), ref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_DeQuant sUT_DeQuant;
#endif

class UT_DecompressS4S8 {
 public:
  UT_DecompressS4S8() {
    UT_START();
    ut(512, 48);
    ut(2, 48);
    ut(111, 48);
  }
  void ut(int row, int col) {
    printf("Test Case : %d %d\n", row, col);
    CheckISA(AVX512F);
    ut::UT_vector_s8 test;
    aligned_vector<int4x2> src(row * col / 2);
    aligned_vector<int8_t> src8(row * col);
    ut::fill_buffer_randn(src8.data(), src8.size(), int8_t(-128), int8_t(127));
    kernel::ref::compress_s8_s4<48>(src8.data(), src.data(), row, col, col, col);
    aligned_vector<int8_t> ref(row * col), tar(row * col);
    kernel::ref::decompress_s4_s8<BTLA_DTYPE::S4_CLIP>(src.data(), ref.data(), row, col, col, col);
    kernel::jit::decompress_s4_s8(src.data(), tar.data(), row, col, col, col);
    ut::buffer_error<int8_t>(ref.data(), tar.data(), ref.size());
  }
};
#ifdef BTLA_UT_KERNEL_JIT
static UT_DecompressS4S8 sUT_DecompressS4S8;
#endif
}  // namespace ut
}  // namespace bestla
