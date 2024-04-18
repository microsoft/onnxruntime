#include "bestla_gemm.h"
#include "bestla_prologue_b.h"
#include "bestla_parallel.h"
#include "bestla_device.h"
#include "bestla_wrapper.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
class UT_BlockQunatize_INT8 {
 public:
  UT_BlockQunatize_INT8() {
    UT_START();
    CheckISA(AVX512F);
    ut(1024, 1024, 32);
    ut(1024, 1024, 32, true);
    ut(4128, 4096, 32);
    ut(4128, 4096, 32, true);
    ut(1024, 4096, 32);
    ut(4096, 1024, 32);

    ut_transpose(4096, 4096, 32);
    ut_transpose(4096, 4096, 32, true);
    ut_transpose(4128, 4096, 32);
    ut_transpose(4128, 4096, 32, true);
    ut_transpose(1024, 4096, 32);
    ut_transpose(4096, 1024, 32);
  }

  void ut(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * ldb + i] = (float(quanW.data()[j * ldb + i]) - float(zero_points[j / blocksize * n + i])) *
                                   scales[j / blocksize * n + i];
        } else {
          dequanRef[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
        }
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx512f<48, 8>, RuntimeISA>;
    PrologueB kernel;
    auto ptr = kernel.createStorage(n, k, blocksize, BTLA_DTYPE::S8, bestla_dtype<float>, bestla_dtype<float>, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    kernel.packWeight(n, k, dequanRef.data(), ldb, &ptr, &DefaultThreading);
    avector<float> dequant(n * k);
    kernel.unpackWeight(n, k, &ptr, dequant.data(), n, &DefaultThreading);
    avector<int8_t> ws8(n * k);
    kernel.unpackWeight(n, k, &ptr, ws8.data(), n, &DefaultThreading);
    ut::buffer_error(quanW.data(), ws8.data(), ws8.size(), (int8_t)1);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }

  void ut_transpose(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    avector<float> dequanT(k * n);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * n + i] = (float(quanW.data()[j * n + i]) - float(zero_points[j / blocksize * n + i])) *
                                 scales[j / blocksize * n + i];
        } else {
          dequanRef[j * n + i] = float(quanW.data()[j * n + i]) * scales[j / blocksize * n + i];
        }
        dequanT[j + i * k] = dequanRef[j * n + i];
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx512f<48, 8>, RuntimeISA>;
    PrologueB kernel;
    auto ptr = kernel.createStorage(n, k, blocksize, BTLA_DTYPE::S8, bestla_dtype<float>, bestla_dtype<float>, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    kernel.packTransposeWeight(n, k, dequanT.data(), k, &ptr, &DefaultThreading);
    avector<float> dequant(n * k), tardequanT(k * n);
    kernel.unpackWeight(n, k, &ptr, dequant.data(), n, &DefaultThreading);
    kernel.unpackTransposeWeight(n, k, &ptr, tardequanT.data(), k, &DefaultThreading);
    ut::buffer_error(dequanT.data(), tardequanT.data(), tardequanT.size(), 0.01f);
    avector<int8_t> ws8(n * k);
    kernel.unpackWeight(n, k, &ptr, ws8.data(), n, &DefaultThreading);
    ut::buffer_error(quanW.data(), ws8.data(), ws8.size(), (int8_t)1);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_BlockQunatize_INT8 sUT_BlockQunatize_INT8;
#endif

class UT_BlockQunatize_F8 {
 public:
  UT_BlockQunatize_F8() {
    UT_START();
    CheckISA(AVX512F);
    ut(127, 1023, 32, BTLA_DTYPE::F8_E4M3);
    ut(127, 1023, 32, BTLA_DTYPE::F8_E5M2);
  }

  void ut(int n, int k, int blocksize, BTLA_DTYPE QUANT_T) {
    printf("%s: %d %d %d\n", __FUNCTION__, n, k, blocksize);
    int ldb = n;
    utils::aligned_vector<float> raw(n * k);
    ut::fill_buffer_randn(raw.data(), raw.size(), -3.f, 3.f);

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNFloat<gemm::SCoreRowNAvx512f<48, 8>, RuntimeISA>;
    using refPorB = prologue_b::gemm::WeightKBlockNFloat<gemm::SCoreRowNAvx512f<48, 8>, BTLA_ISA::NoSIMD>;
    PrologueB kernel;
    refPorB ref_ker;
    auto ptr = kernel.createStorage(n, k, blocksize, QUANT_T, BTLA_DTYPE::F8_E8M0);
    auto ref_ptr = kernel.createStorage(n, k, blocksize, QUANT_T, BTLA_DTYPE::F8_E8M0);
    avector<int8_t> buffer(ptr.mSize);
    avector<int8_t> ref_buffer(ptr.mSize);
    ptr.assign(buffer.data());
    ref_ptr.assign(ref_buffer.data());
    kernel.packWeight(n, k, raw.data(), ldb, &ptr, &DefaultThreading);
    ref_ker.packWeight(n, k, raw.data(), ldb, &ref_ptr, &DefaultThreading);
    avector<float> dequant(n * k, 0);
    avector<float> ref_dequant(n * k, 0);
    kernel.unpackWeight(n, k, &ptr, dequant.data(), n, &DefaultThreading);
    ref_ker.unpackWeight(n, k, &ref_ptr, ref_dequant.data(), n, &DefaultThreading);
    ut::buffer_error(ref_dequant.data(), dequant.data(), dequant.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_BlockQunatize_F8 sUT_BlockQunatize_F8;
#endif

class UT_TransposeBlockQuantize_F4 {
 public:
  UT_TransposeBlockQuantize_F4() {
    UT_START();
    CheckISA(AVX512F);
    ut(4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut(1024, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut(4096, 1024, 32, BTLA_DTYPE::F4_BNB);
    ut(48, 32, 32, BTLA_DTYPE::F4_BNB);
    ut(32, 32, 32, BTLA_DTYPE::F4_BNB);
    ut(48, 32, 32, BTLA_DTYPE::F4_BNB);
    ut(48, 32, 32, BTLA_DTYPE::F4_NF4);
    ut(48, 32, 32, BTLA_DTYPE::F4_E2M1);
  }

  void ut(int n, int k, int blocksize, BTLA_DTYPE F4_T) {
    printf("Test Case: %d %d %d\n", n, k, blocksize);
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 1.f, 5.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          switch (F4_T) {
            case BTLA_DTYPE::F4_E2M1:
              quanW.data()[i * n + j] = 7;  // make sure each block has maximum fp4e2m1 value(0b111) to quantize
              break;
            case BTLA_DTYPE::F4_BNB:
              quanW.data()[i * n + j] = 3;  // make sure each block has maximum fp4bnb value(0b011) to quantize
              break;
            case BTLA_DTYPE::F4_NF4:
              quanW.data()[i * n + j] = 15;  // make sure each block has maximum nf4 value(0b1111) to quantize
              break;
            default:
              break;
          }
        }
      }
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        switch (F4_T) {
          case BTLA_DTYPE::F4_E2M1:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_E2M1>(quanW.data()[j * ldb + i],
                                                                                   scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_E2M1>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          case BTLA_DTYPE::F4_BNB:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_BNB>(quanW.data()[j * ldb + i],
                                                                                  scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_BNB>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          case BTLA_DTYPE::F4_NF4:
            dequanRef[j + i * k] = kernel::ref::f4_dequantize<BTLA_DTYPE::F4_NF4>(quanW.data()[j * ldb + i],
                                                                                  scales[j / blocksize * n + i]);
            quanW.data()[j * ldb + i] =
                kernel::ref::f4_quantize<BTLA_DTYPE::F4_NF4>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
            break;
          default:
            break;
        }
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512F;
    using PrologueB = prologue_b::gemm::WeightKBlockNFloat<gemm::SCoreRowNAvx512f<48, 8>, RuntimeISA>;
    PrologueB kernel;
    auto packedW = kernel.createStorage(n, k, blocksize, F4_T, bestla_dtype<float>);
    auto packedW1 = kernel.createStorage(n, k, blocksize, F4_T, bestla_dtype<float>);
    avector<int8_t> buf(packedW.mSize), buf1(packedW1.mSize);
    packedW.assign(buf.data());
    packedW1.assign(buf1.data());
    kernel.packTransposeWeight(n, k, dequanRef.data(), k, &packedW, &DefaultThreading);
    kernel.packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedW1, &DefaultThreading);
    ut::buffer_error(packedW.SPtr<float>(), packedW1.SPtr<float>(), packedW1.CSize());
    ut::buffer_error(packedW.WPtr<int8_t>(), packedW1.WPtr<int8_t>(), packedW1.mQBuf.size<int8_t>());
    avector<float> dequant(n * k);
    kernel.unpackTransposeWeight(n, k, &packedW1, dequant.data(), k, &DefaultThreading);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequant.size());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_TransposeBlockQuantize_F4 sUT_TransposeBlockQuantize_F4;
#endif

class UT_BlockQuantize_INT4 {
 public:
  UT_BlockQuantize_INT4() {
    UT_START();
    CheckISA(AVX2);
    CheckISA(AVX512F);
    ut_2(4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut_2(4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE, false);
    CheckISA(AVX512F);
    ut_512vnni(4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut_512vnni(4096, 4096, 128, BTLA_DTYPE::S4_CLIP, true);
    ut_512vnni(4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE, false);
  }
  void ut_2(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.005f, 0.01f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    avector<float> dequant(quanW.size());
    avector<float> reduce(scales.size(), 0.f);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
        if (!asym) {
          dequant[i * n + j] = quanW.data()[i * n + j] * scales[i / blocksize * n + j];
        } else {
          dequant[i * n + j] =
              float(quanW.data()[i * n + j] - zero_points[i / blocksize * n + j]) * scales[i / blocksize * n + j];
        }
        reduce[i / blocksize * n + j] += dequant[i * n + j];
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX2;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx2<48, 2>, BTLA_ISA::AVX2>;
    using PrologueB512 = prologue_b::gemm::WeightKBlockNInteger<gemm::SCoreRowNAvx2<48, 2>, BTLA_ISA::AVX512F>;
    PrologueB kernel;
    PrologueB512 kernel512;
    utils::aligned_vector<int8_t> retW(n * k);
    auto packedW = kernel.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<float>, asym);
    avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.packWeight(n, k, dequant.data(), ldb, &packedW, &DefaultThreading);
    avector<float> unpackf32(dequant.size());
    avector<float> unpack512f32(dequant.size());
    kernel.unpackWeight(n, k, &packedW, unpackf32.data(), n, &DefaultThreading);
    kernel512.unpackWeight(n, k, &packedW, unpack512f32.data(), n, &DefaultThreading);
    ut::buffer_error(unpackf32.data(), unpack512f32.data(), unpackf32.size(), 0.01f);
  }
  void ut_512vnni(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.005f, 0.01f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    avector<float> dequant(quanW.size());
    avector<float> reduce(scales.size(), 0.f);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        // quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0; //anyway there will be a float-rounding error
        // about 1 LSB.
        if (!asym) {
          dequant[i * n + j] = quanW.data()[i * n + j] * scales[i / blocksize * n + j];
        } else {
          dequant[i * n + j] =
              float(quanW.data()[i * n + j] - zero_points[i / blocksize * n + j]) * scales[i / blocksize * n + j];
        }
        reduce[i / blocksize * n + j] += dequant[i * n + j];
      }
    }

    auto constexpr RuntimeISA = BTLA_ISA::AVX512_VNNI;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<gemm::ICoreRowNAvx512vnni<48, 8>, RuntimeISA>;

    PrologueB kernel;
    utils::aligned_vector<int8_t> retW(n * k);
    auto packedW = kernel.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<float>, asym);
    avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.packWeight(n, k, dequant.data(), ldb, &packedW, &DefaultThreading);
    avector<float> unpackf32(dequant.size());
    kernel.unpackWeight(n, k, &packedW, unpackf32.data(), n, &DefaultThreading);
    int lsb = 16;
    float err_thres = lsb * 0.01f;  // lsb*max_scale
    ut::buffer_error(dequant.data(), unpackf32.data(), dequant.size(), err_thres);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_BlockQuantize_INT4 sUT_BlockQuantize_INT4;
#endif

class UT_StorageMemCheck {
 public:
  UT_StorageMemCheck() {
    UT_START();
    CheckISA(AVX512F);
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut_s4(4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE, true);
    ut_f4(4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut_f4(4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::SCoreRowNAvx512f<48, 8>;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore, BTLA_ISA::AVX2>;
    PrologueB ProWei;

    auto packedW = ProWei.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, asym);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    storage::gemm::StorageWeightKBlockNInteger tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }

  void ut_f4(int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test C type Case: %d %d %d\n", n, k, blocksize);
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::HCoreRowNAmxbf16<64, 16>;
    using PrologueB = prologue_b::gemm::WeightKBlockNFloat<GemmCore, BTLA_ISA::AMX_BF16>;
    PrologueB ProWei;

    auto packedW = ProWei.createStorage(n, k, blocksize, qtype, bestla_dtype<float>);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    storage::gemm::StorageWeightKBlockNFloat tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_StorageMemCheck sUT_StorageMemCheck;
#endif

class UT_ShuffleIndices {
 public:
  UT_ShuffleIndices() {
    UT_START();
    CheckISA(AVX2);
    // ut_file();
    ut_s4(4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE, true);
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = gemm::SCoreRowNAvx2<24, 4>;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore, BTLA_ISA::AVX2>;
    PrologueB ProWei;
    auto packedW = ProWei.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, asym);
    ProWei.enableShuffle(&packedW);
    avector<int> groupindices(k, 0);
    auto groupsize = utils::updiv(k, blocksize);
    avector<int> reflut(k, 0);
    for (size_t i = 0; i < k; i++) {
      groupindices[i] = i % groupsize;
      auto offset = i / groupsize;
      reflut[groupindices[i] * blocksize + offset] = i;
    }
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    ProWei.setShuffleIndices(groupindices.data(), &packedW, &DefaultThreading);
    buffer_error(reflut.data(), packedW.ShfIndice(), reflut.size());

    storage::gemm::StorageWeightKBlockNInteger tmp(GemmCore::ID);
    tmp.deserialize(buf0.data());
    tmp.serialize(buf1.data());
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }

  void ut_file() {
    int n = 14336;
    int m = 8;
    int k = 4096;
    int blocksize = 32;
    bool constexpr blauncher = false;
    auto qtype = BTLA_DTYPE::S4_CLIP;
    bool asym = true;
    auto warray = ut::readFile2Buffer<int8_t>("src0_data.bin");
    auto aarray = ut::readFile2Buffer<float>("src1_data.bin");
    auto oarray = ut::readFile2Buffer<float>("tensor_data.bin");
    auto refoarray = ut::readFile2Buffer<float>("tensor_data_ref.bin");
    auto wptr = storage::gemm::PackedWeightParser::deserialBuffer(warray.data());
    using GemmCore = gemm::SCoreRowNAvx512f<48, 8>;
    auto wptr_ = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(wptr);
    utils::GemmProblem gp(1, m, n, k, blocksize);
    avector<float> output(m * n);
    if constexpr (blauncher) {
      using Launcher =
          wrapper::gemm::LauncherBase<GemmCore::ISA, GemmCore, prologue_a::gemm::ShuffleActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::AccumulatorWriteBackFp32>;
      static Launcher kernel;
      auto rordA = kernel.mProA.createReorderStorage(m, k, blocksize);
      avector<int8_t> bufA(rordA.mSize);
      rordA.assign(bufA.data());
      typename Launcher::Param args{
          gp, {aarray.data(), k, nullptr, wptr_->ShfIndice(), &rordA}, {wptr_}, {output.data(), n}};
      parallel::GemmRunWithA<parallel::gemm::SchedulerBase<GemmCore>>(kernel, args, &DefaultThreading);

    } else {
      using Launcher =
          wrapper::gemm::LauncherKBlock<GemmCore::ISA, GemmCore, prologue_a::gemm::ShuffleActivationKBlockBaseF32,
                                        prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompFp32BlockEpilogue,
                                        epilogue::gemm::AccumulatorWriteBackFp32>;
      static Launcher kernel;
      auto rordA = kernel.mProA.createReorderStorage(m, k, blocksize);
      auto redA = kernel.mProA.createReduceStorage(m, k, blocksize);
      avector<int8_t> bufA(rordA.mSize + redA.mSize);
      rordA.assign(bufA.data());
      redA.assign(bufA.data() + rordA.mSize);
      typename Launcher::BEpiParam blkargs{
          wptr_->template SPtr<int8_t>(), wptr_->SDtype(), wptr_->CStep(), wptr_->template ZPtr<int8_t>(),
          redA.template RPtr<float>(),    redA.lda};
      typename Launcher::Param args{
          gp, {aarray.data(), k, &redA, wptr_->ShfIndice(), &rordA}, {wptr_}, blkargs, {output.data(), n}};
      parallel::GemmRunWithA<parallel::gemm::SchedulerKBlock<GemmCore>>(kernel, args, &DefaultThreading);
    }

    ut::buffer_error(output.data(), oarray.data(), output.size());
    ut::buffer_error(output.data(), refoarray.data(), output.size());

    delete wptr;
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_ShuffleIndices sUT_ShuffleIndices;
#endif

class UT_CompFp32 {
 public:
  UT_CompFp32() {
    UT_START();
    ut_s4();
    ut_s8();
    ut_f4();
    ut_f8();
  }

  void ut_f8() {
    CheckISA(AVX2);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
    CheckISA(AVX512F);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
  }
  void ut_s4() {
    CheckISA(AVX2);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE, BTLA_DTYPE::F32,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::BF16,
                                                          false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE, BTLA_DTYPE::BF16,
                                                          false);

    CheckISA(AVX512F);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE,
                                                             BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE,
                                                             BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE,
                                                             BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::BF16,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE,
                                                             BTLA_DTYPE::BF16, false);
  }

  void ut_s8() {
    CheckISA(AVX2);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S8, BTLA_DTYPE::BF16, false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S8, BTLA_DTYPE::F32, false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S8, BTLA_DTYPE::F32, false);
    ut_int<sAVX2, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S8, BTLA_DTYPE::F32, false);

    CheckISA(AVX512F);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S8, BTLA_DTYPE::BF16,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 32, BTLA_DTYPE::S8, BTLA_DTYPE::F32, false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, 128, BTLA_DTYPE::S8, BTLA_DTYPE::F32,
                                                             false);
    ut_int<sAVX512F, prologue_b::gemm::WeightKBlockNInteger>(2, 4096, 4096, -1, BTLA_DTYPE::S8, BTLA_DTYPE::F32, false);
  }

  void ut_f4() {
    CheckISA(AVX2);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX2, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);

    CheckISA(AVX512F);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512F, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  }

  template <class GemmCore_T, template <class _T, BTLA_ISA> class Wei>
  void ut_int(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype));
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompFp32BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T, ISA>::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockNInteger<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, stype, bestla_dtype<float>, isAsym);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockNFloat<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, stype);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    utils::GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp,
                                  {matAf32.data(), k},
                                  {&packedw},
                                  {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep()},
                                  {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    auto err = INT8_ERR;
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto constexpr dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == dtype_int) {
      if (dbits == 8) {
        err = INT8_ERR;
      } else {
        err = INT4_ERR;
      }
    } else {
      err = FP4_ERR;
    }
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                      epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T, ISA>::StorageWeight;
    WType packedw(0);
    packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp,
                                  {matAf32.data(), k},
                                  {&packedw},
                                  {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep()},
                                  {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    auto err = FP4_ERR;

    if (qtype == BTLA_DTYPE::F8_E5M2 || qtype == BTLA_DTYPE::F8_E4M3) err = F8_ERR;

    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompFp32 sUT_CompFp32;
#endif

class UTBenchmark_CompFp32 {
 public:
  UTBenchmark_CompFp32() {
    UT_START();
    CheckISA(AVX512F);
    ut_s4();
    /*   ut_s8();
       ut_f4();*/
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2048, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(4096, 4096, 11008, 128, BTLA_DTYPE::S4_CLIP);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32,
    //  BTLA_DTYPE::S4_FULLRANGE);
  }

  // void ut_s8() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  // }

  // void ut_f4() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  // }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);

    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], &DefaultThreading);
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(float));
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          double band = double(memsize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr,
                 log.get_log_str(), flops, flops / threads, band);
        }
      }
    }
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_mem(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems,
                     int threads, BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlock<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, Wei,
                                      epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);

    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], &DefaultThreading);
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(float));
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp,
                                      {A + i * m * k, k},
                                      {&packBs[i]},
                                      {packBs[i].template SPtr<int8_t>(), packBs[i].SDtype(), packBs[i].CStep()},
                                      {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
      }
      if (log.stop()) {
        double t = log.avg_val / batch;
        double flops = double(psize) / t / 1e6;
        double band = double(memsize) / t / 1e6;
        printf("Threads %d %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr, flops,
               flops / threads, band);
      }
    }
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(size_t m, size_t n, size_t k, size_t batch, BTLA_DTYPE qtype) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch);
    avector<float> B(k * n);
    avector<float> C(m * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (size_t i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512F()) {
      int blocksize = 32;
      benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                  C.data(), testtime, 48, qtype);
      benchmark_mem<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                      C.data(), testtime, 48, qtype);
      blocksize = 128;
      benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                  C.data(), testtime, 48, qtype);
      benchmark_mem<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                      C.data(), testtime, 48, qtype);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B_
static UTBenchmark_CompFp32 sUTBenchmark_CompFp32;
#endif

class UT_CompInt8 {
 public:
  UT_CompInt8() {
    UT_START();
    ut_s4();
    ut_s8();
    ut_s4_newkblock();
  }

  void ut_s4() {
    GetCPUDevice();
    if (_cd->AVX_VNNI()) {
      ut<sAVX_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut<sAVX_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut<sAVX_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
      ut<sAVX_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut<sAVX_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    }

    if (_cd->AVX512_VNNI()) {
      ut_dynamic<sAVX512_VNNI, float>(1, 11008, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut_dynamic<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut_dynamic<sAVX512_VNNI, float>(1, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, false, BTLA_DTYPE::BF16);
      ut_dynamic<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX512_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut<sAVX512_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    }

    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut<sAMX_INT8_US, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut<sAMX_INT8_US, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
      ut<sAMX_INT8_US, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAMX_INT8_US, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAMX_INT8_US, float>(16, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAMX_INT8_US, utils::bf16>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut<sAMX_INT8_US, utils::bf16>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
      ut<sAMX_INT8_US, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, true);
      ut_s8s8<sAMX_INT8_SS, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut_s8s8<sAMX_INT8_SS, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
      ut_s8s8<sAMX_INT8_SS, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, true);
    }
  }

  void ut_s4_newkblock() {
    GetCPUDevice();
    if (_cd->AVX_VNNI()) {
      ut_newkblock<gemm::ICoreRowNAvxvnniKBlock<48, 1>, float>(1, 11008, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut_newkblock<gemm::ICoreRowNAvxvnniKBlock<48, 1>, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    }

    if (_cd->AVX512_VNNI()) {
      ut_newkblock<gemm::ICoreRowNAvx512vnniKBlock<48, 4>, float>(1, 11008, 4096, 32, BTLA_DTYPE::S4_CLIP);
      ut_newkblock<gemm::ICoreRowNAvx512vnniKBlock<48, 4>, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    }

    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut_newkblock<gemm::ICoreRowNAmxint8KBlock<48, 16>, float>(128, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
      ut_newkblock<gemm::ICoreRowNAmxint8KBlock<48, 16>, float>(1, 4096, 4096, 64, BTLA_DTYPE::S4_CLIP);
    }
  }

  void ut_s8() {
    GetCPUDevice();
    if (_cd->AVX_VNNI()) {
      ut_dynamic<sAVX_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
      ut_dynamic<sAVX_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
      ut_dynamic<sAVX_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
      ut_dynamic<sAVX_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
    }

    if (_cd->AVX512_VNNI()) {
      ut_dynamic<sAVX512_VNNI, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
      ut_dynamic<sAVX512_VNNI, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
      ut_dynamic<sAVX512_VNNI, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
      ut_dynamic<sAVX512_VNNI, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
    }

    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut_dynamic<sAMX_INT8_US, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
      ut_dynamic<sAMX_INT8_US, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
      ut_dynamic<sAMX_INT8_US, utils::bf16>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
    }
  }

  template <class GemmCore_T, typename Scale_T>
  void ut_newkblock(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isAsym = false) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>, isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherIntKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                                      prologue_b::gemm::WeightKBlockNInteger,
                                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlockS<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw =
        launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, isAsym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<float> reduceAf32(m * kblks, 0.f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
        reduceAf32[i * kblks + j / blocksize] += matAf32[i * k + j];
      }
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    auto quanA = launcher.mProA.createStorage(m, k, blocksize, isAsym);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp, {matAf32.data(), k, &quanA}, {&packedw}, {matC.data(), n}};
    parallel::GemmRunWithA<Parallel>(launcher, args, &DefaultThreading);
    auto err = INT8_ERR;
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == dtype_int) {
      if (dbits == 8) {
        err = INT8_ERR;
      } else {
        err = INT4_ERR;
      }
    }
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), INT8_ERR);  // dynamic quant error
  }

  template <class GemmCore_T, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isAsym = false) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>, isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompInt8BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw =
        launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, isAsym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<float> reduceAf32(m * kblks, 0.f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
        reduceAf32[i * kblks + j / blocksize] += matAf32[i * k + j];
      }
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAu8.data(), k},
        {&packedw},
        {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep(), scaleAf32.data(), kblks, zpAu8.data(),
         packedw.template RPtr<void>(), packedw.RDtype(), isAsym ? packedw.template ZPtr<int8_t>() : nullptr,
         isAsym ? reduceAf32.data() : nullptr, blocksize},
        {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    auto err = INT8_ERR;
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == dtype_int) {
      if (dbits == 8) {
        err = INT8_ERR;
      } else {
        err = INT4_ERR;
      }
    }
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T, typename Scale_T>
  void ut_dynamic(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isAsym = false,
                  BTLA_DTYPE redtype = BTLA_DTYPE::F32) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d reduce dtype:%s\n", __FUNCTION__, m, n, k,
           blocksize, bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>, isAsym,
           bestla_dtype_str(redtype));
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompInt8BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, redtype, isAsym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matAu8(m * k), zpAu8(m * kblks);
    avector<float> scaleAf32(m * kblks);
    auto quanA = launcher.mProA.createStorage(m, k, blocksize, isAsym);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<float> reduceAf32(m * kblks, 0.f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] =
            (float(matAu8[i * k + j]) - zpAu8[i * kblks + j / blocksize]) * scaleAf32[i * kblks + j / blocksize];
        reduceAf32[i * kblks + j / blocksize] += matAf32[i * k + j];
      }
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAf32.data(), k, &quanA},
        {&packedw},
        {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep(), quanA.template SPtr<float>(),
         quanA.CStep(), quanA.template ZPtr<uint8_t>(), packedw.template RPtr<void>(), packedw.RDtype(),
         packedw.template ZPtr<int8_t>(), quanA.template RPtr<float>(), blocksize},
        {matC.data(), n}};
    parallel::GemmRunWithA<Parallel>(launcher, args, &DefaultThreading);
    auto err = INT8_ERR;
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == dtype_int) {
      if (dbits == 8) {
        err = INT8_ERR;
      } else {
        err = INT4_ERR;
      }
    }

    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), INT8_ERR);  // dynamic quant error
  }

  template <class GemmCore_T, typename Scale_T>
  void ut_s8s8(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isAsym = false) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>, isAsym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompInt8BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    int kblks = updiv(k, blocksize);
    using WType = typename Launcher::PrologueB::StorageWeight;
    WType packedw =
        launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, isAsym);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<int8_t> matAu8(m * k);
    avector<float> scaleAf32(m * kblks);
    fill_buffer_randn(matAu8.data(), matAu8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    ut::fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    avector<float> reduceAf32(m * kblks);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] = (float(matAu8[i * k + j])) * scaleAf32[i * kblks + j / blocksize];
        reduceAf32[i * kblks + j / blocksize] += matAf32[i * k + j];
      }
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAu8.data(), k},
        {&packedw},
        {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep(), scaleAf32.data(), kblks, nullptr, nullptr,
         bestla_dtype<float>, packedw.template ZPtr<int8_t>(), reduceAf32.data(), blocksize},
        {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    auto err = INT8_ERR;
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    if (type == dtype_int) {
      if (dbits == 8) {
        err = INT8_ERR;
      } else {
        err = INT4_ERR;
      }
    }

    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompInt8 sUT_CompInt8;
#endif

class UT_CompBf16 {
 public:
  UT_CompBf16() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut_s4();
    ut_s8();
    ut_f4();
    ut_f8();
  }

  void ut_f8() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E4M3);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, f8>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F8_E5M2);
  }

  void ut_s4() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
  }

  void ut_s8() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAMX_BF16, prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  }

  template <class GemmCore_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                      epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;

    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T, ISA>::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockNInteger<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockNFloat<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBf32[i] = matBbf16[i];
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBbf16[i] = static_cast<utils::bf16>(matBf32[i]);
    }
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{gp,
                                  {matAbf16.data(), k},
                                  {&packedw},
                                  {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep()},
                                  {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    auto err = get_ut_err(qtype);
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.05f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_CompBf16 sUT_CompBf16;
#endif

class UTBenchmark_CompBf16 {
 public:
  UTBenchmark_CompBf16() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut_s4();
    /*   ut_s8();
       ut_f4();*/
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(2048, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(4096, 4096, 11008, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32,
    // BTLA_DTYPE::S4_FULLRANGE);
  }

  // void ut_s8() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  // }

  // void ut_f4() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  // }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationConverterFp32, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    DefaultThreading.set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, B + i * n * k, n, &packBs[i], &DefaultThreading);
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, &DefaultThreading);
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          double band = double(memsize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr,
                 log.get_log_str(), flops, flops / threads, band);
        }
      }
    }
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(size_t m, size_t n, size_t k, size_t batch, BTLA_DTYPE qtype) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch);
    avector<float> B(k * n * batch);
    avector<float> C(m * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_BF16()) {
      request_perm_xtile_data();
      int blocksize = 32;
      benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, 48, qtype);
      benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, 48, qtype);
      benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, 48, qtype);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B_
static UTBenchmark_CompBf16 sUTBenchmark_CompBf16;
#endif

class UT_ORT_NBits {
 public:
  UT_ORT_NBits() {
    UT_START();
    ut_s4();
  }

  void ut_s4() {
    CheckISA(AVX2);
    ut<sAVX2>(1, 14336, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(1, 1, 32, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(1, 2, 32, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX2>(1, 11008, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX2>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, false);
    CheckISA(AVX512F);
    ut<sAVX512F>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, true);
    ut<sAVX512F>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX512F>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, false);
    ut<sAVX512F>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP, false);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, bool isasym) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s asym:%d \n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), isasym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompFp32BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    using WType = storage::gemm::StorageWeightKBlockNInteger;
    WType packedw(0);
    packedw =
        launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, isasym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    avector<uint8_t> matBs8(n * k);
    avector<int4x2> matBs4(n * updiv(k, 2));
    int blks = updiv(k, blocksize);
    avector<float> scalesB(n * blks);
    avector<uint8_t> zpBs8(n * blks, 8);
    auto blk_padding = updiv(blks, 2);
    avector<int4x2> zpBs4(n * blk_padding, uint8_t(0x88));
    fill_buffer_randn(matBs8.data(), matBs8.size(), uint8_t(0), uint8_t(15));
    if (isasym) {
      fill_buffer_randn(zpBs8.data(), zpBs8.size(), uint8_t(0), uint8_t(15));
    }
    fill_buffer_randn(scalesB.data(), scalesB.size(), 0.001f, 0.005f);
    avector<float> reduceA(m * blks, 0.f);

    auto rA = launcher.mProA.createStorage(m, k, blocksize);
    avector<int8_t> tmpA(rA.mSize);
    if (isasym) {
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          reduceA[i * blks + j / blocksize] += matAf32[i * k + j];
        }
      }
      rA.assign(tmpA.data());
      launcher.mProA.reduce({matAf32.data(), k, &rA}, m, k, blocksize, &DefaultThreading);  // for reduce UT
      buffer_error(reduceA.data(), rA.template RPtr<float>(), reduceA.size(), FP32_ERR);
      memset(tmpA.data(), 0, tmpA.size());  // clear
    }
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j += 2) {
        *(uint8_t*)&matBs4[i * k / 2 + j / 2] = matBs8[i * k + j] | matBs8[i * k + j + 1] << 4;
        auto koff = j / blocksize + i * blks;
        auto koff1 = (j + 1) / blocksize + i * blks;
        matBf32[j * n + i] = (float(matBs8[i * k + j]) - zpBs8[koff]) * scalesB[koff];
        matBf32[(j + 1) * n + i] = (float(matBs8[i * k + j + 1]) - zpBs8[koff1]) * scalesB[koff1];
      }
      for (size_t j = 0; j < k; j += blocksize * 2) {
        *(uint8_t*)&zpBs4[i * blk_padding + j / blocksize / 2] =
            zpBs8[i * blks + j / blocksize] | zpBs8[i * blks + j / blocksize + 1] << 4;
      }
    }
    launcher.mProB.packNbitsWeightQ4(n, k, isasym, (uint8_t*)matBs4.data(), k, scalesB.data(), (uint8_t*)zpBs4.data(),
                                     &packedw, &DefaultThreading);
    launcher.mProB.reduceWeight(&packedw, &DefaultThreading);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    avector<float> revB(matBf32.size());
    launcher.mProB.unpackWeight(n, k, &packedw, revB.data(), n, &DefaultThreading);
    buffer_error(matBf32.data(), revB.data(), revB.size(), FP32_ERR);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), revB.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAf32.data(), k, &rA},
        {&packedw},
        {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep(),
         isasym ? packedw.template ZPtr<int8_t>() : nullptr, rA.template RPtr<float>(), rA.lda},
        {matC.data(), n}};
    if (isasym) {
      parallel::GemmRunWithA<Parallel>(launcher, args, &DefaultThreading);
    } else {
      parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    }
    auto err = INT4_ERR;
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }

  template <class GemmCore_T>
  void ut_file(int m) {
    int n = 14336, k = 4096, blocksize = 32;
    BTLA_DTYPE qtype = BTLA_DTYPE::S4_CLIP;
    bool isasym = true;
    printf("Test Case %s: %d %d %d-%d type:%s core:%s asym:%d \n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), isasym);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher =
        wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationKBlockBaseF32,
                                      prologue_b::gemm::WeightKBlockNInteger, epilogue::gemm::CompFp32BlockEpilogue,
                                      epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    const char *qfile = "int_weight.bin", *sfile = "scales.bin", *zfile = "zeros.bin";
    auto qdata = ut::readFile2Buffer<int8_t>(qfile);
    auto sdata = readFile2Buffer<float>(sfile);
    auto zdata = readFile2Buffer<int8_t>(zfile);
    using WType = storage::gemm::StorageWeightKBlockNInteger;
    WType packedw =
        launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, isasym);

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    int blks = updiv(k, blocksize);
    avector<float> reduceA(m * blks, 0.f);

    auto rA = launcher.mProA.createStorage(m, k, blocksize);
    avector<int8_t> tmpA(rA.mSize);
    if (isasym) {
      for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
          reduceA[i * blks + j / blocksize] += matAf32[i * k + j];
        }
      }
      rA.assign(tmpA.data());
      launcher.mProA.reduce({matAf32.data(), k, &rA}, m, k, blocksize, &DefaultThreading);  // for reduce UT
      buffer_error(reduceA.data(), rA.template RPtr<float>(), reduceA.size(), FP32_ERR);
      memset(tmpA.data(), 0, tmpA.size());  // clear
    }
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(qdata[i * n + j]) - zdata[i / blocksize * n + j]) * sdata[i / blocksize * n + j];
      }
    }

    launcher.mProB.packQWeight(n, k, qdata.data(), n, sdata.data(), zdata.data(), &packedw, &DefaultThreading);

    auto bfile = readFile2Buffer<int8_t>("bestla_w3.weight.bin");
    WType packedfile(0);
    packedfile.deserialize(bfile.data());
    buffer_error(packedw.WPtr<int8_t>(), packedfile.WPtr<int8_t>(), packedw.mQBuf.size<int8_t>());
    buffer_error(packedw.SPtr<float>(), packedfile.SPtr<float>(), packedw.CSize());
    buffer_error(packedw.ZPtr<int8_t>(), packedfile.ZPtr<int8_t>(), packedw.CSize());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    avector<float> revB(matBf32.size());
    launcher.mProB.unpackWeight(n, k, &packedw, revB.data(), n, &DefaultThreading);
    buffer_error(matBf32.data(), revB.data(), revB.size(), FP32_ERR);
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), revB.data(), refCupk.data(), k, n, n);
    GemmProblem gp(1, m, n, k, blocksize);
    typename Launcher::Param args{
        gp,
        {matAf32.data(), k, &rA},
        {&packedw},
        {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep(),
         isasym ? packedw.template ZPtr<int8_t>() : nullptr, rA.template RPtr<float>(), rA.lda},
        {matC.data(), n}};
    if (isasym) {
      parallel::GemmRunWithA<Parallel>(launcher, args, &DefaultThreading);
    } else {
      parallel::GemmRun<Parallel>(launcher, args, &DefaultThreading);
    }
    auto err = INT4_ERR;
    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.001f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UT_ORT_NBits sUT_ORT_NBits;
#endif

#if 0  // TODO Add getweight fp16 
class UT_CompFp16 {
 public:
  UT_CompFp16() {
    UT_START();
    CheckISA(AVX512_FP16);
    ut_s4();
    ut_s8();
    ut_f4();
  }

  void ut_s4() {
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
  }

  void ut_s8() {
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockS8, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
    ut<sAVX512_FP16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  }

  template <class GemmCore_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype) {
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype),gemm::CoreAttr::to_str(GemmCore_T::ID), type_str<Scale_T>);
    auto constexpr ISA = GemmCore_T::ISA;
    using Launcher = wrapper::gemm::LauncherKBlock<ISA, GemmCore_T, prologue_a::gemm::ActivationBase, Wei,
                                                          epilogue::gemm::CompFp32BlockEpilogue,
                                                          epilogue::gemm::AccumulatorWriteBackFp32>;
    using Parallel = parallel::gemm::SchedulerKBlock<GemmCore_T>;
    Launcher launcher;
    blocksize = blocksize == -1 ? k : blocksize;
    using WType = typename Wei<GemmCore_T, ISA>::StorageWeight;
    WType packedw(0);
    if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockS8<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>,
                                        prologue_b::gemm::WeightKBlockS4<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    } else if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>,
                                        prologue_b::gemm::WeightKBlockF4<GemmCore_T, ISA>>) {
      packedw = launcher.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }

    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n), refCupk(m * n);
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBf32[i] = matBbf16[i];
    }
    launcher.mProB.packWeight(n, k, matBf32.data(), n, &packedw, &DefaultThreading);
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    launcher.mProB.unpackWeight(n, k, &packedw, matBf32.data(), n, &DefaultThreading);
    for (size_t i = 0; i < matBf32.size(); i++) {
      matBbf16[i] = static_cast<utils::bf16>(matBf32[i]);
    }
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refCupk.data(), k, n, n);
    typename Launcher::Param args{m,
                                  n,
                                  k,
                                  blocksize,
                                  {matAbf16.data(), k},
                                  {&packedw},
                                  {packedw.template SPtr<int8_t>(), packedw.SDtype(), packedw.CStep()},
                                  {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args);
    auto err = INT8_ERR;
    if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>, prologue_b::gemm::WeightKBlockS4<GemmCore_T, ISA>>) {
      err = INT4_ERR;
    } else if constexpr (std::is_same_v<Wei<GemmCore_T, ISA>,
                                        prologue_b::gemm::WeightKBlockF4<GemmCore_T, ISA>>) {
      err = FP4_ERR;
    }

    buffer_error(refC.data(), matC.data(), refC.size(), err);
    buffer_error(refCupk.data(), matC.data(), refCupk.size(), 0.05f);
  }
};
#ifdef BTLA_UT_DEBUG
static UT_CompFp16 sUT_CompFp16;
#endif
#endif
}  // namespace ut
}  // namespace bestla
