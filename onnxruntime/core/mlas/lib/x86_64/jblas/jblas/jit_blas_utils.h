//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <vector>
#include <cstdio>
#ifdef _WIN32
#include <cstdlib>
#else
#include <err.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>

#define fatal_error(msg, ...) err(1, "[FAIL]\t" msg, ##__VA_ARGS__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

#endif
#include "jit_blas.h"

// As long as the compiler supports the ISA, we will enable it.
// Only the ISA you use in your project will be compiled.
#ifdef __GNUC__
#define CompileAVX512F() (__GNUC__ >= 6)
#define CompileAVX2() (__GNUC__ >= 5)
#define CompileAMX() (__GNUC__ >= 11)
#define CompileBF16() (__GNUC__ >= 13)
#define CompileFP16() (__GNUC__ >= 13)
#define CompileAMXBF16() (CompileAMX())
#define CompileAMXINT8() (CompileAMX())
#else
#define CompileAVX512F() _MSC_VER && (_MSC_VER >= 1911)
#define CompileAVX2() _MSC_VER && (_MSC_VER >= 1900)
#define CompileAMX() 0
#define CompileBF16() 0
#define CompileFP16() 0
#define CompileAMXBF16() 0
#define CompileAMXINT8() 0
#endif
#if CompileBF16() || CompileFP16()
#include <immintrin.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace jblas {
namespace utils {

template <typename T2, typename T1>
inline const T2 bit_cast(T1 i) {
  static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
  T2 o;
  memcpy(&o, &i, sizeof(T2));
  return o;
}

template <typename T>
inline uint32_t bitand_u32(const T& src, const T& src1) {
  return uint32_t(src) & uint32_t(src1);
}

template <JBLAS_GEMM_CORE SRC, JBLAS_GEMM_CORE Mask, JBLAS_GEMM_CORE Shift, typename DST_T>
inline constexpr DST_T gemm_core_mask() {
  return static_cast<DST_T>((uint32_t(SRC) & uint32_t(Mask)) >> uint32_t(Shift));
}

struct bf16 {
  uint16_t x;
  union bf16f32 {
    float f32;
    unsigned int u;
    uint16_t bf16[2];
  };
  bf16() : x(0) {}

#if CompileBF16()
#pragma GCC target("avx512vl", "avx512bf16")
  explicit bf16(float vf32) : x(bit_cast<uint16_t>(_mm_cvtness_sbh(vf32))) {}
#else
  explicit bf16(float vf32) { fromfloat(vf32); }
#endif

#if CompileBF16()
#pragma GCC target("avx512vl", "avx512bf16")
  float tofloat() const { return static_cast<float>(bit_cast<__bf16>(this->x)); }
#else
  float tofloat() const {
    bf16f32 tmp = {0.f};
    tmp.bf16[1] = x;
    return tmp.f32;
  }
#endif

  operator float() const { return tofloat(); }

  static bf16 from_bin(const uint16_t x) {
    bf16 res;
    res.x = x;
    return res;
  }

  void fromfloat(float _v) {
#if CompileBF16()
    x = bit_cast<uint16_t>(_mm_cvtness_sbh(_v));
#else
    bf16f32 tmp = {0.f};
    tmp.f32 = _v;
    // See document of VCVTNEPS2BF16 in Intel® 64 and IA-32 Architectures Software Developer’s Manual Volume 2
    const auto lsb = tmp.bf16[1] & 1;
    tmp.u += 0x7fff + lsb;
    x = tmp.bf16[1];
#endif
  }
};

struct fp16 {
  uint16_t x;

  fp16() { x = 0; }
  explicit fp16(float val) { (*this) = val; }
  explicit fp16(bf16 val) { (*this) = static_cast<float>(val); }

  fp16& operator=(float val) {
#if CompileFP16()
    this->x = bit_cast<uint16_t>(static_cast<_Float16>(val));
#else
    // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t b = bit_cast<uint32_t>(val) + 0x00001000;
    const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
    // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    const uint32_t m = b & 0x007FFFFF;
    // sign : normalized : denormalized : saturate

    this->x = static_cast<uint16_t>((b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
                                    ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                                    (e > 143) * 0x7FFF);
#endif
    return *this;
  }
  explicit operator float() const {
#if CompileFP16()
    return static_cast<float>(bit_cast<_Float16>(this->x));
#else
    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15,
    // +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint32_t e = (x & 0x7C00) >> 10;  // exponent
    const uint32_t m = (x & 0x03FF) << 13;  // mantissa
    // evil log2 bit hack to count leading zeros in denormalized format
    const uint32_t v = bit_cast<uint32_t>(static_cast<float>(m)) >> 23;
    // sign : normalized : denormalized
    return bit_cast<float>((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                           ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
#endif
  }
  explicit operator bf16() const {
#if CompileBF16() && CompileFP16()
    return bf16(static_cast<float>(bit_cast<_Float16>(this->x)));
#else
    // Extract the exponent, and mantissa from the fp16 value.
    int exponent = x >> 10 & 0x1f;
    int mantissa = x & 0x3ff;

    // If the exponent is 0, the bf16 value is 0.
    if (exponent == 0) {
      return bf16();
    }
    // If the exponent is 31, the bf16 value is the sign bit plus 0x7fff.
    else if (exponent == 31) {
      bf16 res{};
      return bf16::from_bin(x | 0x7fff);
    }
    // Otherwise, the bf16 value is the sign bit plus the exponent minus 15,
    // followed by the mantissa.
    else {
      int sign = x & 0x8000;
      return bf16::from_bin(static_cast<uint16_t>(sign | (exponent + 128 - 16) << 7 | mantissa >> 3));
    }
#endif
  }
};

struct bit4x2 {
  int8_t x : 4;
  int8_t y : 4;
  bit4x2(int8_t v) : x(v), y(v) {}
  bit4x2() : x(0), y(0) {}
};

struct int4x2 : bit4x2 {
  int4x2(int8_t v) : bit4x2(v) {}
  int4x2() : bit4x2() {}
  static int8_t convert(int8_t src) {
    int32_t dst = src;
    dst = dst >= 0 ? dst + 8 : dst - 8;
    dst = dst / 16;
    dst = dst > 7 ? 7 : dst;
    dst = dst < -8 ? -8 : dst;
    return static_cast<int8_t>(dst);
  }
};

struct f4x2 : bit4x2 {
  f4x2(int8_t v) : bit4x2(v) {}
  f4x2() : bit4x2() {}
};

template <typename T>
inline constexpr JBLAS_DTYPE jblas_dtype = std::is_same_v<T, double>        ? JBLAS_DTYPE::F64
                                           : std::is_same_v<T, float>       ? JBLAS_DTYPE::F32
                                           : std::is_same_v<T, utils::bf16> ? JBLAS_DTYPE::BF16
                                           : std::is_same_v<T, utils::fp16> ? JBLAS_DTYPE::F16
                                           : std::is_same_v<T, int8_t>      ? JBLAS_DTYPE::S8
                                           : std::is_same_v<T, uint8_t>     ? JBLAS_DTYPE::U8
                                                                            : (assert(0), JBLAS_DTYPE::F32);
template <typename T>
inline constexpr const char* type_str = std::is_same_v<T, double>    ? "double"
                                        : std::is_same_v<T, float>   ? "float"
                                        : std::is_same_v<T, bf16>    ? "bf16"
                                        : std::is_same_v<T, fp16>    ? "fp16"
                                        : std::is_same_v<T, int8_t>  ? "int8_t"
                                        : std::is_same_v<T, uint8_t> ? "uint8_t"
                                                                     : (assert(0), "undef");

inline const char* dtype2str(JBLAS_DTYPE dtype) {
  switch (dtype) {
    case JBLAS_DTYPE::F64:
      return "float64";
    case JBLAS_DTYPE::F32:
      return "float32";
    case JBLAS_DTYPE::F16:
      return "float16";
    case JBLAS_DTYPE::BF16:
      return "bfloat16";
    case JBLAS_DTYPE::F8_E4M3:
      return "fp8_e4m3";
    case JBLAS_DTYPE::F8_E5M2:
      return "fp8_e5m2";
    case JBLAS_DTYPE::F8_E3M4:
      return "fp8_e3m4";
    case JBLAS_DTYPE::S8:
      return "signed_int8";
    case JBLAS_DTYPE::U8:
      return "unsigned_int8";
    case JBLAS_DTYPE::S4_CLIP:
      return "int4_clip";
    case JBLAS_DTYPE::S4_FULLRANGE:
      return "int4_fullrange";
    case JBLAS_DTYPE::F4_E2M1:
      return "fp4_e2m1";
    case JBLAS_DTYPE::F4_BNB:
      return "fp4_bitsandbytes";
    case JBLAS_DTYPE::F4_NF4:
      return "fp4_nf4";
    case JBLAS_DTYPE::S32:
      return "signed_int32";
    case JBLAS_DTYPE::U32:
      return "unsigned_int32";
    default:
      return "ErrType";
  }
}

template <JBLAS_DTYPE DT>
inline constexpr const char* dtype_str() {
  return dtype2str(DT);
}

inline const char* gemmcore2str(JBLAS_GEMM_CORE GC) {
  switch (GC) {
    case JBLAS_GEMM_CORE::AVX2_4X24:
      return "AVX2_4X24";
    case JBLAS_GEMM_CORE::AVX2_2X48:
      return "AVX2_2X48";
    case JBLAS_GEMM_CORE::AVX512F_8x48:
      return "AVX512F_8x48";
    case JBLAS_GEMM_CORE::AMX_BF16_16x64:
      return "AMX_BF16_16x64";
    case JBLAS_GEMM_CORE::AMX_BF16_16x48:
      return "AMX_BF16_16x48";
    case JBLAS_GEMM_CORE::AVX512_FP16_8x64:
      return "AVX512_FP16_8x64";
    case JBLAS_GEMM_CORE::AVX512_FP16_8x96:
      return "AVX512_FP16_8x96";
    case JBLAS_GEMM_CORE::AVX_VNNI_2x48:
      return "AVX_VNNI_2x48";
    case JBLAS_GEMM_CORE::AVX_VNNI_4x24:
      return "AVX_VNNI_4x24";
    case JBLAS_GEMM_CORE::AVX512_VNNI_8x48:
      return "AVX512_VNNI_8x48";
    case JBLAS_GEMM_CORE::AMX_INT8_16x64_US:
      return "AMX_INT8_16x64_US";
    case JBLAS_GEMM_CORE::AMX_INT8_16x64_SS:
      return "AMX_INT8_16x64_SS";
    case JBLAS_GEMM_CORE::AMX_INT8_16x48_US:
      return "AMX_INT8_16x48_US";
    case JBLAS_GEMM_CORE::AMX_INT8_16x48_SS:
      return "AMX_INT8_16x48_SS";
    default:
      return "Err_Type";
  }
}

template <JBLAS_GEMM_CORE GC>
inline constexpr const char* gemmcore_str() {
  return gemmcore2str(GC);
}

inline constexpr size_t jblas_dtype_size(const JBLAS_DTYPE t) {
  auto bits = (uint32_t)t & 0xff;
  return bits >> 3;  // bits to bytes
}

static inline int jblas_gemm_weight_element_size(JBLAS_GEMM_CORE _gemm) {
  auto PACKROW = (JBLAS_GEMM_CORE)((uint32_t)_gemm & 0xff00);
  switch (PACKROW) {
    case JBLAS_GEMM_CORE::PACKROW_1:
      return 4;
    case JBLAS_GEMM_CORE::PACKROW_4:
      return 1;
    case JBLAS_GEMM_CORE::PACKROW_2:
      return 2;
    case JBLAS_GEMM_CORE::Undef:
    default:
      return 0;
  }
}

#ifndef _WIN32
static void request_perm_xtile_data() {
  unsigned long bitmask;
  long rc;

  rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) fatal_error("XTILE_DATA request failed: %ld", rc);

  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) fatal_error("prctl(ARCH_GET_XCOMP_PERM) error: %ld", rc);

  if (bitmask & XFEATURE_MASK_XTILE) printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
}
#else
static void request_perm_xtile_data() {}
#endif

template <JBLAS_ISA ISA_T>
class isa_base {
 public:
  static bool constexpr avx = ISA_T >= JblasAVX;
  static bool constexpr avx2 = ISA_T >= JblasAVX2;
  static bool constexpr avx512f = ISA_T >= JblasAVX512F;
  static bool constexpr avx512_vnni = ISA_T >= JblasAVX512_VNNI;
  static bool constexpr avx512_fp16 = ISA_T >= JblasAVX512_FP16;
  static bool constexpr amx_bf16 = ISA_T >= JblasAMX_BF16;
  static bool constexpr amx_int8 = ISA_T >= JblasAMX_INT8;
};

static inline int padto_le(int src, int padding) { return src / padding * padding; }

static inline size_t padto_le(size_t src, int padding) { return src / size_t(padding) * size_t(padding); }

static inline int updiv(int a, int b) { return (a + b - 1) / b; }

static inline size_t updiv(size_t a, int b) { return (a + b - 1) / b; }

static inline int downdiv(int a, int b) { return a / b; }

static inline int remainsize(int pos, int size, int N) { return pos + N <= size ? N : size - pos; }

template <typename _SRCT, typename _DSTT>
static inline _DSTT cast(_SRCT _src) {
  return static_cast<_DSTT>(_src);
}

template <>
int8_t cast(float _src) {
  _src = roundf(_src);
  _src = std::min(_src, 127.f);
  _src = std::max(_src, -128.f);
  return static_cast<int8_t>(_src);
}

template <>
uint8_t cast(float _src) {
  _src += 0.5f;
  _src = std::min(_src, 255.f);
  _src = std::max(_src, 0.f);
  return static_cast<uint8_t>(_src);
}

template <>
int cast(float _src) {
  return int(roundf(_src));
}

template <>
float cast(bf16 _src) {
  return _src.tofloat();
}

template <>
bf16 cast(float _src) {
  bf16 tmp;
  tmp.fromfloat(_src);
  return tmp;
}

template <typename _T>
void serialize(int8_t*& buf, _T _val) {
  *(_T*)buf = _val;
  buf += sizeof(_T);
}

template <typename _T>
_T deserialize(int8_t*& buf) {
  auto val = *(_T*)buf;
  buf += sizeof(_T);
  return val;
}

static inline int padto(int a, int b) { return updiv(a, b) * b; }
static inline size_t padto(size_t a, int b) { return updiv(a, b) * b; }

template <typename _T>
static inline _T* amalloc(size_t _size, size_t _alignment = 64) {
  if (_size == 0) {
    return NULL;
  }
  auto psize = padto(_size * sizeof(_T), int(_alignment));
#ifdef _WIN32
  return (_T*)_aligned_malloc(psize, _alignment);
#else
  return (_T*)aligned_alloc(_alignment, psize);
#endif
}

static inline void afree(void* ptr) {
  if (ptr == NULL) {
    return;
  }
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

template <typename _T, int _Alignment = 64>
class aligned_vector {
 public:
  aligned_vector() : mRawsize(0), mPtr(nullptr), mAlignedsize(0) {}
  aligned_vector(size_t _size) { resize(_size); }
  aligned_vector(size_t _size, _T _val) {
    resize(_size);
    std::fill_n(mVec.begin(), mVec.size(), _val);
  }
  size_t size() { return mRawsize; }
  void resize(size_t size) {
    mRawsize = size;
    mAlignedsize = (mRawsize + _Alignment - 1) / _Alignment * _Alignment + _Alignment;
    if (size) {
      mVec.resize(mAlignedsize);
      auto uptr = reinterpret_cast<uint64_t>(mVec.data());
      mPtr = reinterpret_cast<_T*>((uptr + _Alignment - 1) / _Alignment * _Alignment);
    } else {
      mPtr = NULL;
    }
  }
  _T* data() const { return mPtr; }
  _T& operator[](size_t _n) noexcept { return mPtr[_n]; }

 protected:
  size_t mAlignedsize, mRawsize;
  std::vector<_T> mVec;
  _T* mPtr;
};

template <typename _T, int _Alignment = 64>
using avector = aligned_vector<_T, _Alignment>;

using milliseconds = std::chrono::milliseconds;
using nanoseconds = std::chrono::nanoseconds;
using microseconds = std::chrono::microseconds;
template <typename _DUR = std::chrono::milliseconds>
class timer {
 public:
  using sclock_t = std::chrono::steady_clock;
  using stime_point_t = std::chrono::time_point<sclock_t>;

  timer() { clear(); }

  void start() { startT = sclock_t::now(); }

  void clear() { startT = stime_point_t::min(); }

  bool null_state() { return startT == stime_point_t::min(); }

  float stop() { return static_cast<float>(std::chrono::duration_cast<_DUR>(sclock_t::now() - startT).count()); }

  stime_point_t startT;
};

template <typename T>
class minmax_statistics {
 public:
  minmax_statistics() { clear(); }

  void clear() {
    min_val = std::numeric_limits<T>::max();
    max_val = std::numeric_limits<T>::min();
    avg_val = 0;
    count = 0;
  }

  void add(T _val) {
    min_val = min_val > _val ? _val : min_val;
    max_val = max_val < _val ? _val : max_val;
    count += 1;
    avg_val = (avg_val * (count - 1) + _val) / count;
  }

  T min_val, max_val, avg_val;
  size_t count;
};

template <int _PRINT_CYCLE_MS = 100, typename _PRECISION = microseconds, typename _LOG_PRECISION = milliseconds>
class timer_statistics_logger {
 public:
  typedef timer<milliseconds> log_timer_t;
  timer_statistics_logger() {
    clear();
    log_ratio = (float)std::chrono::duration_cast<_PRECISION>(_LOG_PRECISION(1)).count();
  }

  void clear() {
    statis.clear();
    logtm.clear();
  }

  void start() {
    if (logtm.null_state()) {
      logtm.start();
    }
    tm.start();
  }

  bool stop() {
    auto elapsed = tm.stop();
    statis.add(elapsed);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  bool add(float time) {
    statis.add(time);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  const char* get_log_str() {
    sprintf(str, "Min:%.4f, Max:%.4f, Average:%.4f", min_val, max_val, avg_val);
    return str;
  }
  float min_val, max_val, avg_val;

 private:
  void record() {
    min_val = statis.min_val / log_ratio;
    max_val = statis.max_val / log_ratio;
    avg_val = statis.avg_val / log_ratio;
  }
  float log_ratio;
  char str[256];
  timer<_PRECISION> tm;
  minmax_statistics<float> statis;
  timer<milliseconds> logtm;
};
}  // namespace utils

static float fp4_bnb_dequant_fp32_LUT[] = {
    0.00000000f,        5.208333333e-03f,   0.66666667f,        1.00000000f,        0.33333333f,
    0.50000000f,        0.16666667f,        0.25000000f,        -1.f * 0.00000000f, -1.f * 5.208333333e-03f,
    -1.f * 0.66666667f, -1.f * 1.00000000f, -1.f * 0.33333333f, -1.f * 0.50000000f, -1.f * 0.16666667f,
    -1.f * 0.25000000f};

static float fp4_e2m1_dequant_fp32_LUT[] = {
    0.f,
    0.010416666666666666f,
    0.16666666666666666f,
    0.25f,
    0.333333333333333f,
    0.5f,
    0.6666666666666f,
    1.f,
    -1.f * 0.f,
    -1.f * 0.010416666666666666f,
    -1.f * 0.16666666666666666f,
    -1.f * 0.25f,
    -1.f * 0.333333333333333f,
    -1.f * 0.5f,
    -1.f * 0.6666666666666f,
    -1.f * 1.f,
};

static float nf4_dequant_fp32_LUT[] = {0.f,
                                       -0.6961928009986877f,
                                       -0.5250730514526367f,
                                       -0.39491748809814453f,
                                       -0.28444138169288635f,
                                       -0.18477343022823334f,
                                       -0.09105003625154495f,
                                       -1.f,
                                       0.07958029955625534f,
                                       0.16093020141124725f,
                                       0.24611230194568634f,
                                       0.33791524171829224f,
                                       0.44070982933044434f,
                                       0.5626170039176941f,
                                       0.7229568362236023f,
                                       1.0f};
}  // namespace jblas
