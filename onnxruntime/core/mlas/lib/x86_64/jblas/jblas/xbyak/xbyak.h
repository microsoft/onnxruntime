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
#ifndef XBYAK_XBYAK_H_
#define XBYAK_XBYAK_H_
/*!
        @file xbyak.h
        @brief Xbyak ; JIT assembler for x86(IA32)/x64 by C++
        @author herumi
        @url https://github.com/herumi/xbyak
        @note modified new BSD license
        http://opensource.org/licenses/BSD-3-Clause
*/
#if (not +0) && !defined(XBYAK_NO_OP_NAMES)  // trick to detect whether 'not' is operator or not
#define XBYAK_NO_OP_NAMES
#endif

#include <stdio.h>  // for debug print
#include <assert.h>
#include <list>
#include <string>
#include <algorithm>
#ifndef NDEBUG
#include <iostream>
#endif

// #define XBYAK_DISABLE_AVX512

#if !defined(XBYAK_USE_MMAP_ALLOCATOR) && !defined(XBYAK_DONT_USE_MMAP_ALLOCATOR)
#define XBYAK_USE_MMAP_ALLOCATOR
#endif
#if !defined(__GNUC__) || defined(__MINGW32__)
#undef XBYAK_USE_MMAP_ALLOCATOR
#endif

#ifdef __GNUC__
#define XBYAK_GNUC_PREREQ(major, minor) ((__GNUC__)*100 + (__GNUC_MINOR__) >= (major)*100 + (minor))
#else
#define XBYAK_GNUC_PREREQ(major, minor) 0
#endif

// This covers -std=(gnu|c)++(0x|11|1y), -stdlib=libc++, and modern Microsoft.
#if ((defined(_MSC_VER) && (_MSC_VER >= 1600)) || defined(_LIBCPP_VERSION) || \
     ((__cplusplus >= 201103) || defined(__GXX_EXPERIMENTAL_CXX0X__)))
#include <unordered_set>
#define XBYAK_STD_UNORDERED_SET std::unordered_set
#include <unordered_map>
#define XBYAK_STD_UNORDERED_MAP std::unordered_map
#define XBYAK_STD_UNORDERED_MULTIMAP std::unordered_multimap

/*
        Clang/llvm-gcc and ICC-EDG in 'GCC-mode' always claim to be GCC 4.2, using
        libstdcxx 20070719 (from GCC 4.2.1, the last GPL 2 version).
*/
#elif XBYAK_GNUC_PREREQ(4, 5) || (XBYAK_GNUC_PREREQ(4, 2) && __GLIBCXX__ >= 20070719) || defined(__INTEL_COMPILER) || \
    defined(__llvm__)
#include <tr1/unordered_set>
#define XBYAK_STD_UNORDERED_SET std::tr1::unordered_set
#include <tr1/unordered_map>
#define XBYAK_STD_UNORDERED_MAP std::tr1::unordered_map
#define XBYAK_STD_UNORDERED_MULTIMAP std::tr1::unordered_multimap

#elif defined(_MSC_VER) && (_MSC_VER >= 1500) && (_MSC_VER < 1600)
#include <unordered_set>
#define XBYAK_STD_UNORDERED_SET std::tr1::unordered_set
#include <unordered_map>
#define XBYAK_STD_UNORDERED_MAP std::tr1::unordered_map
#define XBYAK_STD_UNORDERED_MULTIMAP std::tr1::unordered_multimap

#else
#include <set>
#define XBYAK_STD_UNORDERED_SET std::set
#include <map>
#define XBYAK_STD_UNORDERED_MAP std::map
#define XBYAK_STD_UNORDERED_MULTIMAP std::multimap
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#ifdef _MSC_VER
#define XBYAK_TLS __declspec(thread)
#else
#define XBYAK_TLS __thread
#endif
#elif defined(__GNUC__)
#include <unistd.h>
#include <sys/mman.h>
#include <stdlib.h>
#define XBYAK_TLS __thread
#endif
#if defined(__APPLE__) && !defined(XBYAK_DONT_USE_MAP_JIT)
#define XBYAK_USE_MAP_JIT
#include <sys/sysctl.h>
#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif
#endif
#if !defined(_MSC_VER) || (_MSC_VER >= 1600)
#include <stdint.h>
#endif

// MFD_CLOEXEC defined only linux 3.17 or later.
// Android wraps the memfd_create syscall from API version 30.
#if !defined(MFD_CLOEXEC) || (defined(__ANDROID__) && __ANDROID_API__ < 30)
#undef XBYAK_USE_MEMFD
#endif

#if defined(_WIN64) || defined(__MINGW64__) || (defined(__CYGWIN__) && defined(__x86_64__))
#define XBYAK64_WIN
#elif defined(__x86_64__)
#define XBYAK64_GCC
#endif
#if !defined(XBYAK64) && !defined(XBYAK32)
#if defined(XBYAK64_GCC) || defined(XBYAK64_WIN)
#define XBYAK64
#else
#define XBYAK32
#endif
#endif

#if (__cplusplus >= 201103) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#undef XBYAK_TLS
#define XBYAK_TLS thread_local
#define XBYAK_VARIADIC_TEMPLATE
#define XBYAK_NOEXCEPT noexcept
#else
#define XBYAK_NOEXCEPT throw()
#endif

// require c++14 or later
// Visual Studio 2017 version 15.0 or later
// g++-6 or later
#if ((__cplusplus >= 201402L) && !(!defined(__clang__) && defined(__GNUC__) && (__GNUC__ <= 5))) || \
    (defined(_MSC_VER) && _MSC_VER >= 1910)
#define XBYAK_CONSTEXPR constexpr
#else
#define XBYAK_CONSTEXPR
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4514) /* remove inline function */
#pragma warning(disable : 4786) /* identifier is too long */
#pragma warning(disable : 4503) /* name is too long */
#pragma warning(disable : 4127) /* constant expresison */
#endif

// disable -Warray-bounds because it may be a bug of gcc. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104603
#if defined(__GNUC__) && !defined(__clang__)
#define XBYAK_DISABLE_WARNING_ARRAY_BOUNDS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace Xbyak {

enum {
  DEFAULT_MAX_CODE_SIZE = 4096,
  VERSION = 0x6730 /* 0xABCD = A.BC(.D) */
};

#ifndef MIE_INTEGER_TYPE_DEFINED
#define MIE_INTEGER_TYPE_DEFINED
// for backward compatibility
typedef uint64_t uint64;
typedef int64_t sint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;
#endif

#ifndef MIE_ALIGN
#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif
#ifndef MIE_PACK  // for shufps
#define MIE_PACK(x, y, z, w) ((x)*64 + (y)*16 + (z)*4 + (w))
#endif

enum {
  ERR_NONE = 0,
  ERR_BAD_ADDRESSING,
  ERR_CODE_IS_TOO_BIG,
  ERR_BAD_SCALE,
  ERR_ESP_CANT_BE_INDEX,
  ERR_BAD_COMBINATION,
  ERR_BAD_SIZE_OF_REGISTER,
  ERR_IMM_IS_TOO_BIG,
  ERR_BAD_ALIGN,
  ERR_LABEL_IS_REDEFINED,
  ERR_LABEL_IS_TOO_FAR,
  ERR_LABEL_IS_NOT_FOUND,
  ERR_CODE_ISNOT_COPYABLE,
  ERR_BAD_PARAMETER,
  ERR_CANT_PROTECT,
  ERR_CANT_USE_64BIT_DISP,
  ERR_OFFSET_IS_TOO_BIG,
  ERR_MEM_SIZE_IS_NOT_SPECIFIED,
  ERR_BAD_MEM_SIZE,
  ERR_BAD_ST_COMBINATION,
  ERR_OVER_LOCAL_LABEL,  // not used
  ERR_UNDER_LOCAL_LABEL,
  ERR_CANT_ALLOC,
  ERR_ONLY_T_NEAR_IS_SUPPORTED_IN_AUTO_GROW,
  ERR_BAD_PROTECT_MODE,
  ERR_BAD_PNUM,
  ERR_BAD_TNUM,
  ERR_BAD_VSIB_ADDRESSING,
  ERR_CANT_CONVERT,
  ERR_LABEL_ISNOT_SET_BY_L,
  ERR_LABEL_IS_ALREADY_SET_BY_L,
  ERR_BAD_LABEL_STR,
  ERR_MUNMAP,
  ERR_OPMASK_IS_ALREADY_SET,
  ERR_ROUNDING_IS_ALREADY_SET,
  ERR_K0_IS_INVALID,
  ERR_EVEX_IS_INVALID,
  ERR_SAE_IS_INVALID,
  ERR_ER_IS_INVALID,
  ERR_INVALID_BROADCAST,
  ERR_INVALID_OPMASK_WITH_MEMORY,
  ERR_INVALID_ZERO,
  ERR_INVALID_RIP_IN_AUTO_GROW,
  ERR_INVALID_MIB_ADDRESS,
  ERR_X2APIC_IS_NOT_SUPPORTED,
  ERR_NOT_SUPPORTED,
  ERR_SAME_REGS_ARE_INVALID,
  ERR_INTERNAL  // Put it at last.
};

inline const char* ConvertErrorToString(int err) {
  static const char* errTbl[] = {"none",
                                 "bad addressing",
                                 "code is too big",
                                 "bad scale",
                                 "esp can't be index",
                                 "bad combination",
                                 "bad size of register",
                                 "imm is too big",
                                 "bad align",
                                 "label is redefined",
                                 "label is too far",
                                 "label is not found",
                                 "code is not copyable",
                                 "bad parameter",
                                 "can't protect",
                                 "can't use 64bit disp(use (void*))",
                                 "offset is too big",
                                 "MEM size is not specified",
                                 "bad mem size",
                                 "bad st combination",
                                 "over local label",
                                 "under local label",
                                 "can't alloc",
                                 "T_SHORT is not supported in AutoGrow",
                                 "bad protect mode",
                                 "bad pNum",
                                 "bad tNum",
                                 "bad vsib addressing",
                                 "can't convert",
                                 "label is not set by L()",
                                 "label is already set by L()",
                                 "bad label string",
                                 "err munmap",
                                 "opmask is already set",
                                 "rounding is already set",
                                 "k0 is invalid",
                                 "evex is invalid",
                                 "sae(suppress all exceptions) is invalid",
                                 "er(embedded rounding) is invalid",
                                 "invalid broadcast",
                                 "invalid opmask with memory",
                                 "invalid zero",
                                 "invalid rip in AutoGrow",
                                 "invalid mib address",
                                 "x2APIC is not supported",
                                 "not supported",
                                 "same regs are invalid",
                                 "internal error"};
  assert(ERR_INTERNAL + 1 == sizeof(errTbl) / sizeof(*errTbl));
  return err <= ERR_INTERNAL ? errTbl[err] : "unknown err";
}

#ifdef XBYAK_NO_EXCEPTION
namespace local {

inline int& GetErrorRef() {
  static XBYAK_TLS int err = 0;
  return err;
}

inline void SetError(int err) {
  if (local::GetErrorRef()) return;  // keep the first err code
  local::GetErrorRef() = err;
}

}  // namespace local

inline void ClearError() { local::GetErrorRef() = 0; }
inline int GetError() { return Xbyak::local::GetErrorRef(); }

#define XBYAK_THROW(err)         \
  {                              \
    Xbyak::local::SetError(err); \
    return;                      \
  }
#define XBYAK_THROW_RET(err, r)  \
  {                              \
    Xbyak::local::SetError(err); \
    return r;                    \
  }

#else
class Error : public std::exception {
  int err_;

 public:
  explicit Error(int err) : err_(err) {
    if (err_ < 0 || err_ > ERR_INTERNAL) {
      err_ = ERR_INTERNAL;
    }
  }
  operator int() const { return err_; }
  const char* what() const XBYAK_NOEXCEPT { return ConvertErrorToString(err_); }
};

// dummy functions
inline void ClearError() {}
inline int GetError() { return 0; }

inline const char* ConvertErrorToString(const Error& err) { return err.what(); }

#define XBYAK_THROW(err) \
  { throw Error(err); }
#define XBYAK_THROW_RET(err, r) \
  { throw Error(err); }

#endif

inline void* AlignedMalloc(size_t size, size_t alignment) {
#ifdef __MINGW32__
  return __mingw_aligned_malloc(size, alignment);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* p;
  int ret = posix_memalign(&p, alignment, size);
  return (ret == 0) ? p : 0;
#endif
}

inline void AlignedFree(void* p) {
#ifdef __MINGW32__
  __mingw_aligned_free(p);
#elif defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

template <class To, class From>
inline const To CastTo(From p) XBYAK_NOEXCEPT {
  return (const To)(size_t)(p);
}
namespace inner {

#ifdef _WIN32
struct SystemInfo {
  SYSTEM_INFO info;
  SystemInfo() { GetSystemInfo(&info); }
};
#endif
// static const size_t ALIGN_PAGE_SIZE = 4096;
inline size_t getPageSize() {
#ifdef _WIN32
  static const SystemInfo si;
  return si.info.dwPageSize;
#elif defined(__GNUC__)
  static const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize > 0) {
    return (size_t)pageSize;
  }
#endif
  return 4096;
}

inline bool IsInDisp8(uint32_t x) { return 0xFFFFFF80 <= x || x <= 0x7F; }
inline bool IsInInt32(uint64_t x) { return ~uint64_t(0x7fffffffu) <= x || x <= 0x7FFFFFFFU; }

inline uint32_t VerifyInInt32(uint64_t x) {
#if defined(XBYAK64) && !defined(__ILP32__)
  if (!IsInInt32(x)) XBYAK_THROW_RET(ERR_OFFSET_IS_TOO_BIG, 0)
#endif
  return static_cast<uint32_t>(x);
}

enum LabelMode {
  LasIs,   // as is
  Labs,    // absolute
  LaddTop  // (addr + top) for mov(reg, label) with AutoGrow
};

}  // namespace inner

/*
        custom allocator
*/
struct Allocator {
  explicit Allocator(const std::string& = "") {}  // same interface with MmapAllocator
  virtual uint8_t* alloc(size_t size) { return reinterpret_cast<uint8_t*>(AlignedMalloc(size, inner::getPageSize())); }
  virtual void free(uint8_t* p) { AlignedFree(p); }
  virtual ~Allocator() {}
  /* override to return false if you call protect() manually */
  virtual bool useProtect() const { return true; }
};

#ifdef XBYAK_USE_MMAP_ALLOCATOR
#ifdef XBYAK_USE_MAP_JIT
namespace util {

inline int getMacOsVersionPure() {
  char buf[64];
  size_t size = sizeof(buf);
  int err = sysctlbyname("kern.osrelease", buf, &size, NULL, 0);
  if (err != 0) return 0;
  char* endp;
  int major = strtol(buf, &endp, 10);
  if (*endp != '.') return 0;
  return major;
}

inline int getMacOsVersion() {
  static const int version = getMacOsVersionPure();
  return version;
}

}  // namespace util
#endif
class MmapAllocator : public Allocator {
  struct Allocation {
    size_t size;
#if defined(XBYAK_USE_MEMFD)
    // fd_ is only used with XBYAK_USE_MEMFD. We keep the file open
    // during the lifetime of each allocation in order to support
    // checkpoint/restore by unprivileged users.
    int fd;
#endif
  };
  const std::string name_;  // only used with XBYAK_USE_MEMFD
  typedef XBYAK_STD_UNORDERED_MAP<uintptr_t, Allocation> AllocationList;
  AllocationList allocList_;

 public:
  explicit MmapAllocator(const std::string& name = "xbyak") : name_(name) {}
  uint8_t* alloc(size_t size) {
    const size_t alignedSizeM1 = inner::getPageSize() - 1;
    size = (size + alignedSizeM1) & ~alignedSizeM1;
#if defined(MAP_ANONYMOUS)
    int mode = MAP_PRIVATE | MAP_ANONYMOUS;
#elif defined(MAP_ANON)
    int mode = MAP_PRIVATE | MAP_ANON;
#else
#error "not supported"
#endif
#if defined(XBYAK_USE_MAP_JIT)
    const int mojaveVersion = 18;
    if (util::getMacOsVersion() >= mojaveVersion) mode |= MAP_JIT;
#endif
    int fd = -1;
#if defined(XBYAK_USE_MEMFD)
    fd = memfd_create(name_.c_str(), MFD_CLOEXEC);
    if (fd != -1) {
      mode = MAP_SHARED;
      if (ftruncate(fd, size) != 0) {
        close(fd);
        XBYAK_THROW_RET(ERR_CANT_ALLOC, 0)
      }
    }
#endif
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, mode, fd, 0);
    if (p == MAP_FAILED) {
      if (fd != -1) close(fd);
      XBYAK_THROW_RET(ERR_CANT_ALLOC, 0)
    }
    assert(p);
    Allocation& alloc = allocList_[(uintptr_t)p];
    alloc.size = size;
#if defined(XBYAK_USE_MEMFD)
    alloc.fd = fd;
#endif
    return (uint8_t*)p;
  }
  void free(uint8_t* p) {
    if (p == 0) return;
    AllocationList::iterator i = allocList_.find((uintptr_t)p);
    if (i == allocList_.end()) XBYAK_THROW(ERR_BAD_PARAMETER)
    if (munmap((void*)i->first, i->second.size) < 0) XBYAK_THROW(ERR_MUNMAP)
#if defined(XBYAK_USE_MEMFD)
    if (i->second.fd != -1) close(i->second.fd);
#endif
    allocList_.erase(i);
  }
};
#else
typedef Allocator MmapAllocator;
#endif

class Address;
class Reg;

class Operand {
  static const uint8_t EXT8BIT = 0x20;
  unsigned int idx_ : 6;  // 0..31 + EXT8BIT = 1 if spl/bpl/sil/dil
  unsigned int kind_ : 10;
  unsigned int bit_ : 14;

 protected:
  unsigned int zero_ : 1;
  unsigned int mask_ : 3;
  unsigned int rounding_ : 3;
  void setIdx(int idx) { idx_ = idx; }

 public:
  enum Kind {
    NONE = 0,
    MEM = 1 << 0,
    REG = 1 << 1,
    MMX = 1 << 2,
    FPU = 1 << 3,
    XMM = 1 << 4,
    YMM = 1 << 5,
    ZMM = 1 << 6,
    OPMASK = 1 << 7,
    BNDREG = 1 << 8,
    TMM = 1 << 9
  };
  enum Code {
#ifdef XBYAK64
    RAX = 0,
    RCX,
    RDX,
    RBX,
    RSP,
    RBP,
    RSI,
    RDI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    R8D = 8,
    R9D,
    R10D,
    R11D,
    R12D,
    R13D,
    R14D,
    R15D,
    R8W = 8,
    R9W,
    R10W,
    R11W,
    R12W,
    R13W,
    R14W,
    R15W,
    R8B = 8,
    R9B,
    R10B,
    R11B,
    R12B,
    R13B,
    R14B,
    R15B,
    SPL = 4,
    BPL,
    SIL,
    DIL,
#endif
    EAX = 0,
    ECX,
    EDX,
    EBX,
    ESP,
    EBP,
    ESI,
    EDI,
    AX = 0,
    CX,
    DX,
    BX,
    SP,
    BP,
    SI,
    DI,
    AL = 0,
    CL,
    DL,
    BL,
    AH,
    CH,
    DH,
    BH
  };
  XBYAK_CONSTEXPR Operand() : idx_(0), kind_(0), bit_(0), zero_(0), mask_(0), rounding_(0) {}
  XBYAK_CONSTEXPR Operand(int idx, Kind kind, int bit, bool ext8bit = 0)
      : idx_(static_cast<uint8_t>(idx | (ext8bit ? EXT8BIT : 0))),
        kind_(kind),
        bit_(bit),
        zero_(0),
        mask_(0),
        rounding_(0) {
    assert((bit_ & (bit_ - 1)) == 0);  // bit must be power of two
  }
  XBYAK_CONSTEXPR Kind getKind() const { return static_cast<Kind>(kind_); }
  XBYAK_CONSTEXPR int getIdx() const { return idx_ & (EXT8BIT - 1); }
  XBYAK_CONSTEXPR bool isNone() const { return kind_ == 0; }
  XBYAK_CONSTEXPR bool isMMX() const { return is(MMX); }
  XBYAK_CONSTEXPR bool isXMM() const { return is(XMM); }
  XBYAK_CONSTEXPR bool isYMM() const { return is(YMM); }
  XBYAK_CONSTEXPR bool isZMM() const { return is(ZMM); }
  XBYAK_CONSTEXPR bool isTMM() const { return is(TMM); }
  XBYAK_CONSTEXPR bool isXMEM() const { return is(XMM | MEM); }
  XBYAK_CONSTEXPR bool isYMEM() const { return is(YMM | MEM); }
  XBYAK_CONSTEXPR bool isZMEM() const { return is(ZMM | MEM); }
  XBYAK_CONSTEXPR bool isOPMASK() const { return is(OPMASK); }
  XBYAK_CONSTEXPR bool isBNDREG() const { return is(BNDREG); }
  XBYAK_CONSTEXPR bool isREG(int bit = 0) const { return is(REG, bit); }
  XBYAK_CONSTEXPR bool isMEM(int bit = 0) const { return is(MEM, bit); }
  XBYAK_CONSTEXPR bool isFPU() const { return is(FPU); }
  XBYAK_CONSTEXPR bool isExt8bit() const { return (idx_ & EXT8BIT) != 0; }
  XBYAK_CONSTEXPR bool isExtIdx() const { return (getIdx() & 8) != 0; }
  XBYAK_CONSTEXPR bool isExtIdx2() const { return (getIdx() & 16) != 0; }
  XBYAK_CONSTEXPR bool hasEvex() const { return isZMM() || isExtIdx2() || getOpmaskIdx() || getRounding(); }
  XBYAK_CONSTEXPR bool hasRex() const { return isExt8bit() || isREG(64) || isExtIdx(); }
  XBYAK_CONSTEXPR bool hasZero() const { return zero_; }
  XBYAK_CONSTEXPR int getOpmaskIdx() const { return mask_; }
  XBYAK_CONSTEXPR int getRounding() const { return rounding_; }
  void setKind(Kind kind) {
    if ((kind & (XMM | YMM | ZMM | TMM)) == 0) return;
    kind_ = kind;
    bit_ = kind == XMM ? 128 : kind == YMM ? 256 : kind == ZMM ? 512 : 8192;
  }
  // err if MMX/FPU/OPMASK/BNDREG
  void setBit(int bit);
  void setOpmaskIdx(int idx, bool /*ignore_idx0*/ = true) {
    if (mask_) XBYAK_THROW(ERR_OPMASK_IS_ALREADY_SET)
    mask_ = idx;
  }
  void setRounding(int idx) {
    if (rounding_) XBYAK_THROW(ERR_ROUNDING_IS_ALREADY_SET)
    rounding_ = idx;
  }
  void setZero() { zero_ = true; }
  // ah, ch, dh, bh?
  bool isHigh8bit() const {
    if (!isBit(8)) return false;
    if (isExt8bit()) return false;
    const int idx = getIdx();
    return AH <= idx && idx <= BH;
  }
  // any bit is accetable if bit == 0
  XBYAK_CONSTEXPR bool is(int kind, uint32_t bit = 0) const {
    return (kind == 0 || (kind_ & kind)) && (bit == 0 || (bit_ & bit));  // cf. you can set (8|16)
  }
  XBYAK_CONSTEXPR bool isBit(uint32_t bit) const { return (bit_ & bit) != 0; }
  XBYAK_CONSTEXPR uint32_t getBit() const { return bit_; }
  const char* toString() const {
    const int idx = getIdx();
    if (kind_ == REG) {
      if (isExt8bit()) {
        static const char* tbl[4] = {"spl", "bpl", "sil", "dil"};
        return tbl[idx - 4];
      }
      static const char* tbl[4][16] = {
          {"al", "cl", "dl", "bl", "ah", "ch", "dh", "bh", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b",
           "r15b"},
          {"ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w",
           "r15w"},
          {"eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d",
           "r15d"},
          {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14",
           "r15"},
      };
      return tbl[bit_ == 8 ? 0 : bit_ == 16 ? 1 : bit_ == 32 ? 2 : 3][idx];
    } else if (isOPMASK()) {
      static const char* tbl[8] = {"k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"};
      return tbl[idx];
    } else if (isTMM()) {
      static const char* tbl[8] = {"tmm0", "tmm1", "tmm2", "tmm3", "tmm4", "tmm5", "tmm6", "tmm7"};
      return tbl[idx];
    } else if (isZMM()) {
      static const char* tbl[32] = {"zmm0",  "zmm1",  "zmm2",  "zmm3",  "zmm4",  "zmm5",  "zmm6",  "zmm7",
                                    "zmm8",  "zmm9",  "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
                                    "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
                                    "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31"};
      return tbl[idx];
    } else if (isYMM()) {
      static const char* tbl[32] = {"ymm0",  "ymm1",  "ymm2",  "ymm3",  "ymm4",  "ymm5",  "ymm6",  "ymm7",
                                    "ymm8",  "ymm9",  "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
                                    "ymm16", "ymm17", "ymm18", "ymm19", "ymm20", "ymm21", "ymm22", "ymm23",
                                    "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31"};
      return tbl[idx];
    } else if (isXMM()) {
      static const char* tbl[32] = {"xmm0",  "xmm1",  "xmm2",  "xmm3",  "xmm4",  "xmm5",  "xmm6",  "xmm7",
                                    "xmm8",  "xmm9",  "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
                                    "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23",
                                    "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31"};
      return tbl[idx];
    } else if (isMMX()) {
      static const char* tbl[8] = {"mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"};
      return tbl[idx];
    } else if (isFPU()) {
      static const char* tbl[8] = {"st0", "st1", "st2", "st3", "st4", "st5", "st6", "st7"};
      return tbl[idx];
    } else if (isBNDREG()) {
      static const char* tbl[4] = {"bnd0", "bnd1", "bnd2", "bnd3"};
      return tbl[idx];
    }
    XBYAK_THROW_RET(ERR_INTERNAL, 0);
  }
  bool isEqualIfNotInherited(const Operand& rhs) const {
    return idx_ == rhs.idx_ && kind_ == rhs.kind_ && bit_ == rhs.bit_ && zero_ == rhs.zero_ && mask_ == rhs.mask_ &&
           rounding_ == rhs.rounding_;
  }
  bool operator==(const Operand& rhs) const;
  bool operator!=(const Operand& rhs) const { return !operator==(rhs); }
  const Address& getAddress() const;
  const Reg& getReg() const;
};

inline void Operand::setBit(int bit) {
  if (bit != 8 && bit != 16 && bit != 32 && bit != 64 && bit != 128 && bit != 256 && bit != 512 && bit != 8192)
    goto ERR;
  if (isBit(bit)) return;
  if (is(MEM | OPMASK)) {
    bit_ = bit;
    return;
  }
  if (is(REG | XMM | YMM | ZMM | TMM)) {
    int idx = getIdx();
    // err if converting ah, bh, ch, dh
    if (isREG(8) && (4 <= idx && idx < 8) && !isExt8bit()) goto ERR;
    Kind kind = REG;
    switch (bit) {
      case 8:
        if (idx >= 16) goto ERR;
#ifdef XBYAK32
        if (idx >= 4) goto ERR;
#else
        if (4 <= idx && idx < 8) idx |= EXT8BIT;
#endif
        break;
      case 16:
      case 32:
      case 64:
        if (idx >= 16) goto ERR;
        break;
      case 128:
        kind = XMM;
        break;
      case 256:
        kind = YMM;
        break;
      case 512:
        kind = ZMM;
        break;
      case 8192:
        kind = TMM;
        break;
    }
    idx_ = idx;
    kind_ = kind;
    bit_ = bit;
    if (bit >= 128) return;  // keep mask_ and rounding_
    mask_ = 0;
    rounding_ = 0;
    return;
  }
ERR:
  XBYAK_THROW(ERR_CANT_CONVERT)
}

class Label;

struct Reg8;
struct Reg16;
struct Reg32;
#ifdef XBYAK64
struct Reg64;
#endif
class Reg : public Operand {
 public:
  XBYAK_CONSTEXPR Reg() {}
  XBYAK_CONSTEXPR Reg(int idx, Kind kind, int bit = 0, bool ext8bit = false) : Operand(idx, kind, bit, ext8bit) {}
  // convert to Reg8/Reg16/Reg32/Reg64/XMM/YMM/ZMM
  Reg changeBit(int bit) const {
    Reg r(*this);
    r.setBit(bit);
    return r;
  }
  uint8_t getRexW() const { return isREG(64) ? 8 : 0; }
  uint8_t getRexR() const { return isExtIdx() ? 4 : 0; }
  uint8_t getRexX() const { return isExtIdx() ? 2 : 0; }
  uint8_t getRexB() const { return isExtIdx() ? 1 : 0; }
  uint8_t getRex(const Reg& base = Reg()) const {
    uint8_t rex = getRexW() | getRexR() | base.getRexW() | base.getRexB();
    if (rex || isExt8bit() || base.isExt8bit()) rex |= 0x40;
    return rex;
  }
  Reg8 cvt8() const;
  Reg16 cvt16() const;
  Reg32 cvt32() const;
#ifdef XBYAK64
  Reg64 cvt64() const;
#endif
};

inline const Reg& Operand::getReg() const {
  assert(!isMEM());
  return static_cast<const Reg&>(*this);
}

struct Reg8 : public Reg {
  explicit XBYAK_CONSTEXPR Reg8(int idx = 0, bool ext8bit = false) : Reg(idx, Operand::REG, 8, ext8bit) {}
};

struct Reg16 : public Reg {
  explicit XBYAK_CONSTEXPR Reg16(int idx = 0) : Reg(idx, Operand::REG, 16) {}
};

struct Mmx : public Reg {
  explicit XBYAK_CONSTEXPR Mmx(int idx = 0, Kind kind = Operand::MMX, int bit = 64) : Reg(idx, kind, bit) {}
};

struct EvexModifierRounding {
  enum { T_RN_SAE = 1, T_RD_SAE = 2, T_RU_SAE = 3, T_RZ_SAE = 4, T_SAE = 5 };
  explicit XBYAK_CONSTEXPR EvexModifierRounding(int rounding) : rounding(rounding) {}
  int rounding;
};
struct EvexModifierZero {
  XBYAK_CONSTEXPR EvexModifierZero() {}
};

struct Xmm : public Mmx {
  explicit XBYAK_CONSTEXPR Xmm(int idx = 0, Kind kind = Operand::XMM, int bit = 128) : Mmx(idx, kind, bit) {}
  XBYAK_CONSTEXPR Xmm(Kind kind, int idx) : Mmx(idx, kind, kind == XMM ? 128 : kind == YMM ? 256 : 512) {}
  Xmm operator|(const EvexModifierRounding& emr) const {
    Xmm r(*this);
    r.setRounding(emr.rounding);
    return r;
  }
  Xmm copyAndSetIdx(int idx) const {
    Xmm ret(*this);
    ret.setIdx(idx);
    return ret;
  }
  Xmm copyAndSetKind(Operand::Kind kind) const {
    Xmm ret(*this);
    ret.setKind(kind);
    return ret;
  }
};

struct Ymm : public Xmm {
  explicit XBYAK_CONSTEXPR Ymm(int idx = 0, Kind kind = Operand::YMM, int bit = 256) : Xmm(idx, kind, bit) {}
  Ymm operator|(const EvexModifierRounding& emr) const {
    Ymm r(*this);
    r.setRounding(emr.rounding);
    return r;
  }
};

struct Zmm : public Ymm {
  explicit XBYAK_CONSTEXPR Zmm(int idx = 0) : Ymm(idx, Operand::ZMM, 512) {}
  Zmm operator|(const EvexModifierRounding& emr) const {
    Zmm r(*this);
    r.setRounding(emr.rounding);
    return r;
  }
};

#ifdef XBYAK64
struct Tmm : public Reg {
  explicit XBYAK_CONSTEXPR Tmm(int idx = 0, Kind kind = Operand::TMM, int bit = 8192) : Reg(idx, kind, bit) {}
};
#endif

struct Opmask : public Reg {
  explicit XBYAK_CONSTEXPR Opmask(int idx = 0) : Reg(idx, Operand::OPMASK, 64) {}
};

struct BoundsReg : public Reg {
  explicit XBYAK_CONSTEXPR BoundsReg(int idx = 0) : Reg(idx, Operand::BNDREG, 128) {}
};

template <class T>
T operator|(const T& x, const Opmask& k) {
  T r(x);
  r.setOpmaskIdx(k.getIdx());
  return r;
}
template <class T>
T operator|(const T& x, const EvexModifierZero&) {
  T r(x);
  r.setZero();
  return r;
}
template <class T>
T operator|(const T& x, const EvexModifierRounding& emr) {
  T r(x);
  r.setRounding(emr.rounding);
  return r;
}

struct Fpu : public Reg {
  explicit XBYAK_CONSTEXPR Fpu(int idx = 0) : Reg(idx, Operand::FPU, 32) {}
};

struct Reg32e : public Reg {
  explicit XBYAK_CONSTEXPR Reg32e(int idx, int bit) : Reg(idx, Operand::REG, bit) {}
};
struct Reg32 : public Reg32e {
  explicit XBYAK_CONSTEXPR Reg32(int idx = 0) : Reg32e(idx, 32) {}
};
#ifdef XBYAK64
struct Reg64 : public Reg32e {
  explicit XBYAK_CONSTEXPR Reg64(int idx = 0) : Reg32e(idx, 64) {}
};
struct RegRip {
  int64_t disp_;
  const Label* label_;
  bool isAddr_;
  explicit XBYAK_CONSTEXPR RegRip(int64_t disp = 0, const Label* label = 0, bool isAddr = false)
      : disp_(disp), label_(label), isAddr_(isAddr) {}
  friend const RegRip operator+(const RegRip& r, int disp) { return RegRip(r.disp_ + disp, r.label_, r.isAddr_); }
  friend const RegRip operator-(const RegRip& r, int disp) { return RegRip(r.disp_ - disp, r.label_, r.isAddr_); }
  friend const RegRip operator+(const RegRip& r, int64_t disp) { return RegRip(r.disp_ + disp, r.label_, r.isAddr_); }
  friend const RegRip operator-(const RegRip& r, int64_t disp) { return RegRip(r.disp_ - disp, r.label_, r.isAddr_); }
  friend const RegRip operator+(const RegRip& r, const Label& label) {
    if (r.label_ || r.isAddr_) XBYAK_THROW_RET(ERR_BAD_ADDRESSING, RegRip());
    return RegRip(r.disp_, &label);
  }
  friend const RegRip operator+(const RegRip& r, const void* addr) {
    if (r.label_ || r.isAddr_) XBYAK_THROW_RET(ERR_BAD_ADDRESSING, RegRip());
    return RegRip(r.disp_ + (int64_t)addr, 0, true);
  }
};
#endif

inline Reg8 Reg::cvt8() const {
  Reg r = changeBit(8);
  return Reg8(r.getIdx(), r.isExt8bit());
}

inline Reg16 Reg::cvt16() const { return Reg16(changeBit(16).getIdx()); }

inline Reg32 Reg::cvt32() const { return Reg32(changeBit(32).getIdx()); }

#ifdef XBYAK64
inline Reg64 Reg::cvt64() const { return Reg64(changeBit(64).getIdx()); }
#endif

#ifndef XBYAK_DISABLE_SEGMENT
// not derived from Reg
class Segment {
  int idx_;

 public:
  enum { es, cs, ss, ds, fs, gs };
  explicit XBYAK_CONSTEXPR Segment(int idx) : idx_(idx) { assert(0 <= idx_ && idx_ < 6); }
  int getIdx() const { return idx_; }
  const char* toString() const {
    static const char tbl[][3] = {"es", "cs", "ss", "ds", "fs", "gs"};
    return tbl[idx_];
  }
};
#endif

class RegExp {
 public:
#ifdef XBYAK64
  enum { i32e = 32 | 64 };
#else
  enum { i32e = 32 };
#endif
  XBYAK_CONSTEXPR RegExp(size_t disp = 0) : scale_(0), disp_(disp) {}
  XBYAK_CONSTEXPR RegExp(const Reg& r, int scale = 1) : scale_(scale), disp_(0) {
    if (!r.isREG(i32e) && !r.is(Reg::XMM | Reg::YMM | Reg::ZMM | Reg::TMM)) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    if (scale == 0) return;
    if (scale != 1 && scale != 2 && scale != 4 && scale != 8) XBYAK_THROW(ERR_BAD_SCALE)
    if (r.getBit() >= 128 || scale != 1) {  // xmm/ymm is always index
      index_ = r;
    } else {
      base_ = r;
    }
  }
  bool isVsib(int bit = 128 | 256 | 512) const { return index_.isBit(bit); }
  RegExp optimize() const {
    RegExp exp = *this;
    // [reg * 2] => [reg + reg]
    if (index_.isBit(i32e) && !base_.getBit() && scale_ == 2) {
      exp.base_ = index_;
      exp.scale_ = 1;
    }
    return exp;
  }
  bool operator==(const RegExp& rhs) const {
    return base_ == rhs.base_ && index_ == rhs.index_ && disp_ == rhs.disp_ && scale_ == rhs.scale_;
  }
  const Reg& getBase() const { return base_; }
  const Reg& getIndex() const { return index_; }
  int getScale() const { return scale_; }
  size_t getDisp() const { return disp_; }
  XBYAK_CONSTEXPR void verify() const {
    if (base_.getBit() >= 128) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    if (index_.getBit() && index_.getBit() <= 64) {
      if (index_.getIdx() == Operand::ESP) XBYAK_THROW(ERR_ESP_CANT_BE_INDEX)
      if (base_.getBit() && base_.getBit() != index_.getBit()) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    }
  }
  friend RegExp operator+(const RegExp& a, const RegExp& b);
  friend RegExp operator-(const RegExp& e, size_t disp);
  uint8_t getRex() const {
    uint8_t rex = index_.getRexX() | base_.getRexB();
    return rex ? uint8_t(rex | 0x40) : 0;
  }

 private:
  /*
          [base_ + index_ * scale_ + disp_]
          base : Reg32e, index : Reg32e(w/o esp), Xmm, Ymm
  */
  Reg base_;
  Reg index_;
  int scale_;
  size_t disp_;
};

inline RegExp operator+(const RegExp& a, const RegExp& b) {
  if (a.index_.getBit() && b.index_.getBit()) XBYAK_THROW_RET(ERR_BAD_ADDRESSING, RegExp())
  RegExp ret = a;
  if (!ret.index_.getBit()) {
    ret.index_ = b.index_;
    ret.scale_ = b.scale_;
  }
  if (b.base_.getBit()) {
    if (ret.base_.getBit()) {
      if (ret.index_.getBit()) XBYAK_THROW_RET(ERR_BAD_ADDRESSING, RegExp())
      // base + base => base + index * 1
      ret.index_ = b.base_;
      // [reg + esp] => [esp + reg]
      if (ret.index_.getIdx() == Operand::ESP) std::swap(ret.base_, ret.index_);
      ret.scale_ = 1;
    } else {
      ret.base_ = b.base_;
    }
  }
  ret.disp_ += b.disp_;
  return ret;
}
inline RegExp operator*(const Reg& r, int scale) { return RegExp(r, scale); }
inline RegExp operator*(int scale, const Reg& r) { return r * scale; }
inline RegExp operator-(const RegExp& e, size_t disp) {
  RegExp ret = e;
  ret.disp_ -= disp;
  return ret;
}

// 2nd parameter for constructor of CodeArray(maxSize, userPtr, alloc)
void* const AutoGrow = (void*)1;           //-V566
void* const DontSetProtectRWE = (void*)2;  //-V566

class CodeArray {
  enum Type {
    USER_BUF = 1,  // use userPtr(non alignment, non protect)
    ALLOC_BUF,     // use new(alignment, protect)
    AUTO_GROW      // automatically move and grow memory if necessary
  };
  CodeArray(const CodeArray& rhs);
  void operator=(const CodeArray&);
  bool isAllocType() const { return type_ == ALLOC_BUF || type_ == AUTO_GROW; }
  struct AddrInfo {
    size_t codeOffset;  // position to write
    size_t jmpAddr;     // value to write
    int jmpSize;        // size of jmpAddr
    inner::LabelMode mode;
    AddrInfo(size_t _codeOffset, size_t _jmpAddr, int _jmpSize, inner::LabelMode _mode)
        : codeOffset(_codeOffset), jmpAddr(_jmpAddr), jmpSize(_jmpSize), mode(_mode) {}
    uint64_t getVal(const uint8_t* top) const {
      uint64_t disp = (mode == inner::LaddTop) ? jmpAddr + size_t(top)
                      : (mode == inner::LasIs) ? jmpAddr
                                               : jmpAddr - size_t(top);
      if (jmpSize == 4) disp = inner::VerifyInInt32(disp);
      return disp;
    }
  };
  typedef std::list<AddrInfo> AddrInfoList;
  AddrInfoList addrInfoList_;
  const Type type_;
#ifdef XBYAK_USE_MMAP_ALLOCATOR
  MmapAllocator defaultAllocator_;
#else
  Allocator defaultAllocator_;
#endif
  Allocator* alloc_;

 protected:
  size_t maxSize_;
  uint8_t* top_;
  size_t size_;
  bool isCalledCalcJmpAddress_;

  bool useProtect() const { return alloc_->useProtect(); }
  /*
          allocate new memory and copy old data to the new area
  */
  void growMemory() {
    const size_t newSize = (std::max<size_t>)(DEFAULT_MAX_CODE_SIZE, maxSize_ * 2);
    uint8_t* newTop = alloc_->alloc(newSize);
    if (newTop == 0) XBYAK_THROW(ERR_CANT_ALLOC)
    for (size_t i = 0; i < size_; i++) newTop[i] = top_[i];
    alloc_->free(top_);
    top_ = newTop;
    maxSize_ = newSize;
  }
  /*
          calc jmp address for AutoGrow mode
  */
  void calcJmpAddress() {
    if (isCalledCalcJmpAddress_) return;
    for (AddrInfoList::const_iterator i = addrInfoList_.begin(), ie = addrInfoList_.end(); i != ie; ++i) {
      uint64_t disp = i->getVal(top_);
      rewrite(i->codeOffset, disp, i->jmpSize);
    }
    isCalledCalcJmpAddress_ = true;
  }

 public:
  enum ProtectMode {
    PROTECT_RW = 0,   // read/write
    PROTECT_RWE = 1,  // read/write/exec
    PROTECT_RE = 2    // read/exec
  };
  explicit CodeArray(size_t maxSize, void* userPtr = 0, Allocator* allocator = 0)
      : type_(userPtr == AutoGrow                              ? AUTO_GROW
              : (userPtr == 0 || userPtr == DontSetProtectRWE) ? ALLOC_BUF
                                                               : USER_BUF),
        alloc_(allocator ? allocator : (Allocator*)&defaultAllocator_),
        maxSize_(maxSize),
        top_(type_ == USER_BUF ? reinterpret_cast<uint8_t*>(userPtr) : alloc_->alloc((std::max<size_t>)(maxSize, 1))),
        size_(0),
        isCalledCalcJmpAddress_(false) {
    if (maxSize_ > 0 && top_ == 0) XBYAK_THROW(ERR_CANT_ALLOC)
    if ((type_ == ALLOC_BUF && userPtr != DontSetProtectRWE && useProtect()) && !setProtectMode(PROTECT_RWE, false)) {
      alloc_->free(top_);
      XBYAK_THROW(ERR_CANT_PROTECT)
    }
  }
  virtual ~CodeArray() {
    if (isAllocType()) {
      if (useProtect()) setProtectModeRW(false);
      alloc_->free(top_);
    }
  }
  bool setProtectMode(ProtectMode mode, bool throwException = true) {
    bool isOK = protect(top_, maxSize_, mode);
    if (isOK) return true;
    if (throwException) XBYAK_THROW_RET(ERR_CANT_PROTECT, false)
    return false;
  }
  bool setProtectModeRE(bool throwException = true) { return setProtectMode(PROTECT_RE, throwException); }
  bool setProtectModeRW(bool throwException = true) { return setProtectMode(PROTECT_RW, throwException); }
  void resetSize() {
    size_ = 0;
    addrInfoList_.clear();
    isCalledCalcJmpAddress_ = false;
  }
  void db(int code) {
    if (size_ >= maxSize_) {
      if (type_ == AUTO_GROW) {
        growMemory();
      } else {
        XBYAK_THROW(ERR_CODE_IS_TOO_BIG)
      }
    }
    top_[size_++] = static_cast<uint8_t>(code);
  }
  void db(const uint8_t* code, size_t codeSize) {
    for (size_t i = 0; i < codeSize; i++) db(code[i]);
  }
  void db(uint64_t code, size_t codeSize) {
    if (codeSize > 8) XBYAK_THROW(ERR_BAD_PARAMETER)
    for (size_t i = 0; i < codeSize; i++) db(static_cast<uint8_t>(code >> (i * 8)));
  }
  void dw(uint32_t code) { db(code, 2); }
  void dd(uint32_t code) { db(code, 4); }
  void dq(uint64_t code) { db(code, 8); }
  const uint8_t* getCode() const { return top_; }
  template <class F>
  const F getCode() const {
    return reinterpret_cast<F>(top_);
  }
  const uint8_t* getCurr() const { return &top_[size_]; }
  template <class F>
  const F getCurr() const {
    return reinterpret_cast<F>(&top_[size_]);
  }
  size_t getSize() const { return size_; }
  void setSize(size_t size) {
    if (size > maxSize_) XBYAK_THROW(ERR_OFFSET_IS_TOO_BIG)
    size_ = size;
  }
  void dump() const {
    const uint8_t* p = getCode();
    size_t bufSize = getSize();
    size_t remain = bufSize;
    for (int i = 0; i < 4; i++) {
      size_t disp = 16;
      if (remain < 16) {
        disp = remain;
      }
      for (size_t j = 0; j < 16; j++) {
        if (j < disp) {
          printf("%02X", p[i * 16 + j]);
        }
      }
      putchar('\n');
      remain -= disp;
      if (remain == 0) {
        break;
      }
    }
  }
  /*
          @param offset [in] offset from top
          @param disp [in] offset from the next of jmp
          @param size [in] write size(1, 2, 4, 8)
  */
  void rewrite(size_t offset, uint64_t disp, size_t size) {
    assert(offset < maxSize_);
    if (size != 1 && size != 2 && size != 4 && size != 8) XBYAK_THROW(ERR_BAD_PARAMETER)
    uint8_t* const data = top_ + offset;
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<uint8_t>(disp >> (i * 8));
    }
  }
  void save(size_t offset, size_t val, int size, inner::LabelMode mode) {
    addrInfoList_.push_back(AddrInfo(offset, val, size, mode));
  }
  bool isAutoGrow() const { return type_ == AUTO_GROW; }
  bool isCalledCalcJmpAddress() const { return isCalledCalcJmpAddress_; }
  /**
          change exec permission of memory
          @param addr [in] buffer address
          @param size [in] buffer size
          @param protectMode [in] mode(RW/RWE/RE)
          @return true(success), false(failure)
  */
  static inline bool protect(const void* addr, size_t size, int protectMode) {
#if defined(_WIN32)
    const DWORD c_rw = PAGE_READWRITE;
    const DWORD c_rwe = PAGE_EXECUTE_READWRITE;
    const DWORD c_re = PAGE_EXECUTE_READ;
    DWORD mode;
#else
    const int c_rw = PROT_READ | PROT_WRITE;
    const int c_rwe = PROT_READ | PROT_WRITE | PROT_EXEC;
    const int c_re = PROT_READ | PROT_EXEC;
    int mode;
#endif
    switch (protectMode) {
      case PROTECT_RW:
        mode = c_rw;
        break;
      case PROTECT_RWE:
        mode = c_rwe;
        break;
      case PROTECT_RE:
        mode = c_re;
        break;
      default:
        return false;
    }
#if defined(_WIN32)
    DWORD oldProtect;
    return VirtualProtect(const_cast<void*>(addr), size, mode, &oldProtect) != 0;
#elif defined(__GNUC__)
    size_t pageSize = sysconf(_SC_PAGESIZE);
    size_t iaddr = reinterpret_cast<size_t>(addr);
    size_t roundAddr = iaddr & ~(pageSize - static_cast<size_t>(1));
    return mprotect(reinterpret_cast<void*>(roundAddr), size + (iaddr - roundAddr), mode) == 0;
#else
    return true;
#endif
  }
  /**
          get aligned memory pointer
          @param addr [in] address
          @param alignedSize [in] power of two
          @return aligned addr by alingedSize
  */
  static inline uint8_t* getAlignedAddress(uint8_t* addr, size_t alignedSize = 16) {
    return reinterpret_cast<uint8_t*>((reinterpret_cast<size_t>(addr) + alignedSize - 1) &
                                      ~(alignedSize - static_cast<size_t>(1)));
  }
};

class Address : public Operand {
 public:
  enum Mode { M_ModRM, M_64bitDisp, M_rip, M_ripAddr };
  XBYAK_CONSTEXPR Address(uint32_t sizeBit, bool broadcast, const RegExp& e)
      : Operand(0, MEM, sizeBit), e_(e), label_(0), mode_(M_ModRM), broadcast_(broadcast) {
    e_.verify();
  }
#ifdef XBYAK64
  explicit XBYAK_CONSTEXPR Address(size_t disp)
      : Operand(0, MEM, 64), e_(disp), label_(0), mode_(M_64bitDisp), broadcast_(false) {}
  XBYAK_CONSTEXPR Address(uint32_t sizeBit, bool broadcast, const RegRip& addr)
      : Operand(0, MEM, sizeBit),
        e_(addr.disp_),
        label_(addr.label_),
        mode_(addr.isAddr_ ? M_ripAddr : M_rip),
        broadcast_(broadcast) {}
#endif
  RegExp getRegExp(bool optimize = true) const { return optimize ? e_.optimize() : e_; }
  Mode getMode() const { return mode_; }
  bool is32bit() const { return e_.getBase().getBit() == 32 || e_.getIndex().getBit() == 32; }
  bool isOnlyDisp() const { return !e_.getBase().getBit() && !e_.getIndex().getBit(); }  // for mov eax
  size_t getDisp() const { return e_.getDisp(); }
  uint8_t getRex() const {
    if (mode_ != M_ModRM) return 0;
    return getRegExp().getRex();
  }
  bool is64bitDisp() const { return mode_ == M_64bitDisp; }  // for moffset
  bool isBroadcast() const { return broadcast_; }
  const Label* getLabel() const { return label_; }
  bool operator==(const Address& rhs) const {
    return getBit() == rhs.getBit() && e_ == rhs.e_ && label_ == rhs.label_ && mode_ == rhs.mode_ &&
           broadcast_ == rhs.broadcast_;
  }
  bool operator!=(const Address& rhs) const { return !operator==(rhs); }
  bool isVsib() const { return e_.isVsib(); }

 private:
  RegExp e_;
  const Label* label_;
  Mode mode_;
  bool broadcast_;
};

inline const Address& Operand::getAddress() const {
  assert(isMEM());
  return static_cast<const Address&>(*this);
}

inline bool Operand::operator==(const Operand& rhs) const {
  if (isMEM() && rhs.isMEM()) return this->getAddress() == rhs.getAddress();
  return isEqualIfNotInherited(rhs);
}

class AddressFrame {
  void operator=(const AddressFrame&);
  AddressFrame(const AddressFrame&);

 public:
  const uint32_t bit_;
  const bool broadcast_;
  explicit XBYAK_CONSTEXPR AddressFrame(uint32_t bit, bool broadcast = false) : bit_(bit), broadcast_(broadcast) {}
  Address operator[](const RegExp& e) const { return Address(bit_, broadcast_, e); }
  Address operator[](const void* disp) const {
    return Address(bit_, broadcast_, RegExp(reinterpret_cast<size_t>(disp)));
  }
#ifdef XBYAK64
  Address operator[](uint64_t disp) const { return Address(disp); }
  Address operator[](const RegRip& addr) const { return Address(bit_, broadcast_, addr); }
#endif
};

struct JmpLabel {
  size_t endOfJmp; /* offset from top to the end address of jmp */
  int jmpSize;
  inner::LabelMode mode;
  size_t disp;  // disp for [rip + disp]
  explicit JmpLabel(size_t endOfJmp = 0, int jmpSize = 0, inner::LabelMode mode = inner::LasIs, size_t disp = 0)
      : endOfJmp(endOfJmp), jmpSize(jmpSize), mode(mode), disp(disp) {}
};

class LabelManager;

class Label {
  mutable LabelManager* mgr;
  mutable int id;
  friend class LabelManager;

 public:
  Label() : mgr(0), id(0) {}
  Label(const Label& rhs);
  Label& operator=(const Label& rhs);
  ~Label();
  void clear() {
    mgr = 0;
    id = 0;
  }
  int getId() const { return id; }
  const uint8_t* getAddress() const;

  // backward compatibility
  static inline std::string toStr(int num) {
    char buf[16];
#if defined(_MSC_VER) && (_MSC_VER < 1900)
    _snprintf_s
#else
    snprintf
#endif
        (buf, sizeof(buf), ".%08x", num);
    return buf;
  }
};

class LabelManager {
  // for string label
  struct SlabelVal {
    size_t offset;
    SlabelVal(size_t offset) : offset(offset) {}
  };
  typedef XBYAK_STD_UNORDERED_MAP<std::string, SlabelVal> SlabelDefList;
  typedef XBYAK_STD_UNORDERED_MULTIMAP<std::string, const JmpLabel> SlabelUndefList;
  struct SlabelState {
    SlabelDefList defList;
    SlabelUndefList undefList;
  };
  typedef std::list<SlabelState> StateList;
  // for Label class
  struct ClabelVal {
    ClabelVal(size_t offset = 0) : offset(offset), refCount(1) {}
    size_t offset;
    int refCount;
  };
  typedef XBYAK_STD_UNORDERED_MAP<int, ClabelVal> ClabelDefList;
  typedef XBYAK_STD_UNORDERED_MULTIMAP<int, const JmpLabel> ClabelUndefList;
  typedef XBYAK_STD_UNORDERED_SET<Label*> LabelPtrList;

  CodeArray* base_;
  // global : stateList_.front(), local : stateList_.back()
  StateList stateList_;
  mutable int labelId_;
  ClabelDefList clabelDefList_;
  ClabelUndefList clabelUndefList_;
  LabelPtrList labelPtrList_;

  int getId(const Label& label) const {
    if (label.id == 0) label.id = labelId_++;
    return label.id;
  }
  template <class DefList, class UndefList, class T>
  void define_inner(DefList& defList, UndefList& undefList, const T& labelId, size_t addrOffset) {
    // add label
    typename DefList::value_type item(labelId, addrOffset);
    std::pair<typename DefList::iterator, bool> ret = defList.insert(item);
    if (!ret.second) XBYAK_THROW(ERR_LABEL_IS_REDEFINED)
    // search undefined label
    for (;;) {
      typename UndefList::iterator itr = undefList.find(labelId);
      if (itr == undefList.end()) break;
      const JmpLabel* jmp = &itr->second;
      const size_t offset = jmp->endOfJmp - jmp->jmpSize;
      size_t disp;
      if (jmp->mode == inner::LaddTop) {
        disp = addrOffset;
      } else if (jmp->mode == inner::Labs) {
        disp = size_t(base_->getCurr());
      } else {
        disp = addrOffset - jmp->endOfJmp + jmp->disp;
#ifdef XBYAK64
        if (jmp->jmpSize <= 4 && !inner::IsInInt32(disp)) XBYAK_THROW(ERR_OFFSET_IS_TOO_BIG)
#endif
        if (jmp->jmpSize == 1 && !inner::IsInDisp8((uint32_t)disp)) XBYAK_THROW(ERR_LABEL_IS_TOO_FAR)
      }
      if (base_->isAutoGrow()) {
        base_->save(offset, disp, jmp->jmpSize, jmp->mode);
      } else {
        base_->rewrite(offset, disp, jmp->jmpSize);
      }
      undefList.erase(itr);
    }
  }
  template <class DefList, class T>
  bool getOffset_inner(const DefList& defList, size_t* offset, const T& label) const {
    typename DefList::const_iterator i = defList.find(label);
    if (i == defList.end()) return false;
    *offset = i->second.offset;
    return true;
  }
  friend class Label;
  void incRefCount(int id, Label* label) {
    clabelDefList_[id].refCount++;
    labelPtrList_.insert(label);
  }
  void decRefCount(int id, Label* label) {
    labelPtrList_.erase(label);
    ClabelDefList::iterator i = clabelDefList_.find(id);
    if (i == clabelDefList_.end()) return;
    if (i->second.refCount == 1) {
      clabelDefList_.erase(id);
    } else {
      --i->second.refCount;
    }
  }
  template <class T>
  bool hasUndefinedLabel_inner(const T& list) const {
#ifndef NDEBUG
    for (typename T::const_iterator i = list.begin(); i != list.end(); ++i) {
      std::cerr << "undefined label:" << i->first << std::endl;
    }
#endif
    return !list.empty();
  }
  // detach all labels linked to LabelManager
  void resetLabelPtrList() {
    for (LabelPtrList::iterator i = labelPtrList_.begin(), ie = labelPtrList_.end(); i != ie; ++i) {
      (*i)->clear();
    }
    labelPtrList_.clear();
  }

 public:
  LabelManager() { reset(); }
  ~LabelManager() { resetLabelPtrList(); }
  void reset() {
    base_ = 0;
    labelId_ = 1;
    stateList_.clear();
    stateList_.push_back(SlabelState());
    stateList_.push_back(SlabelState());
    clabelDefList_.clear();
    clabelUndefList_.clear();
    resetLabelPtrList();
  }
  void enterLocal() { stateList_.push_back(SlabelState()); }
  void leaveLocal() {
    if (stateList_.size() <= 2) XBYAK_THROW(ERR_UNDER_LOCAL_LABEL)
    if (hasUndefinedLabel_inner(stateList_.back().undefList)) XBYAK_THROW(ERR_LABEL_IS_NOT_FOUND)
    stateList_.pop_back();
  }
  void set(CodeArray* base) { base_ = base; }
  void defineSlabel(std::string label) {
    if (label == "@b" || label == "@f") XBYAK_THROW(ERR_BAD_LABEL_STR)
    if (label == "@@") {
      SlabelDefList& defList = stateList_.front().defList;
      SlabelDefList::iterator i = defList.find("@f");
      if (i != defList.end()) {
        defList.erase(i);
        label = "@b";
      } else {
        i = defList.find("@b");
        if (i != defList.end()) {
          defList.erase(i);
        }
        label = "@f";
      }
    }
    SlabelState& st = *label.c_str() == '.' ? stateList_.back() : stateList_.front();
    define_inner(st.defList, st.undefList, label, base_->getSize());
  }
  void defineClabel(Label& label) {
    define_inner(clabelDefList_, clabelUndefList_, getId(label), base_->getSize());
    label.mgr = this;
    labelPtrList_.insert(&label);
  }
  void assign(Label& dst, const Label& src) {
    ClabelDefList::const_iterator i = clabelDefList_.find(src.id);
    if (i == clabelDefList_.end()) XBYAK_THROW(ERR_LABEL_ISNOT_SET_BY_L)
    define_inner(clabelDefList_, clabelUndefList_, dst.id, i->second.offset);
    dst.mgr = this;
    labelPtrList_.insert(&dst);
  }
  bool getOffset(size_t* offset, std::string& label) const {
    const SlabelDefList& defList = stateList_.front().defList;
    if (label == "@b") {
      if (defList.find("@f") != defList.end()) {
        label = "@f";
      } else if (defList.find("@b") == defList.end()) {
        XBYAK_THROW_RET(ERR_LABEL_IS_NOT_FOUND, false)
      }
    } else if (label == "@f") {
      if (defList.find("@f") != defList.end()) {
        label = "@b";
      }
    }
    const SlabelState& st = *label.c_str() == '.' ? stateList_.back() : stateList_.front();
    return getOffset_inner(st.defList, offset, label);
  }
  bool getOffset(size_t* offset, const Label& label) const {
    return getOffset_inner(clabelDefList_, offset, getId(label));
  }
  void addUndefinedLabel(const std::string& label, const JmpLabel& jmp) {
    SlabelState& st = *label.c_str() == '.' ? stateList_.back() : stateList_.front();
    st.undefList.insert(SlabelUndefList::value_type(label, jmp));
  }
  void addUndefinedLabel(const Label& label, const JmpLabel& jmp) {
    clabelUndefList_.insert(ClabelUndefList::value_type(label.id, jmp));
  }
  bool hasUndefSlabel() const {
    for (StateList::const_iterator i = stateList_.begin(), ie = stateList_.end(); i != ie; ++i) {
      if (hasUndefinedLabel_inner(i->undefList)) return true;
    }
    return false;
  }
  bool hasUndefClabel() const { return hasUndefinedLabel_inner(clabelUndefList_); }
  const uint8_t* getCode() const { return base_->getCode(); }
  bool isReady() const { return !base_->isAutoGrow() || base_->isCalledCalcJmpAddress(); }
};

inline Label::Label(const Label& rhs) {
  id = rhs.id;
  mgr = rhs.mgr;
  if (mgr) mgr->incRefCount(id, this);
}
inline Label& Label::operator=(const Label& rhs) {
  if (id) XBYAK_THROW_RET(ERR_LABEL_IS_ALREADY_SET_BY_L, *this)
  id = rhs.id;
  mgr = rhs.mgr;
  if (mgr) mgr->incRefCount(id, this);
  return *this;
}
inline Label::~Label() {
  if (id && mgr) mgr->decRefCount(id, this);
}
inline const uint8_t* Label::getAddress() const {
  if (mgr == 0 || !mgr->isReady()) return 0;
  size_t offset;
  if (!mgr->getOffset(&offset, *this)) return 0;
  return mgr->getCode() + offset;
}

typedef enum { DefaultEncoding, VexEncoding, EvexEncoding } PreferredEncoding;

class CodeGenerator : public CodeArray {
 public:
  enum LabelType {
    T_SHORT,
    T_NEAR,
    T_FAR,  // far jump
    T_AUTO  // T_SHORT if possible
  };

 private:
  CodeGenerator operator=(const CodeGenerator&);  // don't call
#ifdef XBYAK64
  enum {i32e = 32 | 64, BIT = 64};
  static const uint64_t dummyAddr = uint64_t(0x1122334455667788ull);
  typedef Reg64 NativeReg;
#else
  enum {i32e = 32, BIT = 32};
  static const size_t dummyAddr = 0x12345678;
  typedef Reg32 NativeReg;
#endif
  // (XMM, XMM|MEM)
  static inline bool isXMM_XMMorMEM(const Operand& op1, const Operand& op2) {
    return op1.isXMM() && (op2.isXMM() || op2.isMEM());
  }
  // (MMX, MMX|MEM) or (XMM, XMM|MEM)
  static inline bool isXMMorMMX_MEM(const Operand& op1, const Operand& op2) {
    return (op1.isMMX() && (op2.isMMX() || op2.isMEM())) || isXMM_XMMorMEM(op1, op2);
  }
  // (XMM, MMX|MEM)
  static inline bool isXMM_MMXorMEM(const Operand& op1, const Operand& op2) {
    return op1.isXMM() && (op2.isMMX() || op2.isMEM());
  }
  // (MMX, XMM|MEM)
  static inline bool isMMX_XMMorMEM(const Operand& op1, const Operand& op2) {
    return op1.isMMX() && (op2.isXMM() || op2.isMEM());
  }
  // (XMM, REG32|MEM)
  static inline bool isXMM_REG32orMEM(const Operand& op1, const Operand& op2) {
    return op1.isXMM() && (op2.isREG(i32e) || op2.isMEM());
  }
  // (REG32, XMM|MEM)
  static inline bool isREG32_XMMorMEM(const Operand& op1, const Operand& op2) {
    return op1.isREG(i32e) && (op2.isXMM() || op2.isMEM());
  }
  // (REG32, REG32|MEM)
  static inline bool isREG32_REG32orMEM(const Operand& op1, const Operand& op2) {
    return op1.isREG(i32e) && ((op2.isREG(i32e) && op1.getBit() == op2.getBit()) || op2.isMEM());
  }
  static inline bool isValidSSE(const Operand& op1) {
    // SSE instructions do not support XMM16 - XMM31
    return !(op1.isXMM() && op1.getIdx() >= 16);
  }
  void rex(const Operand& op1, const Operand& op2 = Operand()) {
    uint8_t rex = 0;
    const Operand *p1 = &op1, *p2 = &op2;
    if (p1->isMEM()) std::swap(p1, p2);
    if (p1->isMEM()) XBYAK_THROW(ERR_BAD_COMBINATION)
    if (p2->isMEM()) {
      const Address& addr = p2->getAddress();
      if (BIT == 64 && addr.is32bit()) db(0x67);
      rex = addr.getRex() | p1->getReg().getRex();
    } else {
      // ModRM(reg, base);
      rex = op2.getReg().getRex(op1.getReg());
    }
    // except movsx(16bit, 32/64bit)
    if ((op1.isBit(16) && !op2.isBit(i32e)) || (op2.isBit(16) && !op1.isBit(i32e))) db(0x66);
    if (rex) db(rex);
  }
  enum AVXtype {
    // low 3 bit
    T_N1 = 1,
    T_N2 = 2,
    T_N4 = 3,
    T_N8 = 4,
    T_N16 = 5,
    T_N32 = 6,
    T_NX_MASK = 7,
    //
    T_N_VL = 1 << 3,     // N * (1, 2, 4) for VL
    T_DUP = 1 << 4,      // N = (8, 32, 64)
    T_66 = 1 << 5,       // pp = 1
    T_F3 = 1 << 6,       // pp = 2
    T_F2 = T_66 | T_F3,  // pp = 3
    T_ER_R = 1 << 7,     // reg{er}
    T_0F = 1 << 8,
    T_0F38 = 1 << 9,
    T_0F3A = 1 << 10,
    T_L0 = 1 << 11,
    T_L1 = 1 << 12,
    T_W0 = 1 << 13,
    T_W1 = 1 << 14,
    T_EW0 = 1 << 15,
    T_EW1 = 1 << 16,
    T_YMM = 1 << 17,  // support YMM, ZMM
    T_EVEX = 1 << 18,
    T_ER_X = 1 << 19,       // xmm{er}
    T_ER_Y = 1 << 20,       // ymm{er}
    T_ER_Z = 1 << 21,       // zmm{er}
    T_SAE_X = 1 << 22,      // xmm{sae}
    T_SAE_Y = 1 << 23,      // ymm{sae}
    T_SAE_Z = 1 << 24,      // zmm{sae}
    T_MUST_EVEX = 1 << 25,  // contains T_EVEX
    T_B32 = 1 << 26,        // m32bcst
    T_B64 = 1 << 27,        // m64bcst
    T_B16 = T_B32 | T_B64,  // m16bcst (Be careful)
    T_M_K = 1 << 28,        // mem{k}
    T_VSIB = 1 << 29,
    T_MEM_EVEX = 1 << 30,  // use evex if mem
    T_FP16 = 1 << 31,      // avx512-fp16
    T_MAP5 = T_FP16 | T_0F,
    T_MAP6 = T_FP16 | T_0F38,
    T_XXX
  };
  // T_66 = 1, T_F3 = 2, T_F2 = 3
  uint32_t getPP(int type) const { return (type >> 5) & 3; }
  void vex(const Reg& reg, const Reg& base, const Operand* v, int type, int code, bool x = false) {
    int w = (type & T_W1) ? 1 : 0;
    bool is256 = (type & T_L1) ? true : (type & T_L0) ? false : reg.isYMM();
    bool r = reg.isExtIdx();
    bool b = base.isExtIdx();
    int idx = v ? v->getIdx() : 0;
    if ((idx | reg.getIdx() | base.getIdx()) >= 16) XBYAK_THROW(ERR_BAD_COMBINATION)
    uint32_t pp = getPP(type);
    uint32_t vvvv = (((~idx) & 15) << 3) | (is256 ? 4 : 0) | pp;
    if (!b && !x && !w && (type & T_0F)) {
      db(0xC5);
      db((r ? 0 : 0x80) | vvvv);
    } else {
      uint32_t mmmm = (type & T_0F) ? 1 : (type & T_0F38) ? 2 : (type & T_0F3A) ? 3 : 0;
      db(0xC4);
      db((r ? 0 : 0x80) | (x ? 0 : 0x40) | (b ? 0 : 0x20) | mmmm);
      db((w << 7) | vvvv);
    }
    db(code);
  }
  void verifySAE(const Reg& r, int type) const {
    if (((type & T_SAE_X) && r.isXMM()) || ((type & T_SAE_Y) && r.isYMM()) || ((type & T_SAE_Z) && r.isZMM())) return;
    XBYAK_THROW(ERR_SAE_IS_INVALID)
  }
  void verifyER(const Reg& r, int type) const {
    if ((type & T_ER_R) && r.isREG(32 | 64)) return;
    if (((type & T_ER_X) && r.isXMM()) || ((type & T_ER_Y) && r.isYMM()) || ((type & T_ER_Z) && r.isZMM())) return;
    XBYAK_THROW(ERR_ER_IS_INVALID)
  }
  // (a, b, c) contains non zero two or three values then err
  int verifyDuplicate(int a, int b, int c, int err) {
    int v = a | b | c;
    if ((a > 0 && a != v) + (b > 0 && b != v) + (c > 0 && c != v) > 0) XBYAK_THROW_RET(err, 0)
    return v;
  }
  int evex(const Reg& reg, const Reg& base, const Operand* v, int type, int code, bool x = false, bool b = false,
           int aaa = 0, uint32_t VL = 0, bool Hi16Vidx = false) {
    if (!(type & (T_EVEX | T_MUST_EVEX))) XBYAK_THROW_RET(ERR_EVEX_IS_INVALID, 0)
    int w = (type & T_EW1) ? 1 : 0;
    uint32_t mmm = (type & T_0F) ? 1 : (type & T_0F38) ? 2 : (type & T_0F3A) ? 3 : 0;
    if (type & T_FP16) mmm |= 4;
    uint32_t pp = getPP(type);
    int idx = v ? v->getIdx() : 0;
    uint32_t vvvv = ~idx;

    bool R = !reg.isExtIdx();
    bool X = x ? false : !base.isExtIdx2();
    bool B = !base.isExtIdx();
    bool Rp = !reg.isExtIdx2();
    int LL;
    int rounding =
        verifyDuplicate(reg.getRounding(), base.getRounding(), v ? v->getRounding() : 0, ERR_ROUNDING_IS_ALREADY_SET);
    int disp8N = 1;
    if (rounding) {
      if (rounding == EvexModifierRounding::T_SAE) {
        verifySAE(base, type);
        LL = 0;
      } else {
        verifyER(base, type);
        LL = rounding - 1;
      }
      b = true;
    } else {
      if (v) VL = (std::max)(VL, v->getBit());
      VL = (std::max)((std::max)(reg.getBit(), base.getBit()), VL);
      LL = (VL == 512) ? 2 : (VL == 256) ? 1 : 0;
      if (b) {
        disp8N = ((type & T_B16) == T_B16) ? 2 : (type & T_B32) ? 4 : 8;
      } else if (type & T_DUP) {
        disp8N = VL == 128 ? 8 : VL == 256 ? 32 : 64;
      } else {
        if ((type & (T_NX_MASK | T_N_VL)) == 0) {
          type |= T_N16 | T_N_VL;  // default
        }
        int low = type & T_NX_MASK;
        if (low > 0) {
          disp8N = 1 << (low - 1);
          if (type & T_N_VL) disp8N *= (VL == 512 ? 4 : VL == 256 ? 2 : 1);
        }
      }
    }
    bool Vp = !((v ? v->isExtIdx2() : 0) | Hi16Vidx);
    bool z = reg.hasZero() || base.hasZero() || (v ? v->hasZero() : false);
    if (aaa == 0)
      aaa = verifyDuplicate(base.getOpmaskIdx(), reg.getOpmaskIdx(), (v ? v->getOpmaskIdx() : 0),
                            ERR_OPMASK_IS_ALREADY_SET);
    if (aaa == 0) z = 0;  // clear T_z if mask is not set
    db(0x62);
    db((R ? 0x80 : 0) | (X ? 0x40 : 0) | (B ? 0x20 : 0) | (Rp ? 0x10 : 0) | mmm);
    db((w == 1 ? 0x80 : 0) | ((vvvv & 15) << 3) | 4 | (pp & 3));
    db((z ? 0x80 : 0) | ((LL & 3) << 5) | (b ? 0x10 : 0) | (Vp ? 8 : 0) | (aaa & 7));
    db(code);
    return disp8N;
  }
  void setModRM(int mod, int r1, int r2) { db(static_cast<uint8_t>((mod << 6) | ((r1 & 7) << 3) | (r2 & 7))); }
  void setSIB(const RegExp& e, int reg, int disp8N = 0) {
    uint64_t disp64 = e.getDisp();
#if defined(XBYAK64) && !defined(__ILP32__)
#ifdef XBYAK_OLD_DISP_CHECK
    // treat 0xffffffff as 0xffffffffffffffff
    uint64_t high = disp64 >> 32;
    if (high != 0 && high != 0xFFFFFFFF) XBYAK_THROW(ERR_OFFSET_IS_TOO_BIG)
#else
    // displacement should be a signed 32-bit value, so also check sign bit
    uint64_t high = disp64 >> 31;
    if (high != 0 && high != 0x1FFFFFFFF) XBYAK_THROW(ERR_OFFSET_IS_TOO_BIG)
#endif
#endif
    uint32_t disp = static_cast<uint32_t>(disp64);
    const Reg& base = e.getBase();
    const Reg& index = e.getIndex();
    const int baseIdx = base.getIdx();
    const int baseBit = base.getBit();
    const int indexBit = index.getBit();
    enum { mod00 = 0, mod01 = 1, mod10 = 2 };
    int mod = mod10;  // disp32
    if (!baseBit || ((baseIdx & 7) != Operand::EBP && disp == 0)) {
      mod = mod00;
    } else {
      if (disp8N == 0) {
        if (inner::IsInDisp8(disp)) {
          mod = mod01;
        }
      } else {
        // disp must be casted to signed
        uint32_t t = static_cast<uint32_t>(static_cast<int>(disp) / disp8N);
        if ((disp % disp8N) == 0 && inner::IsInDisp8(t)) {
          disp = t;
          mod = mod01;
        }
      }
    }
    const int newBaseIdx = baseBit ? (baseIdx & 7) : Operand::EBP;
    /* ModR/M = [2:3:3] = [Mod:reg/code:R/M] */
    bool hasSIB = indexBit || (baseIdx & 7) == Operand::ESP;
#ifdef XBYAK64
    if (!baseBit && !indexBit) hasSIB = true;
#endif
    if (hasSIB) {
      setModRM(mod, reg, Operand::ESP);
      /* SIB = [2:3:3] = [SS:index:base(=rm)] */
      const int idx = indexBit ? (index.getIdx() & 7) : Operand::ESP;
      const int scale = e.getScale();
      const int SS = (scale == 8) ? 3 : (scale == 4) ? 2 : (scale == 2) ? 1 : 0;
      setModRM(SS, idx, newBaseIdx);
    } else {
      setModRM(mod, reg, newBaseIdx);
    }
    if (mod == mod01) {
      db(disp);
    } else if (mod == mod10 || (mod == mod00 && !baseBit)) {
      dd(disp);
    }
  }
  LabelManager labelMgr_;
  bool isInDisp16(uint32_t x) const { return 0xFFFF8000 <= x || x <= 0x7FFF; }
  void opModR(const Reg& reg1, const Reg& reg2, int code0, int code1 = NONE, int code2 = NONE) {
    rex(reg2, reg1);
    db(code0 | (reg1.isBit(8) ? 0 : 1));
    if (code1 != NONE) db(code1);
    if (code2 != NONE) db(code2);
    setModRM(3, reg1.getIdx(), reg2.getIdx());
  }
  void opModM(const Address& addr, const Reg& reg, int code0, int code1 = NONE, int code2 = NONE, int immSize = 0) {
    if (addr.is64bitDisp()) XBYAK_THROW(ERR_CANT_USE_64BIT_DISP)
    rex(addr, reg);
    db(code0 | (reg.isBit(8) ? 0 : 1));
    if (code1 != NONE) db(code1);
    if (code2 != NONE) db(code2);
    opAddr(addr, reg.getIdx(), immSize);
  }
  void opLoadSeg(const Address& addr, const Reg& reg, int code0, int code1 = NONE) {
    if (addr.is64bitDisp()) XBYAK_THROW(ERR_CANT_USE_64BIT_DISP)
    if (reg.isBit(8)) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    rex(addr, reg);
    db(code0);
    if (code1 != NONE) db(code1);
    opAddr(addr, reg.getIdx());
  }
  void opMIB(const Address& addr, const Reg& reg, int code0, int code1) {
    if (addr.is64bitDisp()) XBYAK_THROW(ERR_CANT_USE_64BIT_DISP)
    if (addr.getMode() != Address::M_ModRM) XBYAK_THROW(ERR_INVALID_MIB_ADDRESS)
    if (BIT == 64 && addr.is32bit()) db(0x67);
    const RegExp& regExp = addr.getRegExp(false);
    uint8_t rex = regExp.getRex();
    if (rex) db(rex);
    db(code0);
    db(code1);
    setSIB(regExp, reg.getIdx());
  }
  void makeJmp(uint32_t disp, LabelType type, uint8_t shortCode, uint8_t longCode, uint8_t longPref) {
    const int shortJmpSize = 2;
    const int longHeaderSize = longPref ? 2 : 1;
    const int longJmpSize = longHeaderSize + 4;
    if (type != T_NEAR && inner::IsInDisp8(disp - shortJmpSize)) {
      db(shortCode);
      db(disp - shortJmpSize);
    } else {
      if (type == T_SHORT) XBYAK_THROW(ERR_LABEL_IS_TOO_FAR)
      if (longPref) db(longPref);
      db(longCode);
      dd(disp - longJmpSize);
    }
  }
  bool isNEAR(LabelType type) const { return type == T_NEAR || (type == T_AUTO && isDefaultJmpNEAR_); }
  template <class T>
  void opJmp(T& label, LabelType type, uint8_t shortCode, uint8_t longCode, uint8_t longPref) {
    if (type == T_FAR) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (isAutoGrow() && size_ + 16 >= maxSize_) growMemory(); /* avoid splitting code of jmp */
    size_t offset = 0;
    if (labelMgr_.getOffset(&offset, label)) { /* label exists */
      makeJmp(inner::VerifyInInt32(offset - size_), type, shortCode, longCode, longPref);
    } else {
      int jmpSize = 0;
      if (isNEAR(type)) {
        jmpSize = 4;
        if (longPref) db(longPref);
        db(longCode);
        dd(0);
      } else {
        jmpSize = 1;
        db(shortCode);
        db(0);
      }
      JmpLabel jmp(size_, jmpSize, inner::LasIs);
      labelMgr_.addUndefinedLabel(label, jmp);
    }
  }
  void opJmpAbs(const void* addr, LabelType type, uint8_t shortCode, uint8_t longCode, uint8_t longPref = 0) {
    if (type == T_FAR) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (isAutoGrow()) {
      if (!isNEAR(type)) XBYAK_THROW(ERR_ONLY_T_NEAR_IS_SUPPORTED_IN_AUTO_GROW)
      if (size_ + 16 >= maxSize_) growMemory();
      if (longPref) db(longPref);
      db(longCode);
      dd(0);
      save(size_ - 4, size_t(addr) - size_, 4, inner::Labs);
    } else {
      makeJmp(inner::VerifyInInt32(reinterpret_cast<const uint8_t*>(addr) - getCurr()), type, shortCode, longCode,
              longPref);
    }
  }
  void opJmpOp(const Operand& op, LabelType type, int ext) {
    const int bit = 16 | i32e;
    if (type == T_FAR) {
      if (!op.isMEM(bit)) XBYAK_THROW(ERR_NOT_SUPPORTED)
      opR_ModM(op, bit, ext + 1, 0xFF, NONE, NONE, false);
    } else {
      opR_ModM(op, bit, ext, 0xFF, NONE, NONE, true);
    }
  }
  // reg is reg field of ModRM
  // immSize is the size for immediate value
  // disp8N = 0(normal), disp8N = 1(force disp32), disp8N = {2, 4, 8} ; compressed displacement
  void opAddr(const Address& addr, int reg, int immSize = 0, int disp8N = 0, bool permitVisb = false) {
    if (!permitVisb && addr.isVsib()) XBYAK_THROW(ERR_BAD_VSIB_ADDRESSING)
    if (addr.getMode() == Address::M_ModRM) {
      setSIB(addr.getRegExp(), reg, disp8N);
    } else if (addr.getMode() == Address::M_rip || addr.getMode() == Address::M_ripAddr) {
      setModRM(0, reg, 5);
      if (addr.getLabel()) {  // [rip + Label]
        putL_inner(*addr.getLabel(), true, addr.getDisp() - immSize);
      } else {
        size_t disp = addr.getDisp();
        if (addr.getMode() == Address::M_ripAddr) {
          if (isAutoGrow()) XBYAK_THROW(ERR_INVALID_RIP_IN_AUTO_GROW)
          disp -= (size_t)getCurr() + 4 + immSize;
        }
        dd(inner::VerifyInInt32(disp));
      }
    }
  }
  /* preCode is for SSSE3/SSE4 */
  void opGen(const Operand& reg, const Operand& op, int code, int pref, bool isValid(const Operand&, const Operand&),
             int imm8 = NONE, int preCode = NONE) {
    if (isValid && !isValid(reg, op)) XBYAK_THROW(ERR_BAD_COMBINATION)
    if (!isValidSSE(reg) || !isValidSSE(op)) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (pref != NONE) db(pref);
    if (op.isMEM()) {
      opModM(op.getAddress(), reg.getReg(), 0x0F, preCode, code, (imm8 != NONE) ? 1 : 0);
    } else {
      opModR(reg.getReg(), op.getReg(), 0x0F, preCode, code);
    }
    if (imm8 != NONE) db(imm8);
  }
  void opMMX_IMM(const Mmx& mmx, int imm8, int code, int ext) {
    if (!isValidSSE(mmx)) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (mmx.isXMM()) db(0x66);
    opModR(Reg32(ext), mmx, 0x0F, code);
    db(imm8);
  }
  void opMMX(const Mmx& mmx, const Operand& op, int code, int pref = 0x66, int imm8 = NONE, int preCode = NONE) {
    opGen(mmx, op, code, mmx.isXMM() ? pref : NONE, isXMMorMMX_MEM, imm8, preCode);
  }
  void opMovXMM(const Operand& op1, const Operand& op2, int code, int pref) {
    if (!isValidSSE(op1) || !isValidSSE(op2)) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (pref != NONE) db(pref);
    if (op1.isXMM() && op2.isMEM()) {
      opModM(op2.getAddress(), op1.getReg(), 0x0F, code);
    } else if (op1.isMEM() && op2.isXMM()) {
      opModM(op1.getAddress(), op2.getReg(), 0x0F, code | 1);
    } else {
      XBYAK_THROW(ERR_BAD_COMBINATION)
    }
  }
  void opExt(const Operand& op, const Mmx& mmx, int code, int imm, bool hasMMX2 = false) {
    if (!isValidSSE(op) || !isValidSSE(mmx)) XBYAK_THROW(ERR_NOT_SUPPORTED)
    if (hasMMX2 && op.isREG(i32e)) { /* pextrw is special */
      if (mmx.isXMM()) db(0x66);
      opModR(op.getReg(), mmx, 0x0F, 0xC5);
      db(imm);
    } else {
      opGen(mmx, op, code, 0x66, isXMM_REG32orMEM, imm, 0x3A);
    }
  }
  void opR_ModM(const Operand& op, int bit, int ext, int code0, int code1 = NONE, int code2 = NONE,
                bool disableRex = false, int immSize = 0) {
    int opBit = op.getBit();
    if (disableRex && opBit == 64) opBit = 32;
    if (op.isREG(bit)) {
      opModR(Reg(ext, Operand::REG, opBit), op.getReg().changeBit(opBit), code0, code1, code2);
    } else if (op.isMEM()) {
      opModM(op.getAddress(), Reg(ext, Operand::REG, opBit), code0, code1, code2, immSize);
    } else {
      XBYAK_THROW(ERR_BAD_COMBINATION)
    }
  }
  void opShift(const Operand& op, int imm, int ext) {
    verifyMemHasSize(op);
    opR_ModM(op, 0, ext, (0xC0 | ((imm == 1 ? 1 : 0) << 4)), NONE, NONE, false, (imm != 1) ? 1 : 0);
    if (imm != 1) db(imm);
  }
  void opShift(const Operand& op, const Reg8& _cl, int ext) {
    if (_cl.getIdx() != Operand::CL) XBYAK_THROW(ERR_BAD_COMBINATION)
    opR_ModM(op, 0, ext, 0xD2);
  }
  void opModRM(const Operand& op1, const Operand& op2, bool condR, bool condM, int code0, int code1 = NONE,
               int code2 = NONE, int immSize = 0) {
    if (condR) {
      opModR(op1.getReg(), op2.getReg(), code0, code1, code2);
    } else if (condM) {
      opModM(op2.getAddress(), op1.getReg(), code0, code1, code2, immSize);
    } else {
      XBYAK_THROW(ERR_BAD_COMBINATION)
    }
  }
  void opShxd(const Operand& op, const Reg& reg, uint8_t imm, int code, const Reg8* _cl = 0) {
    if (_cl && _cl->getIdx() != Operand::CL) XBYAK_THROW(ERR_BAD_COMBINATION)
    opModRM(reg, op, (op.isREG(16 | i32e) && op.getBit() == reg.getBit()), op.isMEM() && (reg.isREG(16 | i32e)), 0x0F,
            code | (_cl ? 1 : 0), NONE, _cl ? 0 : 1);
    if (!_cl) db(imm);
  }
  // (REG, REG|MEM), (MEM, REG)
  void opRM_RM(const Operand& op1, const Operand& op2, int code) {
    if (op1.isREG() && op2.isMEM()) {
      opModM(op2.getAddress(), op1.getReg(), code | 2);
    } else {
      opModRM(op2, op1, op1.isREG() && op1.getKind() == op2.getKind(), op1.isMEM() && op2.isREG(), code);
    }
  }
  // (REG|MEM, IMM)
  void opRM_I(const Operand& op, uint32_t imm, int code, int ext) {
    verifyMemHasSize(op);
    uint32_t immBit = inner::IsInDisp8(imm) ? 8 : isInDisp16(imm) ? 16 : 32;
    if (op.isBit(8)) immBit = 8;
    if (op.getBit() < immBit) XBYAK_THROW(ERR_IMM_IS_TOO_BIG)
    if (op.isBit(32 | 64) && immBit == 16) immBit = 32; /* don't use MEM16 if 32/64bit mode */
    if (op.isREG() && op.getIdx() == 0 &&
        (op.getBit() == immBit || (op.isBit(64) && immBit == 32))) {  // rax, eax, ax, al
      rex(op);
      db(code | 4 | (immBit == 8 ? 0 : 1));
    } else {
      int tmp = immBit < (std::min)(op.getBit(), 32U) ? 2 : 0;
      opR_ModM(op, 0, ext, 0x80 | tmp, NONE, NONE, false, immBit / 8);
    }
    db(imm, immBit / 8);
  }
  void opIncDec(const Operand& op, int code, int ext) {
    verifyMemHasSize(op);
#ifndef XBYAK64
    if (op.isREG() && !op.isBit(8)) {
      rex(op);
      db(code | op.getIdx());
      return;
    }
#endif
    code = 0xFE;
    if (op.isREG()) {
      opModR(Reg(ext, Operand::REG, op.getBit()), op.getReg(), code);
    } else {
      opModM(op.getAddress(), Reg(ext, Operand::REG, op.getBit()), code);
    }
  }
  void opPushPop(const Operand& op, int code, int ext, int alt) {
    int bit = op.getBit();
    if (bit == 16 || bit == BIT) {
      if (bit == 16) db(0x66);
      if (op.isREG()) {
        if (op.getReg().getIdx() >= 8) db(0x41);
        db(alt | (op.getIdx() & 7));
        return;
      }
      if (op.isMEM()) {
        opModM(op.getAddress(), Reg(ext, Operand::REG, 32), code);
        return;
      }
    }
    XBYAK_THROW(ERR_BAD_COMBINATION)
  }
  void verifyMemHasSize(const Operand& op) const {
    if (op.isMEM() && op.getBit() == 0) XBYAK_THROW(ERR_MEM_SIZE_IS_NOT_SPECIFIED)
  }
  /*
          mov(r, imm) = db(imm, mov_imm(r, imm))
  */
  int mov_imm(const Reg& reg, uint64_t imm) {
    int bit = reg.getBit();
    const int idx = reg.getIdx();
    int code = 0xB0 | ((bit == 8 ? 0 : 1) << 3);
    if (bit == 64 && (imm & ~uint64_t(0xffffffffu)) == 0) {
      rex(Reg32(idx));
      bit = 32;
    } else {
      rex(reg);
      if (bit == 64 && inner::IsInInt32(imm)) {
        db(0xC7);
        code = 0xC0;
        bit = 32;
      }
    }
    db(code | (idx & 7));
    return bit / 8;
  }
  template <class T>
  void putL_inner(T& label, bool relative = false, size_t disp = 0) {
    const int jmpSize = relative ? 4 : (int)sizeof(size_t);
    if (isAutoGrow() && size_ + 16 >= maxSize_) growMemory();
    size_t offset = 0;
    if (labelMgr_.getOffset(&offset, label)) {
      if (relative) {
        db(inner::VerifyInInt32(offset + disp - size_ - jmpSize), jmpSize);
      } else if (isAutoGrow()) {
        db(uint64_t(0), jmpSize);
        save(size_ - jmpSize, offset, jmpSize, inner::LaddTop);
      } else {
        db(size_t(top_) + offset, jmpSize);
      }
      return;
    }
    db(uint64_t(0), jmpSize);
    JmpLabel jmp(size_, jmpSize, (relative ? inner::LasIs : isAutoGrow() ? inner::LaddTop : inner::Labs), disp);
    labelMgr_.addUndefinedLabel(label, jmp);
  }
  void opMovxx(const Reg& reg, const Operand& op, uint8_t code) {
    if (op.isBit(32)) XBYAK_THROW(ERR_BAD_COMBINATION)
    int w = op.isBit(16);
    bool cond = reg.isREG() && (reg.getBit() > op.getBit());
    opModRM(reg, op, cond && op.isREG(), cond && op.isMEM(), 0x0F, code | w);
  }
  void opFpuMem(const Address& addr, uint8_t m16, uint8_t m32, uint8_t m64, uint8_t ext, uint8_t m64ext) {
    if (addr.is64bitDisp()) XBYAK_THROW(ERR_CANT_USE_64BIT_DISP)
    uint8_t code = addr.isBit(16) ? m16 : addr.isBit(32) ? m32 : addr.isBit(64) ? m64 : 0;
    if (!code) XBYAK_THROW(ERR_BAD_MEM_SIZE)
    if (m64ext && addr.isBit(64)) ext = m64ext;

    rex(addr, st0);
    db(code);
    opAddr(addr, ext);
  }
  // use code1 if reg1 == st0
  // use code2 if reg1 != st0 && reg2 == st0
  void opFpuFpu(const Fpu& reg1, const Fpu& reg2, uint32_t code1, uint32_t code2) {
    uint32_t code = reg1.getIdx() == 0 ? code1 : reg2.getIdx() == 0 ? code2 : 0;
    if (!code) XBYAK_THROW(ERR_BAD_ST_COMBINATION)
    db(uint8_t(code >> 8));
    db(uint8_t(code | (reg1.getIdx() | reg2.getIdx())));
  }
  void opFpu(const Fpu& reg, uint8_t code1, uint8_t code2) {
    db(code1);
    db(code2 | reg.getIdx());
  }
  void opVex(const Reg& r, const Operand* p1, const Operand& op2, int type, int code, int imm8 = NONE) {
    if (op2.isMEM()) {
      const Address& addr = op2.getAddress();
      const RegExp& regExp = addr.getRegExp();
      const Reg& base = regExp.getBase();
      const Reg& index = regExp.getIndex();
      if (BIT == 64 && addr.is32bit()) db(0x67);
      int disp8N = 0;
      bool x = index.isExtIdx();
      if ((type & (T_MUST_EVEX | T_MEM_EVEX)) || r.hasEvex() || (p1 && p1->hasEvex()) || addr.isBroadcast() ||
          addr.getOpmaskIdx()) {
        int aaa = addr.getOpmaskIdx();
        if (aaa && !(type & T_M_K)) XBYAK_THROW(ERR_INVALID_OPMASK_WITH_MEMORY)
        bool b = false;
        if (addr.isBroadcast()) {
          if (!(type & (T_B32 | T_B64))) XBYAK_THROW(ERR_INVALID_BROADCAST)
          b = true;
        }
        int VL = regExp.isVsib() ? index.getBit() : 0;
        disp8N = evex(r, base, p1, type, code, x, b, aaa, VL, index.isExtIdx2());
      } else {
        vex(r, base, p1, type, code, x);
      }
      opAddr(addr, r.getIdx(), (imm8 != NONE) ? 1 : 0, disp8N, (type & T_VSIB) != 0);
    } else {
      const Reg& base = op2.getReg();
      if ((type & T_MUST_EVEX) || r.hasEvex() || (p1 && p1->hasEvex()) || base.hasEvex()) {
        evex(r, base, p1, type, code);
      } else {
        vex(r, base, p1, type, code);
      }
      setModRM(3, r.getIdx(), base.getIdx());
    }
    if (imm8 != NONE) db(imm8);
  }
  // (r, r, r/m) if isR_R_RM
  // (r, r/m, r)
  void opGpr(const Reg32e& r, const Operand& op1, const Operand& op2, int type, uint8_t code, bool isR_R_RM,
             int imm8 = NONE) {
    const Operand* p1 = &op1;
    const Operand* p2 = &op2;
    if (!isR_R_RM) std::swap(p1, p2);
    const unsigned int bit = r.getBit();
    if (p1->getBit() != bit || (p2->isREG() && p2->getBit() != bit)) XBYAK_THROW(ERR_BAD_COMBINATION)
    type |= (bit == 64) ? T_W1 : T_W0;
    opVex(r, p1, *p2, type, code, imm8);
  }
  void opAVX_X_X_XM(const Xmm& x1, const Operand& op1, const Operand& op2, int type, int code0, int imm8 = NONE) {
    const Xmm* x2 = static_cast<const Xmm*>(&op1);
    const Operand* op = &op2;
    if (op2.isNone()) {  // (x1, op1) -> (x1, x1, op1)
      x2 = &x1;
      op = &op1;
    }
    // (x1, x2, op)
    if (!((x1.isXMM() && x2->isXMM()) ||
          ((type & T_YMM) && ((x1.isYMM() && x2->isYMM()) || (x1.isZMM() && x2->isZMM())))))
      XBYAK_THROW(ERR_BAD_COMBINATION)
    opVex(x1, x2, *op, type, code0, imm8);
  }
  void opAVX_K_X_XM(const Opmask& k, const Xmm& x2, const Operand& op3, int type, int code0, int imm8 = NONE) {
    if (!op3.isMEM() && (x2.getKind() != op3.getKind())) XBYAK_THROW(ERR_BAD_COMBINATION)
    opVex(k, &x2, op3, type, code0, imm8);
  }
  // (x, x/m), (y, x/m256), (z, y/m)
  void checkCvt1(const Operand& x, const Operand& op) const {
    if (!op.isMEM() && !(x.is(Operand::XMM | Operand::YMM) && op.isXMM()) && !(x.isZMM() && op.isYMM()))
      XBYAK_THROW(ERR_BAD_COMBINATION)
  }
  // (x, x/m), (x, y/m256), (y, z/m)
  void checkCvt2(const Xmm& x, const Operand& op) const {
    if (!(x.isXMM() && op.is(Operand::XMM | Operand::YMM | Operand::MEM)) &&
        !(x.isYMM() && op.is(Operand::ZMM | Operand::MEM)))
      XBYAK_THROW(ERR_BAD_COMBINATION)
  }
  void opCvt(const Xmm& x, const Operand& op, int type, int code) {
    Operand::Kind kind = x.isXMM() ? (op.isBit(256) ? Operand::YMM : Operand::XMM) : Operand::ZMM;
    opVex(x.copyAndSetKind(kind), &xm0, op, type, code);
  }
  void opCvt2(const Xmm& x, const Operand& op, int type, int code) {
    checkCvt2(x, op);
    opCvt(x, op, type, code);
  }
  void opCvt3(const Xmm& x1, const Xmm& x2, const Operand& op, int type, int type64, int type32, uint8_t code) {
    if (!(x1.isXMM() && x2.isXMM() && (op.isREG(i32e) || op.isMEM()))) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    Xmm x(op.getIdx());
    const Operand* p = op.isREG() ? &x : &op;
    opVex(x1, &x2, *p, type | (op.isBit(64) ? type64 : type32), code);
  }
  // (x, x/y/xword/yword), (y, z/m)
  void checkCvt4(const Xmm& x, const Operand& op) const {
    if (!(x.isXMM() && op.is(Operand::XMM | Operand::YMM | Operand::MEM) && op.isBit(128 | 256)) &&
        !(x.isYMM() && op.is(Operand::ZMM | Operand::MEM)))
      XBYAK_THROW(ERR_BAD_COMBINATION)
  }
  // (x, x/y/z/xword/yword/zword)
  void opCvt5(const Xmm& x, const Operand& op, int type, int code) {
    if (!(x.isXMM() && op.isBit(128 | 256 | 512))) XBYAK_THROW(ERR_BAD_COMBINATION)
    Operand::Kind kind = op.isBit(128) ? Operand::XMM : op.isBit(256) ? Operand::YMM : Operand::ZMM;
    opVex(x.copyAndSetKind(kind), &xm0, op, type, code);
  }
  const Xmm& cvtIdx0(const Operand& x) const { return x.isZMM() ? zm0 : x.isYMM() ? ym0 : xm0; }
  // support (x, x/m, imm), (y, y/m, imm)
  void opAVX_X_XM_IMM(const Xmm& x, const Operand& op, int type, int code, int imm8 = NONE) {
    opAVX_X_X_XM(x, cvtIdx0(x), op, type, code, imm8);
  }
  // QQQ:need to refactor
  void opSp1(const Reg& reg, const Operand& op, uint8_t pref, uint8_t code0, uint8_t code1) {
    if (reg.isBit(8)) XBYAK_THROW(ERR_BAD_SIZE_OF_REGISTER)
    bool is16bit = reg.isREG(16) && (op.isREG(16) || op.isMEM());
    if (!is16bit && !(reg.isREG(i32e) && (op.isREG(reg.getBit()) || op.isMEM()))) XBYAK_THROW(ERR_BAD_COMBINATION)
    if (is16bit) db(0x66);
    db(pref);
    opModRM(reg.changeBit(i32e == 32 ? 32 : reg.getBit()), op, op.isREG(), true, code0, code1);
  }
  void opGather(const Xmm& x1, const Address& addr, const Xmm& x2, int type, uint8_t code, int mode) {
    const RegExp& regExp = addr.getRegExp();
    if (!regExp.isVsib(128 | 256)) XBYAK_THROW(ERR_BAD_VSIB_ADDRESSING)
    const int y_vx_y = 0;
    const int y_vy_y = 1;
    //		const int x_vy_x = 2;
    const bool isAddrYMM = regExp.getIndex().getBit() == 256;
    if (!x1.isXMM() || isAddrYMM || !x2.isXMM()) {
      bool isOK = false;
      if (mode == y_vx_y) {
        isOK = x1.isYMM() && !isAddrYMM && x2.isYMM();
      } else if (mode == y_vy_y) {
        isOK = x1.isYMM() && isAddrYMM && x2.isYMM();
      } else {  // x_vy_x
        isOK = !x1.isYMM() && isAddrYMM && !x2.isYMM();
      }
      if (!isOK) XBYAK_THROW(ERR_BAD_VSIB_ADDRESSING)
    }
    int i1 = x1.getIdx();
    int i2 = regExp.getIndex().getIdx();
    int i3 = x2.getIdx();
    if (i1 == i2 || i1 == i3 || i2 == i3) XBYAK_THROW(ERR_SAME_REGS_ARE_INVALID);
    opAVX_X_X_XM(isAddrYMM ? Ymm(i1) : x1, isAddrYMM ? Ymm(i3) : x2, addr, type, code);
  }
  enum { xx_yy_zz = 0, xx_yx_zy = 1, xx_xy_yz = 2 };
  void checkGather2(const Xmm& x1, const Reg& x2, int mode) const {
    if (x1.isXMM() && x2.isXMM()) return;
    switch (mode) {
      case xx_yy_zz:
        if ((x1.isYMM() && x2.isYMM()) || (x1.isZMM() && x2.isZMM())) return;
        break;
      case xx_yx_zy:
        if ((x1.isYMM() && x2.isXMM()) || (x1.isZMM() && x2.isYMM())) return;
        break;
      case xx_xy_yz:
        if ((x1.isXMM() && x2.isYMM()) || (x1.isYMM() && x2.isZMM())) return;
        break;
    }
    XBYAK_THROW(ERR_BAD_VSIB_ADDRESSING)
  }
  void opGather2(const Xmm& x, const Address& addr, int type, uint8_t code, int mode) {
    if (x.hasZero()) XBYAK_THROW(ERR_INVALID_ZERO)
    const RegExp& regExp = addr.getRegExp();
    checkGather2(x, regExp.getIndex(), mode);
    int maskIdx = x.getOpmaskIdx();
    if ((type & T_M_K) && addr.getOpmaskIdx()) maskIdx = addr.getOpmaskIdx();
    if (maskIdx == 0) XBYAK_THROW(ERR_K0_IS_INVALID);
    if (!(type & T_M_K) && x.getIdx() == regExp.getIndex().getIdx()) XBYAK_THROW(ERR_SAME_REGS_ARE_INVALID);
    opVex(x, 0, addr, type, code);
  }
  /*
          xx_xy_yz ; mode = true
          xx_xy_xz ; mode = false
  */
  void opVmov(const Operand& op, const Xmm& x, int type, uint8_t code, bool mode) {
    if (mode) {
      if (!op.isMEM() && !((op.isXMM() && x.isXMM()) || (op.isXMM() && x.isYMM()) || (op.isYMM() && x.isZMM())))
        XBYAK_THROW(ERR_BAD_COMBINATION)
    } else {
      if (!op.isMEM() && !op.isXMM()) XBYAK_THROW(ERR_BAD_COMBINATION)
    }
    opVex(x, 0, op, type, code);
  }
  void opGatherFetch(const Address& addr, const Xmm& x, int type, uint8_t code, Operand::Kind kind) {
    if (addr.hasZero()) XBYAK_THROW(ERR_INVALID_ZERO)
    if (addr.getRegExp().getIndex().getKind() != kind) XBYAK_THROW(ERR_BAD_VSIB_ADDRESSING)
    opVex(x, 0, addr, type, code);
  }
  void opEncoding(const Xmm& x1, const Xmm& x2, const Operand& op, int type, int code0, PreferredEncoding encoding) {
    opAVX_X_X_XM(x1, x2, op, type | orEvexIf(encoding), code0);
  }
  int orEvexIf(PreferredEncoding encoding) {
    if (encoding == DefaultEncoding) {
      encoding = defaultEncoding_;
    }
    if (encoding == EvexEncoding) {
#ifdef XBYAK_DISABLE_AVX512
      XBYAK_THROW(ERR_EVEX_IS_INVALID)
#endif
      return T_MUST_EVEX;
    }
    return 0;
  }
  void opInOut(const Reg& a, const Reg& d, uint8_t code) {
    if (a.getIdx() == Operand::AL && d.getIdx() == Operand::DX && d.getBit() == 16) {
      switch (a.getBit()) {
        case 8:
          db(code);
          return;
        case 16:
          db(0x66);
          db(code + 1);
          return;
        case 32:
          db(code + 1);
          return;
      }
    }
    XBYAK_THROW(ERR_BAD_COMBINATION)
  }
  void opInOut(const Reg& a, uint8_t code, uint8_t v) {
    if (a.getIdx() == Operand::AL) {
      switch (a.getBit()) {
        case 8:
          db(code);
          db(v);
          return;
        case 16:
          db(0x66);
          db(code + 1);
          db(v);
          return;
        case 32:
          db(code + 1);
          db(v);
          return;
      }
    }
    XBYAK_THROW(ERR_BAD_COMBINATION)
  }
#ifdef XBYAK64
  void opAMX(const Tmm& t1, const Address& addr, int type, int code0) {
    // require both base and index
    const RegExp exp = addr.getRegExp(false);
    if (exp.getBase().getBit() == 0 || exp.getIndex().getBit() == 0) XBYAK_THROW(ERR_NOT_SUPPORTED)
    opVex(t1, &tmm0, addr, type, code0);
  }
#endif
 public:
  unsigned int getVersion() const { return VERSION; }
  using CodeArray::db;
  const Mmx mm0, mm1, mm2, mm3, mm4, mm5, mm6, mm7;
  const Xmm xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
  const Ymm ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
  const Zmm zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  const Xmm &xm0, &xm1, &xm2, &xm3, &xm4, &xm5, &xm6, &xm7;
  const Ymm &ym0, &ym1, &ym2, &ym3, &ym4, &ym5, &ym6, &ym7;
  const Zmm &zm0, &zm1, &zm2, &zm3, &zm4, &zm5, &zm6, &zm7;
  const Reg32 eax, ecx, edx, ebx, esp, ebp, esi, edi;
  const Reg16 ax, cx, dx, bx, sp, bp, si, di;
  const Reg8 al, cl, dl, bl, ah, ch, dh, bh;
  const AddressFrame ptr, byte, word, dword, qword, xword, yword, zword;  // xword is same as oword of NASM
  const AddressFrame ptr_b, xword_b, yword_b, zword_b;  // broadcast such as {1to2}, {1to4}, {1to8}, {1to16}, {b}
  const Fpu st0, st1, st2, st3, st4, st5, st6, st7;
  const Opmask k0, k1, k2, k3, k4, k5, k6, k7;
  const BoundsReg bnd0, bnd1, bnd2, bnd3;
  const EvexModifierRounding T_sae, T_rn_sae, T_rd_sae, T_ru_sae,
      T_rz_sae;                // {sae}, {rn-sae}, {rd-sae}, {ru-sae}, {rz-sae}
  const EvexModifierZero T_z;  // {z}
#ifdef XBYAK64
  const Reg64 rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15;
  const Reg32 r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d;
  const Reg16 r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w;
  const Reg8 r8b, r9b, r10b, r11b, r12b, r13b, r14b, r15b;
  const Reg8 spl, bpl, sil, dil;
  const Xmm xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
  const Xmm xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23;
  const Xmm xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30, xmm31;
  const Ymm ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
  const Ymm ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23;
  const Ymm ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31;
  const Zmm zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
  const Zmm zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
  const Zmm zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
  const Tmm tmm0, tmm1, tmm2, tmm3, tmm4, tmm5, tmm6, tmm7;
  const Xmm &xm8, &xm9, &xm10, &xm11, &xm12, &xm13, &xm14, &xm15;  // for my convenience
  const Xmm &xm16, &xm17, &xm18, &xm19, &xm20, &xm21, &xm22, &xm23;
  const Xmm &xm24, &xm25, &xm26, &xm27, &xm28, &xm29, &xm30, &xm31;
  const Ymm &ym8, &ym9, &ym10, &ym11, &ym12, &ym13, &ym14, &ym15;
  const Ymm &ym16, &ym17, &ym18, &ym19, &ym20, &ym21, &ym22, &ym23;
  const Ymm &ym24, &ym25, &ym26, &ym27, &ym28, &ym29, &ym30, &ym31;
  const Zmm &zm8, &zm9, &zm10, &zm11, &zm12, &zm13, &zm14, &zm15;
  const Zmm &zm16, &zm17, &zm18, &zm19, &zm20, &zm21, &zm22, &zm23;
  const Zmm &zm24, &zm25, &zm26, &zm27, &zm28, &zm29, &zm30, &zm31;
  const RegRip rip;
#endif
#ifndef XBYAK_DISABLE_SEGMENT
  const Segment es, cs, ss, ds, fs, gs;
#endif
 private:
  bool isDefaultJmpNEAR_;
  PreferredEncoding defaultEncoding_;

 public:
  void L(const std::string& label) { labelMgr_.defineSlabel(label); }
  void L(Label& label) { labelMgr_.defineClabel(label); }
  Label L() {
    Label label;
    L(label);
    return label;
  }
  void inLocalLabel() { labelMgr_.enterLocal(); }
  void outLocalLabel() { labelMgr_.leaveLocal(); }
  /*
          assign src to dst
          require
          dst : does not used by L()
          src : used by L()
  */
  void assignL(Label& dst, const Label& src) { labelMgr_.assign(dst, src); }
  /*
          put address of label to buffer
          @note the put size is 4(32-bit), 8(64-bit)
  */
  void putL(std::string label) { putL_inner(label); }
  void putL(const Label& label) { putL_inner(label); }

  // set default type of `jmp` of undefined label to T_NEAR
  void setDefaultJmpNEAR(bool isNear) { isDefaultJmpNEAR_ = isNear; }
  void jmp(const Operand& op, LabelType type = T_AUTO) { opJmpOp(op, type, 4); }
  void jmp(std::string label, LabelType type = T_AUTO) { opJmp(label, type, 0xEB, 0xE9, 0); }
  void jmp(const char* label, LabelType type = T_AUTO) { jmp(std::string(label), type); }
  void jmp(const Label& label, LabelType type = T_AUTO) { opJmp(label, type, 0xEB, 0xE9, 0); }
  void jmp(const void* addr, LabelType type = T_AUTO) { opJmpAbs(addr, type, 0xEB, 0xE9); }

  void call(const Operand& op, LabelType type = T_AUTO) { opJmpOp(op, type, 2); }
  // call(string label), not const std::string&
  void call(std::string label) { opJmp(label, T_NEAR, 0, 0xE8, 0); }
  void call(const char* label) { call(std::string(label)); }
  void call(const Label& label) { opJmp(label, T_NEAR, 0, 0xE8, 0); }
  // call(function pointer)
#ifdef XBYAK_VARIADIC_TEMPLATE
  template <class Ret, class... Params>
  void call(Ret (*func)(Params...)) {
    call(reinterpret_cast<const void*>(func));
  }
#endif
  void call(const void* addr) { opJmpAbs(addr, T_NEAR, 0, 0xE8); }

  void test(const Operand& op, const Reg& reg) {
    opModRM(reg, op, op.isREG() && (op.getKind() == reg.getKind()), op.isMEM(), 0x84);
  }
  void test(const Operand& op, uint32_t imm) {
    verifyMemHasSize(op);
    int immSize = (std::min)(op.getBit() / 8, 4U);
    if (op.isREG() && op.getIdx() == 0) {  // al, ax, eax
      rex(op);
      db(0xA8 | (op.isBit(8) ? 0 : 1));
    } else {
      opR_ModM(op, 0, 0, 0xF6, NONE, NONE, false, immSize);
    }
    db(imm, immSize);
  }
  void imul(const Reg& reg, const Operand& op) {
    opModRM(reg, op, op.isREG() && (reg.getKind() == op.getKind()), op.isMEM(), 0x0F, 0xAF);
  }
  void imul(const Reg& reg, const Operand& op, int imm) {
    int s = inner::IsInDisp8(imm) ? 1 : 0;
    int immSize = s ? 1 : reg.isREG(16) ? 2 : 4;
    opModRM(reg, op, op.isREG() && (reg.getKind() == op.getKind()), op.isMEM(), 0x69 | (s << 1), NONE, NONE, immSize);
    db(imm, immSize);
  }
  void push(const Operand& op) { opPushPop(op, 0xFF, 6, 0x50); }
  void pop(const Operand& op) { opPushPop(op, 0x8F, 0, 0x58); }
  void push(const AddressFrame& af, uint32_t imm) {
    if (af.bit_ == 8) {
      db(0x6A);
      db(imm);
    } else if (af.bit_ == 16) {
      db(0x66);
      db(0x68);
      dw(imm);
    } else {
      db(0x68);
      dd(imm);
    }
  }
  /* use "push(word, 4)" if you want "push word 4" */
  void push(uint32_t imm) {
    if (inner::IsInDisp8(imm)) {
      push(byte, imm);
    } else {
      push(dword, imm);
    }
  }
  void mov(const Operand& reg1, const Operand& reg2) {
    const Reg* reg = 0;
    const Address* addr = 0;
    uint8_t code = 0;
    if (reg1.isREG() && reg1.getIdx() == 0 && reg2.isMEM()) {  // mov eax|ax|al, [disp]
      reg = &reg1.getReg();
      addr = &reg2.getAddress();
      code = 0xA0;
    } else if (reg1.isMEM() && reg2.isREG() && reg2.getIdx() == 0) {  // mov [disp], eax|ax|al
      reg = &reg2.getReg();
      addr = &reg1.getAddress();
      code = 0xA2;
    }
#ifdef XBYAK64
    if (addr && addr->is64bitDisp()) {
      if (code) {
        rex(*reg);
        db(reg1.isREG(8) ? 0xA0 : reg1.isREG() ? 0xA1 : reg2.isREG(8) ? 0xA2 : 0xA3);
        db(addr->getDisp(), 8);
      } else {
        XBYAK_THROW(ERR_BAD_COMBINATION)
      }
    } else
#else
    if (code && addr->isOnlyDisp()) {
      rex(*reg, *addr);
      db(code | (reg->isBit(8) ? 0 : 1));
      dd(static_cast<uint32_t>(addr->getDisp()));
    } else
#endif
    {
      opRM_RM(reg1, reg2, 0x88);
    }
  }
  void mov(const Operand& op, uint64_t imm) {
    if (op.isREG()) {
      const int size = mov_imm(op.getReg(), imm);
      db(imm, size);
    } else if (op.isMEM()) {
      verifyMemHasSize(op);
      int immSize = op.getBit() / 8;
      if (immSize <= 4) {
        int64_t s = int64_t(imm) >> (immSize * 8);
        if (s != 0 && s != -1) XBYAK_THROW(ERR_IMM_IS_TOO_BIG)
      } else {
        if (!inner::IsInInt32(imm)) XBYAK_THROW(ERR_IMM_IS_TOO_BIG)
        immSize = 4;
      }
      opModM(op.getAddress(), Reg(0, Operand::REG, op.getBit()), 0xC6, NONE, NONE, immSize);
      db(static_cast<uint32_t>(imm), immSize);
    } else {
      XBYAK_THROW(ERR_BAD_COMBINATION)
    }
  }

  // The template is used to avoid ambiguity when the 2nd argument is 0.
  // When the 2nd argument is 0 the call goes to
  // `void mov(const Operand& op, uint64_t imm)`.
  template <typename T1, typename T2>
  void mov(const T1&, const T2*) {
    T1::unexpected;
  }
  void mov(const NativeReg& reg, const Label& label) {
    mov_imm(reg, dummyAddr);
    putL(label);
  }
  void xchg(const Operand& op1, const Operand& op2) {
    const Operand *p1 = &op1, *p2 = &op2;
    if (p1->isMEM() || (p2->isREG(16 | i32e) && p2->getIdx() == 0)) {
      p1 = &op2;
      p2 = &op1;
    }
    if (p1->isMEM()) XBYAK_THROW(ERR_BAD_COMBINATION)
    if (p2->isREG() && (p1->isREG(16 | i32e) && p1->getIdx() == 0)
#ifdef XBYAK64
        && (p2->getIdx() != 0 || !p1->isREG(32))
#endif
    ) {
      rex(*p2, *p1);
      db(0x90 | (p2->getIdx() & 7));
      return;
    }
    opModRM(*p1, *p2, (p1->isREG() && p2->isREG() && (p1->getBit() == p2->getBit())), p2->isMEM(),
            0x86 | (p1->isBit(8) ? 0 : 1));
  }

#ifndef XBYAK_DISABLE_SEGMENT
  void push(const Segment& seg) {
    switch (seg.getIdx()) {
      case Segment::es:
        db(0x06);
        break;
      case Segment::cs:
        db(0x0E);
        break;
      case Segment::ss:
        db(0x16);
        break;
      case Segment::ds:
        db(0x1E);
        break;
      case Segment::fs:
        db(0x0F);
        db(0xA0);
        break;
      case Segment::gs:
        db(0x0F);
        db(0xA8);
        break;
      default:
        assert(0);
    }
  }
  void pop(const Segment& seg) {
    switch (seg.getIdx()) {
      case Segment::es:
        db(0x07);
        break;
      case Segment::cs:
        XBYAK_THROW(ERR_BAD_COMBINATION)
      case Segment::ss:
        db(0x17);
        break;
      case Segment::ds:
        db(0x1F);
        break;
      case Segment::fs:
        db(0x0F);
        db(0xA1);
        break;
      case Segment::gs:
        db(0x0F);
        db(0xA9);
        break;
      default:
        assert(0);
    }
  }
  void putSeg(const Segment& seg) {
    switch (seg.getIdx()) {
      case Segment::es:
        db(0x2E);
        break;
      case Segment::cs:
        db(0x36);
        break;
      case Segment::ss:
        db(0x3E);
        break;
      case Segment::ds:
        db(0x26);
        break;
      case Segment::fs:
        db(0x64);
        break;
      case Segment::gs:
        db(0x65);
        break;
      default:
        assert(0);
    }
  }
  void mov(const Operand& op, const Segment& seg) {
    opModRM(Reg8(seg.getIdx()), op, op.isREG(16 | i32e), op.isMEM(), 0x8C);
  }
  void mov(const Segment& seg, const Operand& op) {
    opModRM(Reg8(seg.getIdx()), op.isREG(16 | i32e) ? static_cast<const Operand&>(op.getReg().cvt32()) : op,
            op.isREG(16 | i32e), op.isMEM(), 0x8E);
  }
#endif

  enum { NONE = 256 };
  // constructor
  CodeGenerator(size_t maxSize = DEFAULT_MAX_CODE_SIZE, void* userPtr = 0, Allocator* allocator = 0)
      : CodeArray(maxSize, userPtr, allocator),
        mm0(0),
        mm1(1),
        mm2(2),
        mm3(3),
        mm4(4),
        mm5(5),
        mm6(6),
        mm7(7),
        xmm0(0),
        xmm1(1),
        xmm2(2),
        xmm3(3),
        xmm4(4),
        xmm5(5),
        xmm6(6),
        xmm7(7),
        ymm0(0),
        ymm1(1),
        ymm2(2),
        ymm3(3),
        ymm4(4),
        ymm5(5),
        ymm6(6),
        ymm7(7),
        zmm0(0),
        zmm1(1),
        zmm2(2),
        zmm3(3),
        zmm4(4),
        zmm5(5),
        zmm6(6),
        zmm7(7)
        // for my convenience
        ,
        xm0(xmm0),
        xm1(xmm1),
        xm2(xmm2),
        xm3(xmm3),
        xm4(xmm4),
        xm5(xmm5),
        xm6(xmm6),
        xm7(xmm7),
        ym0(ymm0),
        ym1(ymm1),
        ym2(ymm2),
        ym3(ymm3),
        ym4(ymm4),
        ym5(ymm5),
        ym6(ymm6),
        ym7(ymm7),
        zm0(zmm0),
        zm1(zmm1),
        zm2(zmm2),
        zm3(zmm3),
        zm4(zmm4),
        zm5(zmm5),
        zm6(zmm6),
        zm7(zmm7)

        ,
        eax(Operand::EAX),
        ecx(Operand::ECX),
        edx(Operand::EDX),
        ebx(Operand::EBX),
        esp(Operand::ESP),
        ebp(Operand::EBP),
        esi(Operand::ESI),
        edi(Operand::EDI),
        ax(Operand::AX),
        cx(Operand::CX),
        dx(Operand::DX),
        bx(Operand::BX),
        sp(Operand::SP),
        bp(Operand::BP),
        si(Operand::SI),
        di(Operand::DI),
        al(Operand::AL),
        cl(Operand::CL),
        dl(Operand::DL),
        bl(Operand::BL),
        ah(Operand::AH),
        ch(Operand::CH),
        dh(Operand::DH),
        bh(Operand::BH),
        ptr(0),
        byte(8),
        word(16),
        dword(32),
        qword(64),
        xword(128),
        yword(256),
        zword(512),
        ptr_b(0, true),
        xword_b(128, true),
        yword_b(256, true),
        zword_b(512, true),
        st0(0),
        st1(1),
        st2(2),
        st3(3),
        st4(4),
        st5(5),
        st6(6),
        st7(7),
        k0(0),
        k1(1),
        k2(2),
        k3(3),
        k4(4),
        k5(5),
        k6(6),
        k7(7),
        bnd0(0),
        bnd1(1),
        bnd2(2),
        bnd3(3),
        T_sae(EvexModifierRounding::T_SAE),
        T_rn_sae(EvexModifierRounding::T_RN_SAE),
        T_rd_sae(EvexModifierRounding::T_RD_SAE),
        T_ru_sae(EvexModifierRounding::T_RU_SAE),
        T_rz_sae(EvexModifierRounding::T_RZ_SAE),
        T_z()
#ifdef XBYAK64
        ,
        rax(Operand::RAX),
        rcx(Operand::RCX),
        rdx(Operand::RDX),
        rbx(Operand::RBX),
        rsp(Operand::RSP),
        rbp(Operand::RBP),
        rsi(Operand::RSI),
        rdi(Operand::RDI),
        r8(Operand::R8),
        r9(Operand::R9),
        r10(Operand::R10),
        r11(Operand::R11),
        r12(Operand::R12),
        r13(Operand::R13),
        r14(Operand::R14),
        r15(Operand::R15),
        r8d(8),
        r9d(9),
        r10d(10),
        r11d(11),
        r12d(12),
        r13d(13),
        r14d(14),
        r15d(15),
        r8w(8),
        r9w(9),
        r10w(10),
        r11w(11),
        r12w(12),
        r13w(13),
        r14w(14),
        r15w(15),
        r8b(8),
        r9b(9),
        r10b(10),
        r11b(11),
        r12b(12),
        r13b(13),
        r14b(14),
        r15b(15),
        spl(Operand::SPL, true),
        bpl(Operand::BPL, true),
        sil(Operand::SIL, true),
        dil(Operand::DIL, true),
        xmm8(8),
        xmm9(9),
        xmm10(10),
        xmm11(11),
        xmm12(12),
        xmm13(13),
        xmm14(14),
        xmm15(15),
        xmm16(16),
        xmm17(17),
        xmm18(18),
        xmm19(19),
        xmm20(20),
        xmm21(21),
        xmm22(22),
        xmm23(23),
        xmm24(24),
        xmm25(25),
        xmm26(26),
        xmm27(27),
        xmm28(28),
        xmm29(29),
        xmm30(30),
        xmm31(31),
        ymm8(8),
        ymm9(9),
        ymm10(10),
        ymm11(11),
        ymm12(12),
        ymm13(13),
        ymm14(14),
        ymm15(15),
        ymm16(16),
        ymm17(17),
        ymm18(18),
        ymm19(19),
        ymm20(20),
        ymm21(21),
        ymm22(22),
        ymm23(23),
        ymm24(24),
        ymm25(25),
        ymm26(26),
        ymm27(27),
        ymm28(28),
        ymm29(29),
        ymm30(30),
        ymm31(31),
        zmm8(8),
        zmm9(9),
        zmm10(10),
        zmm11(11),
        zmm12(12),
        zmm13(13),
        zmm14(14),
        zmm15(15),
        zmm16(16),
        zmm17(17),
        zmm18(18),
        zmm19(19),
        zmm20(20),
        zmm21(21),
        zmm22(22),
        zmm23(23),
        zmm24(24),
        zmm25(25),
        zmm26(26),
        zmm27(27),
        zmm28(28),
        zmm29(29),
        zmm30(30),
        zmm31(31),
        tmm0(0),
        tmm1(1),
        tmm2(2),
        tmm3(3),
        tmm4(4),
        tmm5(5),
        tmm6(6),
        tmm7(7)
        // for my convenience
        ,
        xm8(xmm8),
        xm9(xmm9),
        xm10(xmm10),
        xm11(xmm11),
        xm12(xmm12),
        xm13(xmm13),
        xm14(xmm14),
        xm15(xmm15),
        xm16(xmm16),
        xm17(xmm17),
        xm18(xmm18),
        xm19(xmm19),
        xm20(xmm20),
        xm21(xmm21),
        xm22(xmm22),
        xm23(xmm23),
        xm24(xmm24),
        xm25(xmm25),
        xm26(xmm26),
        xm27(xmm27),
        xm28(xmm28),
        xm29(xmm29),
        xm30(xmm30),
        xm31(xmm31),
        ym8(ymm8),
        ym9(ymm9),
        ym10(ymm10),
        ym11(ymm11),
        ym12(ymm12),
        ym13(ymm13),
        ym14(ymm14),
        ym15(ymm15),
        ym16(ymm16),
        ym17(ymm17),
        ym18(ymm18),
        ym19(ymm19),
        ym20(ymm20),
        ym21(ymm21),
        ym22(ymm22),
        ym23(ymm23),
        ym24(ymm24),
        ym25(ymm25),
        ym26(ymm26),
        ym27(ymm27),
        ym28(ymm28),
        ym29(ymm29),
        ym30(ymm30),
        ym31(ymm31),
        zm8(zmm8),
        zm9(zmm9),
        zm10(zmm10),
        zm11(zmm11),
        zm12(zmm12),
        zm13(zmm13),
        zm14(zmm14),
        zm15(zmm15),
        zm16(zmm16),
        zm17(zmm17),
        zm18(zmm18),
        zm19(zmm19),
        zm20(zmm20),
        zm21(zmm21),
        zm22(zmm22),
        zm23(zmm23),
        zm24(zmm24),
        zm25(zmm25),
        zm26(zmm26),
        zm27(zmm27),
        zm28(zmm28),
        zm29(zmm29),
        zm30(zmm30),
        zm31(zmm31),
        rip()
#endif
#ifndef XBYAK_DISABLE_SEGMENT
        ,
        es(Segment::es),
        cs(Segment::cs),
        ss(Segment::ss),
        ds(Segment::ds),
        fs(Segment::fs),
        gs(Segment::gs)
#endif
        ,
        isDefaultJmpNEAR_(false),
        defaultEncoding_(EvexEncoding) {
    labelMgr_.set(this);
  }
  void reset() {
    ClearError();
    resetSize();
    labelMgr_.reset();
    labelMgr_.set(this);
  }
  bool hasUndefinedLabel() const { return labelMgr_.hasUndefSlabel() || labelMgr_.hasUndefClabel(); }
  /*
          MUST call ready() to complete generating code if you use AutoGrow mode.
          It is not necessary for the other mode if hasUndefinedLabel() is true.
  */
  void ready(ProtectMode mode = PROTECT_RWE) {
    if (hasUndefinedLabel()) XBYAK_THROW(ERR_LABEL_IS_NOT_FOUND)
    if (isAutoGrow()) {
      calcJmpAddress();
      if (useProtect()) setProtectMode(mode);
    }
  }
  // set read/exec
  void readyRE() { return ready(PROTECT_RE); }
#ifdef XBYAK_TEST
  void dump(bool doClear = true) {
    CodeArray::dump();
    if (doClear) size_ = 0;
  }
#endif

#ifdef XBYAK_UNDEF_JNL
#undef jnl
#endif

  // set default encoding to select Vex or Evex
  void setDefaultEncoding(PreferredEncoding encoding) { defaultEncoding_ = encoding; }

  /*
          use single byte nop if useMultiByteNop = false
  */
  void nop(size_t size = 1, bool useMultiByteNop = true) {
    if (!useMultiByteNop) {
      for (size_t i = 0; i < size; i++) {
        db(0x90);
      }
      return;
    }
    /*
            Intel Architectures Software Developer's Manual Volume 2
            recommended multi-byte sequence of NOP instruction
            AMD and Intel seem to agree on the same sequences for up to 9 bytes:
            https://support.amd.com/TechDocs/55723_SOG_Fam_17h_Processors_3.00.pdf
    */
    static const uint8_t nopTbl[9][9] = {
        {0x90},
        {0x66, 0x90},
        {0x0F, 0x1F, 0x00},
        {0x0F, 0x1F, 0x40, 0x00},
        {0x0F, 0x1F, 0x44, 0x00, 0x00},
        {0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
        {0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
        {0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
        {0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    };
    const size_t n = sizeof(nopTbl) / sizeof(nopTbl[0]);
    while (size > 0) {
      size_t len = (std::min)(n, size);
      const uint8_t* seq = nopTbl[len - 1];
      db(seq, len);
      size -= len;
    }
  }

#ifndef XBYAK_DONT_READ_LIST
#include "xbyak_mnemonic.h"
  /*
          use single byte nop if useMultiByteNop = false
  */
  void align(size_t x = 16, bool useMultiByteNop = true) {
    if (x == 1) return;
    if (x < 1 || (x & (x - 1))) XBYAK_THROW(ERR_BAD_ALIGN)
    if (isAutoGrow()) XBYAK_THROW(ERR_BAD_ALIGN)
    size_t remain = size_t(getCurr()) % x;
    if (remain) {
      nop(x - remain, useMultiByteNop);
    }
  }
#endif
};

template <>
inline void CodeGenerator::mov(const NativeReg& reg, const char* label)  // can't use std::string
{
  assert(label);
  mov_imm(reg, dummyAddr);
  putL(label);
}

namespace util {
static const XBYAK_CONSTEXPR Mmx mm0(0), mm1(1), mm2(2), mm3(3), mm4(4), mm5(5), mm6(6), mm7(7);
static const XBYAK_CONSTEXPR Xmm xmm0(0), xmm1(1), xmm2(2), xmm3(3), xmm4(4), xmm5(5), xmm6(6), xmm7(7);
static const XBYAK_CONSTEXPR Ymm ymm0(0), ymm1(1), ymm2(2), ymm3(3), ymm4(4), ymm5(5), ymm6(6), ymm7(7);
static const XBYAK_CONSTEXPR Zmm zmm0(0), zmm1(1), zmm2(2), zmm3(3), zmm4(4), zmm5(5), zmm6(6), zmm7(7);
static const XBYAK_CONSTEXPR Reg32 eax(Operand::EAX), ecx(Operand::ECX), edx(Operand::EDX), ebx(Operand::EBX),
    esp(Operand::ESP), ebp(Operand::EBP), esi(Operand::ESI), edi(Operand::EDI);
static const XBYAK_CONSTEXPR Reg16 ax(Operand::AX), cx(Operand::CX), dx(Operand::DX), bx(Operand::BX), sp(Operand::SP),
    bp(Operand::BP), si(Operand::SI), di(Operand::DI);
static const XBYAK_CONSTEXPR Reg8 al(Operand::AL), cl(Operand::CL), dl(Operand::DL), bl(Operand::BL), ah(Operand::AH),
    ch(Operand::CH), dh(Operand::DH), bh(Operand::BH);
static const XBYAK_CONSTEXPR AddressFrame ptr(0), byte(8), word(16), dword(32), qword(64), xword(128), yword(256),
    zword(512);
static const XBYAK_CONSTEXPR AddressFrame ptr_b(0, true), xword_b(128, true), yword_b(256, true), zword_b(512, true);
static const XBYAK_CONSTEXPR Fpu st0(0), st1(1), st2(2), st3(3), st4(4), st5(5), st6(6), st7(7);
static const XBYAK_CONSTEXPR Opmask k0(0), k1(1), k2(2), k3(3), k4(4), k5(5), k6(6), k7(7);
static const XBYAK_CONSTEXPR BoundsReg bnd0(0), bnd1(1), bnd2(2), bnd3(3);
static const XBYAK_CONSTEXPR EvexModifierRounding T_sae(EvexModifierRounding::T_SAE),
    T_rn_sae(EvexModifierRounding::T_RN_SAE), T_rd_sae(EvexModifierRounding::T_RD_SAE),
    T_ru_sae(EvexModifierRounding::T_RU_SAE), T_rz_sae(EvexModifierRounding::T_RZ_SAE);
static const XBYAK_CONSTEXPR EvexModifierZero T_z;
#ifdef XBYAK64
static const XBYAK_CONSTEXPR Reg64 rax(Operand::RAX), rcx(Operand::RCX), rdx(Operand::RDX), rbx(Operand::RBX),
    rsp(Operand::RSP), rbp(Operand::RBP), rsi(Operand::RSI), rdi(Operand::RDI), r8(Operand::R8), r9(Operand::R9),
    r10(Operand::R10), r11(Operand::R11), r12(Operand::R12), r13(Operand::R13), r14(Operand::R14), r15(Operand::R15);
static const XBYAK_CONSTEXPR Reg32 r8d(8), r9d(9), r10d(10), r11d(11), r12d(12), r13d(13), r14d(14), r15d(15);
static const XBYAK_CONSTEXPR Reg16 r8w(8), r9w(9), r10w(10), r11w(11), r12w(12), r13w(13), r14w(14), r15w(15);
static const XBYAK_CONSTEXPR Reg8 r8b(8), r9b(9), r10b(10), r11b(11), r12b(12), r13b(13), r14b(14), r15b(15),
    spl(Operand::SPL, true), bpl(Operand::BPL, true), sil(Operand::SIL, true), dil(Operand::DIL, true);
static const XBYAK_CONSTEXPR Xmm xmm8(8), xmm9(9), xmm10(10), xmm11(11), xmm12(12), xmm13(13), xmm14(14), xmm15(15);
static const XBYAK_CONSTEXPR Xmm xmm16(16), xmm17(17), xmm18(18), xmm19(19), xmm20(20), xmm21(21), xmm22(22), xmm23(23);
static const XBYAK_CONSTEXPR Xmm xmm24(24), xmm25(25), xmm26(26), xmm27(27), xmm28(28), xmm29(29), xmm30(30), xmm31(31);
static const XBYAK_CONSTEXPR Ymm ymm8(8), ymm9(9), ymm10(10), ymm11(11), ymm12(12), ymm13(13), ymm14(14), ymm15(15);
static const XBYAK_CONSTEXPR Ymm ymm16(16), ymm17(17), ymm18(18), ymm19(19), ymm20(20), ymm21(21), ymm22(22), ymm23(23);
static const XBYAK_CONSTEXPR Ymm ymm24(24), ymm25(25), ymm26(26), ymm27(27), ymm28(28), ymm29(29), ymm30(30), ymm31(31);
static const XBYAK_CONSTEXPR Zmm zmm8(8), zmm9(9), zmm10(10), zmm11(11), zmm12(12), zmm13(13), zmm14(14), zmm15(15);
static const XBYAK_CONSTEXPR Zmm zmm16(16), zmm17(17), zmm18(18), zmm19(19), zmm20(20), zmm21(21), zmm22(22), zmm23(23);
static const XBYAK_CONSTEXPR Zmm zmm24(24), zmm25(25), zmm26(26), zmm27(27), zmm28(28), zmm29(29), zmm30(30), zmm31(31);
static const XBYAK_CONSTEXPR Zmm tmm0(0), tmm1(1), tmm2(2), tmm3(3), tmm4(4), tmm5(5), tmm6(6), tmm7(7);
static const XBYAK_CONSTEXPR RegRip rip;
#endif
#ifndef XBYAK_DISABLE_SEGMENT
static const XBYAK_CONSTEXPR Segment es(Segment::es), cs(Segment::cs), ss(Segment::ss), ds(Segment::ds),
    fs(Segment::fs), gs(Segment::gs);
#endif
}  // namespace util

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

}  // namespace Xbyak

#endif  // XBYAK_XBYAK_H_
