// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// MLAS diagnostic helper. Always-on (no env var) so it works in CI
// environments where setting environment variables is not possible.
//
// At process startup (before any test runs) this dumps:
//   * Build-time ISA macros (__AVX2__, __AVX512F__, ...)
//   * Runtime MlasPlatform feature flags (Avx2Supported_, Avx512Supported_)
//   * Active QNBitGemmDispatch table identity (Avx2 / Avx2vnni / Avx512 /
//     Avx512vnni / other)
//   * On Linux, raw CPUID leaves 1 and 7 sub-leaf 0 so we can confirm what
//     the kernel/hypervisor actually exposes vs. what MlasPlatform decided.
//
// When this file is linked into the test binary it also installs a SIGILL
// handler (Linux only) that prints the faulting RIP, the module/symbol
// nearest to it (dladdr), and the first 16 bytes at RIP so we can
// disassemble the offending instruction offline. The handler is
// async-signal-safe (write(2) only). It re-raises SIGILL with the default
// disposition afterwards so the process still terminates and gtest still
// reports the failing test.
//
// All output is tagged with the "MLAS-DIAG:" prefix so it can be grep'd
// out of CI logs.
//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gtest/gtest.h"

#include "core/mlas/lib/mlasi.h"

#if defined(__linux__)
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
#endif

namespace {

constexpr const char* kTag = "MLAS-DIAG: ";

void PrintBuildMacros() {
  std::fprintf(stderr, "%sbuild macros:", kTag);
#if defined(__AVX__)
  std::fprintf(stderr, " __AVX__");
#endif
#if defined(__AVX2__)
  std::fprintf(stderr, " __AVX2__");
#endif
#if defined(__FMA__)
  std::fprintf(stderr, " __FMA__");
#endif
#if defined(__AVX512F__)
  std::fprintf(stderr, " __AVX512F__");
#endif
#if defined(__AVX512BW__)
  std::fprintf(stderr, " __AVX512BW__");
#endif
#if defined(__AVX512DQ__)
  std::fprintf(stderr, " __AVX512DQ__");
#endif
#if defined(__AVX512VL__)
  std::fprintf(stderr, " __AVX512VL__");
#endif
#if defined(__AVX512VNNI__)
  std::fprintf(stderr, " __AVX512VNNI__");
#endif
#if defined(__GNUC__)
  std::fprintf(stderr, " __GNUC__=%d", __GNUC__);
#endif
#if defined(_MSC_VER)
  std::fprintf(stderr, " _MSC_VER=%d", _MSC_VER);
#endif
  std::fprintf(stderr, "\n");
}

void PrintMlasPlatform() {
  const auto& p = GetMlasPlatform();
  std::fprintf(stderr,
               "%sMlasPlatform: Avx2Supported_=%d Avx512Supported_=%d\n",
               kTag,
               static_cast<int>(p.Avx2Supported_),
               static_cast<int>(p.Avx512Supported_));

  const void* dispatch = static_cast<const void*>(p.QNBitGemmDispatch);
  const char* name = "unknown/null";
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
  if (dispatch == &MlasSQNBitGemmDispatchAvx2) {
    name = "Avx2";
  } else if (dispatch == &MlasSQNBitGemmDispatchAvx2vnni) {
    name = "Avx2vnni";
  } else if (dispatch == &MlasSQNBitGemmDispatchAvx512) {
    name = "Avx512";
  } else if (dispatch == &MlasSQNBitGemmDispatchAvx512vnni) {
    name = "Avx512vnni";
  } else if (dispatch == nullptr) {
    name = "null";
  }
#endif
  std::fprintf(stderr,
               "%sQNBitGemmDispatch=%p (%s)\n",
               kTag, dispatch, name);
}

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))

void PrintRawCpuid() {
  auto cpuid = [](uint32_t leaf, uint32_t subleaf,
                  uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
    __asm__ volatile(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(leaf), "c"(subleaf));
  };
  uint32_t a, b, c, d;
  cpuid(1, 0, a, b, c, d);
  std::fprintf(stderr, "%sCPUID(1,0): eax=%08x ebx=%08x ecx=%08x edx=%08x\n",
               kTag, a, b, c, d);
  cpuid(7, 0, a, b, c, d);
  std::fprintf(stderr, "%sCPUID(7,0): eax=%08x ebx=%08x ecx=%08x edx=%08x\n",
               kTag, a, b, c, d);
}

// Async-signal-safe writer: integer -> hex into a small buffer.
size_t HexWrite(uintptr_t v, char* out, size_t out_len) {
  static const char kHex[] = "0123456789abcdef";
  // Reserve room for "0x" + up to 16 hex digits + null.
  char tmp[20];
  size_t n = 0;
  if (v == 0) {
    tmp[n++] = '0';
  } else {
    while (v != 0 && n < sizeof(tmp)) {
      tmp[n++] = kHex[v & 0xF];
      v >>= 4;
    }
  }
  // Now reverse into out, prefixed with "0x".
  size_t w = 0;
  if (out_len > 0) out[w++] = '0';
  if (out_len > 1) out[w++] = 'x';
  while (n > 0 && w < out_len) {
    out[w++] = tmp[--n];
  }
  return w;
}

void SafeWriteStr(const char* s) {
  if (s == nullptr) return;
  size_t n = 0;
  while (s[n] != '\0') ++n;
  (void)write(STDERR_FILENO, s, n);
}

void SafeWriteHex(uintptr_t v) {
  char buf[20];
  size_t n = HexWrite(v, buf, sizeof(buf));
  (void)write(STDERR_FILENO, buf, n);
}

void SigillHandler(int sig, siginfo_t* info, void* /*ucontext*/) {
  SafeWriteStr(kTag);
  SafeWriteStr("SIGILL caught. si_addr=");
  uintptr_t addr = reinterpret_cast<uintptr_t>(info ? info->si_addr : nullptr);
  SafeWriteHex(addr);
  SafeWriteStr("\n");

  // Try to resolve module + nearest symbol. dladdr is not strictly
  // async-signal-safe, but in practice on glibc it is safe enough for a
  // best-effort crash dump; if it deadlocks we still get the si_addr above.
  Dl_info dlinfo{};
  if (info && dladdr(info->si_addr, &dlinfo) != 0) {
    SafeWriteStr(kTag);
    SafeWriteStr("dladdr: fname=");
    SafeWriteStr(dlinfo.dli_fname ? dlinfo.dli_fname : "(null)");
    SafeWriteStr(" fbase=");
    SafeWriteHex(reinterpret_cast<uintptr_t>(dlinfo.dli_fbase));
    SafeWriteStr(" sname=");
    SafeWriteStr(dlinfo.dli_sname ? dlinfo.dli_sname : "(null)");
    SafeWriteStr(" saddr=");
    SafeWriteHex(reinterpret_cast<uintptr_t>(dlinfo.dli_saddr));
    SafeWriteStr("\n");
  }

  // Dump first 16 bytes at si_addr so we can decode the instruction offline.
  if (info && info->si_addr != nullptr) {
    SafeWriteStr(kTag);
    SafeWriteStr("bytes@si_addr:");
    const unsigned char* p = static_cast<const unsigned char*>(info->si_addr);
    for (int i = 0; i < 16; ++i) {
      char two[3];
      static const char kHex[] = "0123456789abcdef";
      two[0] = kHex[(p[i] >> 4) & 0xF];
      two[1] = kHex[p[i] & 0xF];
      two[2] = ' ';
      (void)write(STDERR_FILENO, " ", 1);
      (void)write(STDERR_FILENO, two, 3);
    }
    SafeWriteStr("\n");
  }

  // Re-raise with default disposition so gtest still records the crash.
  struct sigaction dfl{};
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  sigaction(sig, &dfl, nullptr);
  raise(sig);
}

void InstallSigillHandler() {
  struct sigaction sa{};
  sa.sa_sigaction = &SigillHandler;
  sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGILL, &sa, nullptr) != 0) {
    std::fprintf(stderr, "%sfailed to install SIGILL handler\n", kTag);
  } else {
    std::fprintf(stderr, "%sSIGILL handler installed\n", kTag);
  }
}

#else  // !linux x86

void PrintRawCpuid() {}
void InstallSigillHandler() {}

#endif

class MlasDiagEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    std::fprintf(stderr, "%s---- MLAS diagnostic dump ----\n", kTag);
    PrintBuildMacros();
    PrintRawCpuid();
    PrintMlasPlatform();
    InstallSigillHandler();
    std::fprintf(stderr, "%s---- end MLAS diagnostic dump ----\n", kTag);
    std::fflush(stderr);
  }
};

// Register at program startup so it runs before any TEST() body executes.
const ::testing::Environment* const kMlasDiagEnv =
    ::testing::AddGlobalTestEnvironment(new MlasDiagEnvironment());

}  // namespace
