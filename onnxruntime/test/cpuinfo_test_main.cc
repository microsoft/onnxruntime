#include "cpuinfo.h"

#if defined(_M_IX86) || (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__i386__) || defined(__x86_64__)
#define CPUIDINFO_ARCH_X86
#endif

#if defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#define CPUIDINFO_ARCH_ARM
#endif  // ARM or ARM64

#if !defined(CPUIDINFO_ARCH_X86) && !defined(CPUIDINFO_ARCH_ARM)
//#error neither x86 nor arm are defined
#endif

int main() {
    cpuinfo_initialize();
    return 0;
}
