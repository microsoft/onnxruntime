#include "core/framework/allocator.h"

struct xnn_allocator;

namespace onnxruntime {
namespace xnnpack {

// copy #define logic from XNNPACK src/xnnpack/common.h to determine workspace alignment
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
#define XNN_ARCH_X86 1
#else
#define XNN_ARCH_X86 0
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) && !defined(_M_ARM64EC)
#define XNN_ARCH_X86_64 1
#else
#define XNN_ARCH_X86_64 0
#endif

#if defined(__wasm__) && !defined(__wasm_relaxed_simd__) && !defined(__wasm_simd128__)
#define XNN_ARCH_WASM 1
#else
#define XNN_ARCH_WASM 0
#endif

#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
#define XNN_PLATFORM_MOBILE 1
#else
#define XNN_PLATFORM_MOBILE 0
#endif

#if XNN_ARCH_WASM
#define XNN_ALLOCATION_ALIGNMENT 4
#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_PLATFORM_MOBILE
#define XNN_ALLOCATION_ALIGNMENT 32
#else
#define XNN_ALLOCATION_ALIGNMENT 64
#endif
#else
#define XNN_ALLOCATION_ALIGNMENT 16
#endif

std::pair<AllocatorPtr&, xnn_allocator*> GetStoredAllocator();

}  // namespace xnnpack
}  // namespace onnxruntime
