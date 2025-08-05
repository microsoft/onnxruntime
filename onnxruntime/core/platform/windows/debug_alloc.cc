// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Debug Memory Leak Checking
//
// Implements a custom operator new and delete that will capture a callstack in each allocation
// It creates a separate heap at startup and walks the remaining allocations at process exit,
// dumping out the callstacks to the console and showing a message box if there were any leaks.
//
// It creates & destroys itself in init_seg(lib) so it should scope all user code
//
#ifndef NDEBUG
#ifdef ONNXRUNTIME_ENABLE_MEMLEAK_CHECK
constexpr int c_callstack_limit = 32;  // Maximum depth of callstack in leak trace
#define VALIDATE_HEAP_EVERY_ALLOC 0    // Call HeapValidate on every new/delete

#pragma warning(disable : 4073)  // initializers put in library initialization area (this is intentional)
#pragma init_seg(lib)

// as this is a debug only checker that does some very low level things and isn't used in the released code
// ignore a bunch of C++ Core Guidelines code analysis warnings
#pragma warning(disable : 26409)  // r.11 Don't use 'new' explicitly.
#pragma warning(disable : 26426)  // i.22 Static local variables use non-constexpr initializer.
#pragma warning(disable : 26481)  // bounds.1 Don't use pointer arithmetic.
#pragma warning(disable : 26482)  // bounds.2 Only index into arrays using constant expressions.
#pragma warning(disable : 26485)  // bounds.3 No array to pointer decay.
#pragma warning(disable : 26490)  // type.1 Don't use reinterpret_cast
#pragma warning(disable : 26493)  // type.4 Don't use C-style casts

#include <windows.h>
#include <sstream>
#include <iostream>
#include "debug_alloc.h"
#include <DbgHelp.h>
#pragma comment(lib, "Dbghelp.lib")

// If you are seeing errors of
// "Error LNK2005: "void __cdecl operator delete(void *)" (??3@YAXPEAX@Z) already defined in LIBCMTD.lib(delete_scalar.obj)"
// Please read:https://developercommunity.visualstudio.com/content/problem/534202/visual-studio-2017-msvcrtlib-link-error.html

_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new(size_t size) { return DebugHeapAlloc(size, 1); }
_Ret_notnull_ _Post_writable_byte_size_(size) void* operator new[](size_t size) { return DebugHeapAlloc(size, 1); }
void operator delete(void* p) noexcept { DebugHeapFree(p); }
void operator delete[](void* p) noexcept { DebugHeapFree(p); }

struct MemoryBlock {
  MemoryBlock(unsigned framesToSkip = 1) noexcept {
    unsigned i = CaptureStackBackTrace(framesToSkip + 1, _countof(m_pTraces), m_pTraces, nullptr);
    for (; i < _countof(m_pTraces); i++)
      m_pTraces[i] = nullptr;
  }

  void* m_pTraces[c_callstack_limit];
};

struct SymbolHelper {
  HANDLE process_handle_ = GetCurrentProcess();
  bool initialized_ = false;

  bool InitializeWhenNeeded() {
    // We try only once
    if (!initialized_) {
      SymSetOptions(SymGetOptions() | SYMOPT_DEFERRED_LOADS);
      // We use GetCurrentProcess() because other libs are likely to use it
      if (!SymInitialize(process_handle_, nullptr, true)) {
        const unsigned long long error{GetLastError()};
        std::cerr << "SymInitialize() failed: " << error << std::endl;
        return false;
      }
      initialized_ = true;
    }
    return true;
  }

  SymbolHelper() = default;

  bool LookupSymAndInitialize(const void* address, SYMBOL_INFO* symbol, std::ostream& message) {
    if (SymFromAddr(process_handle_, reinterpret_cast<ULONG_PTR>(address), 0, symbol) != TRUE) {
      if (GetLastError() == ERROR_INVALID_HANDLE) {
        // Try to initialize first
        if (!InitializeWhenNeeded() ||
            SymFromAddr(process_handle_, reinterpret_cast<ULONG_PTR>(address), 0, symbol) != TRUE) {
          message << "0x" << address << " (Unknown symbol)";
          return false;
        }
      } else {
        message << "0x" << address << " (Unknown symbol)";
        return false;
      }
    }
    return true;
  }

  void Lookup(const void* address, std::ostream& message) {
    SYMBOL_INFO_PACKAGE symbol_info_package{};
    SYMBOL_INFO* symbol = &symbol_info_package.si;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = std::size(symbol_info_package.name);

    if (!LookupSymAndInitialize(address, symbol, message)) {
      return;
    }

    Line line;
    DWORD displacement;
    if (SymGetLineFromAddr(process_handle_, reinterpret_cast<ULONG_PTR>(address), &displacement, &line) == false) {
      message << "(unknown file & line number): " << symbol->Name;
      return;
    }

    message << line.FileName << "(" << line.LineNumber << "): " << symbol->Name;
  }

  struct Line : IMAGEHLP_LINE {
    Line() noexcept {
      SizeOfStruct = sizeof(IMAGEHLP_LINE);
    }
  };
};

static HANDLE g_heap{};
unsigned g_cumulativeAllocationCount{};
unsigned g_allocationCount{};
uint64_t g_cumulativeAllocationBytes{};

// Disable C6386: Buffer overrun for just this section.
// 'p' is considered a 0 byte array as it's a void*, so the write to 'p'
// in DebugHeapAlloc and DebugHeapReAlloc trigger spurious warnings.
#pragma warning(push)
#pragma warning(disable : 6386)

void* DebugHeapAlloc(size_t size, unsigned framesToSkip) {
#if (VALIDATE_HEAP_EVERY_ALLOC)
  if (HeapValidate(g_heap, 0, nullptr) == 0)
    exit(-1);
#endif

  g_cumulativeAllocationCount++;
  g_cumulativeAllocationBytes += size;
  void* p = HeapAlloc(g_heap, 0, size + sizeof(MemoryBlock));
  if (!p)
    throw std::bad_alloc();

  g_allocationCount++;
  new (p) MemoryBlock(framesToSkip + 1);
  return static_cast<BYTE*>(p) + sizeof(MemoryBlock);  // Adjust outgoing pointer
}

void* DebugHeapReAlloc(void* p, size_t size) {
  if (!p)  // Std library will call realloc(nullptr, size)
    return DebugHeapAlloc(size);

  g_cumulativeAllocationCount++;
  g_cumulativeAllocationBytes += size;
  p = static_cast<BYTE*>(p) - sizeof(MemoryBlock);  // Adjust incoming pointer
  p = HeapReAlloc(g_heap, 0, p, size + sizeof(MemoryBlock));
  if (!p)
    throw std::bad_alloc();

  new (p) MemoryBlock;                                 // Redo the callstack
  return static_cast<BYTE*>(p) + sizeof(MemoryBlock);  // Adjust outgoing pointer
}

#pragma warning(pop)  // buffer overrun

void DebugHeapFree(void* p) noexcept {
#if (VALIDATE_HEAP_EVERY_ALLOC)
  if (HeapValidate(g_heap, 0, nullptr) == 0)
    exit(-1);
#endif

  if (!p)
    return;

  g_allocationCount--;
  p = static_cast<BYTE*>(p) - sizeof(MemoryBlock);  // Adjust incoming pointer
  if (HeapFree(g_heap, 0, p) == 0)
    __debugbreak();  // If this hits, we either double deleted memory or we somehow tried to delete main heap memory after the leak checker started
}

static struct Memory_LeakCheck {
  Memory_LeakCheck() noexcept;
  ~Memory_LeakCheck();
  Memory_LeakCheck(const Memory_LeakCheck&) = delete;
  Memory_LeakCheck& operator=(const Memory_LeakCheck&) = delete;
  Memory_LeakCheck(Memory_LeakCheck&&) = delete;
  Memory_LeakCheck& operator=(Memory_LeakCheck&&) = delete;
} g_memory_leak_check;

Memory_LeakCheck::Memory_LeakCheck() noexcept {
  g_heap = HeapCreate(0, 0, 0);
}

// print message to debug output and stdout
// no trailing newline will be added
static void DebugPrint(const char* message) {
  OutputDebugStringA(message);
  std::cout << "memleakdbg: " << message;
}

Memory_LeakCheck::~Memory_LeakCheck() {
  SymbolHelper symbols;

  // Create a new heap so we can still allocate memory while dumping the memory leaks
  HANDLE heap = HeapCreate(0, 0, 0);
  std::swap(heap, g_heap);  // Swap it out with our current heap

  unsigned leaked_bytes = 0;
  unsigned leak_count = 0;

  PROCESS_HEAP_ENTRY entry{};
  while (HeapWalk(heap, &entry)) {
    if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) == 0)
      continue;

    const MemoryBlock& block = *static_cast<const MemoryBlock*>(entry.lpData);
    const BYTE* pBlock = static_cast<const BYTE*>(entry.lpData) + sizeof(MemoryBlock);

    std::ostringstream message;
    message << (entry.cbData - sizeof(MemoryBlock)) << " bytes at location 0x" << static_cast<const void*>(pBlock)
            << "\n";
    for (auto& p : block.m_pTraces) {
      if (!p) break;
      symbols.Lookup(p, message);
      message << "\n";
    }

    const std::string string = message.str();

    // Google test has memory leaks that they haven't fixed. One such issue is tracked here: https://github.com/google/googletest/issues/692
    //
    // In gtest-port.cc in function: static ThreadIdToThreadLocals* GetThreadLocalsMapLocked()
    //     static ThreadIdToThreadLocals* map = new ThreadIdToThreadLocals;
    //
    // In gtest-port.cc in Mutex::~Mutex() there is this comment:
    //     "Static mutexes are leaked intentionally. It is not thread-safe to try to clean them up."
    // Which explains this leak inside of: void Mutex::ThreadSafeLazyInit()
    //     critical_section_ = new CRITICAL_SECTION;
    //
    // in google/re2 re2.cc initializes leaking singletons
    //     std::call_once(empty_once, []() {
    //     empty_string = new string;
    //     empty_named_groups = new std::map<string, int>;
    //     empty_group_names = new std::map<int, string>; });
    //
    // In the Abseil (ABSL) flags library used by onnxruntime_perf_test, specifying "--help"
    // causes the program to call exit(1). This is an intentional design choice from Google,
    // treating "--help" as an early termination condition (the program does not perform its
    // normal execution. See MaybeExit in usage.cc).
    //
    // In normal execution of onnxruntime_perf_test, Abseil flags are defined as global variables
    // and persist for the lifetime of the program. They are not explicitly freed, so leak checkers
    // may report them, but these are not true leaks. Valgrind, for example, reports them as
    // "still reachable" rather than "definitely lost".
    //
    // As a result, many resources will not be cleaned up, including:
    //   - Abseil's internal storage for flags, allocated in static/global objects inside
    //     absl::flags_internal (e.g., FlagImpl::Init)
    //   - The absl::FlagsUsageConfig instance
    //   - Performance test utilities that hold std::vector objects for converting argv to UTF-8 strings
    //   - The onnxruntime::perftest::PerformanceTestConfig instance
    //
    // Essentially, any object instantiated before calling absl::ParseCommandLine will not
    // be cleaned up. This behavior is expected when running with "--help".
    if (string.find("RtlRunOnceExecuteOnce") == std::string::npos &&
        string.find("re2::RE2::Init") == std::string::npos &&
        string.find("dynamic initializer for 'FLAGS_") == std::string::npos &&
        string.find("AbslFlagDefaultGenForgtest_") == std::string::npos &&
        string.find("AbslFlagDefaultGenForundefok::Gen") == std::string::npos &&
        string.find("::SetProgramUsageMessage") == std::string::npos &&
        string.find("testing::internal::ParseGoogleTestFlagsOnly") == std::string::npos &&
        string.find("testing::internal::Mutex::ThreadSafeLazyInit") == std::string::npos &&
        string.find("testing::internal::ThreadLocalRegistryImpl::GetThreadLocalsMapLocked") == std::string::npos &&
        string.find("testing::internal::ThreadLocalRegistryImpl::GetValueOnCurrentThread") == std::string::npos &&
        string.find("PyInit_onnxruntime_pybind11_state") == std::string::npos &&
        string.find("google::protobuf::internal::InitProtobufDefaultsSlow") == std::string::npos &&
        string.find("flags_internal::ParseCommandLineImpl") == std::string::npos &&
        string.find("flags_internal::FlagImpl::Init") == std::string::npos &&
        string.find("SetFlagsUsageConfig") == std::string::npos &&
        string.find("perftest::utils::ConvertArgvToUtf8Strings") == std::string::npos &&
        string.find("perftest::utils::CStringsFromStrings") == std::string::npos &&
        string.find("perftest::PerformanceTestConfig::PerformanceTestConfig") == std::string::npos) {
      if (leaked_bytes == 0)
        DebugPrint("\n-----Starting Heap Trace-----\n\n");

      leak_count++;
      leaked_bytes += entry.cbData - sizeof(MemoryBlock);
      DebugPrint(string.c_str());
      DebugPrint("\n");
    }
  }

  if (leaked_bytes) {
    DebugPrint("-----Ending Heap Trace-----\n\n");

    std::cout << "\n----- MEMORY LEAKS: " << leaked_bytes << " bytes of memory leaked in "
              << leak_count << " allocations\n";
    if (!IsDebuggerPresent()) {
      exit(-1);
    }

  } else {
    DebugPrint("\n----- No memory leaks detected -----\n\n");
  }

  HeapDestroy(heap);
  HeapDestroy(g_heap);
  g_heap = nullptr;  // Any allocations after this point will fail
}
#endif
#endif
