// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include <iostream>
#include <mutex>
#include <sstream>

#include <windows.h>
#include <DbgHelp.h>

#include "core/common/logging/logging.h"
#include "gsl/gsl"

namespace onnxruntime {

namespace detail {
class CaptureStackTrace {
 public:
  CaptureStackTrace() = default;

  std::vector<std::string> Trace() const;

 private:
  std::string Lookup(void* address_in) const;

  HANDLE process_ = GetCurrentProcess();
  static const int kCallstackLimit = 64;  // Maximum depth of callstack
};
}  // namespace detail

// Get the stack trace. Currently only enabled for a DEBUG build as we require the DbgHelp library.
std::vector<std::string> GetStackTrace() {
#ifndef NDEBUG
// TVM need to run with shared CRT, so won't work with debug helper now
#if !(defined USE_TVM || (defined USE_NGRAPH && defined _WIN32))
  return detail::CaptureStackTrace().Trace();
#else
  return {};
#endif
#else
  return {};
#endif
}

namespace detail {
#ifndef NDEBUG
#if !(defined USE_TVM || (defined USE_NGRAPH && defined _WIN32))
class SymbolHelper {
 public:
  SymbolHelper() noexcept {
    SymSetOptions(SymGetOptions() | SYMOPT_DEFERRED_LOADS);
    // this could have been called earlier by a higher level component, so failure doesn't necessarily mean
    // this won't work. however we should only call SymCleanup if it was successful.
    if (SymInitialize(process_, nullptr, true)) {
      cleanup_ = true;
    } else {
      // Log it so we know it happened. Can't do anything else about it.
      LOGS_DEFAULT(WARNING) << "Failed to initialize symbols for providing stack trace. Error: 0x"
                            << std::hex << GetLastError();
    }
  }

  struct Symbol : SYMBOL_INFO {
    Symbol() noexcept {
      SizeOfStruct = sizeof(SYMBOL_INFO);
      GSL_SUPPRESS(bounds .3)
      MaxNameLen = _countof(buffer);
    }

    char buffer[1024] = {0};
  };

  struct Line : IMAGEHLP_LINE64 {
    Line() noexcept {
      SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    }
  };

  ~SymbolHelper() {
    if (cleanup_)
      SymCleanup(process_);
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SymbolHelper);

  HANDLE process_ = GetCurrentProcess();
  bool cleanup_ = false;
};

std::vector<std::string> CaptureStackTrace::Trace() const {
#pragma warning(push)
#pragma warning(disable : 26426)
  static SymbolHelper sh;
#pragma warning(pop)

  std::vector<std::string> stacktrace;

  PVOID frames[kCallstackLimit];
  const auto f = gsl::make_span(frames);
  const auto num_frames = CaptureStackBackTrace(0, kCallstackLimit, f.data(), nullptr);

  stacktrace.reserve(num_frames);

  // hide CaptureStackTrace::Trace and GetStackTrace so the output starts with the 'real' location
  const int frames_to_skip = 2;

  // we generally want to skip the first two frames, but if something weird is going on (e.g. code coverage is
  // running) and we only have 1 or 2 frames, output them so there's at least something that may be meaningful
  const uint16_t start_frame = num_frames > frames_to_skip ? frames_to_skip : 0;
  for (uint16_t i = start_frame; i < num_frames; ++i) {
    stacktrace.push_back(Lookup(f[i]));
  }

  return stacktrace;
}

std::string CaptureStackTrace::Lookup(void* address_in) const {
  SymbolHelper::Symbol symbol;
  std::ostringstream result;

  DWORD64 address = 0;

  GSL_SUPPRESS(type .1) {
    address = reinterpret_cast<DWORD64>(address_in);
  }

  if (SymFromAddr(process_, address, 0, &symbol) == false) {
    result << "0x" << std::hex << address << " (Unknown symbol)";
  } else
    GSL_SUPPRESS(bounds .3)  // symbol.Name converts to char*
    {
      SymbolHelper::Line line;
      DWORD displacement;
      if (SymGetLineFromAddr64(process_, address, &displacement, &line) == false) {
        result << "???: " << symbol.Name;
      } else {
        result << line.FileName << '(' << line.LineNumber << "): " << symbol.Name;
      }
    }

  return result.str();
}

#endif
#endif
}  // namespace detail
}  // namespace onnxruntime

#include <tchar.h>
#include <DbgHelp.h>
#pragma comment(lib, "Dbghelp.lib")

struct SymbolHelper {
  SymbolHelper() noexcept {
    SymSetOptions(SymGetOptions() | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), nullptr, true);
  }

  void Lookup(const ULONG_PTR address) {
    char buffer[2048] = {0};
    Symbol symbol;
    if (SymFromAddr(GetCurrentProcess(), address, 0, &symbol) == false) {
      _snprintf_s(buffer, _TRUNCATE, "0x%08IX (Unknown symbol)", address);
      std::cout << buffer;
      return;
    }

    Line line;
    DWORD displacement;
    if (SymGetLineFromAddr(GetCurrentProcess(), address, &displacement, &line) == false) {
      _snprintf_s(buffer, _TRUNCATE, "(unknown file & line number): %s", symbol.Name);
      std::cout << buffer;
      return;
    }

    _snprintf_s(buffer, _TRUNCATE, "%s(%d): %s", line.FileName, line.LineNumber, symbol.Name);
    std::cout << buffer;
  }

  struct Symbol : SYMBOL_INFO {
    Symbol() noexcept {
      SizeOfStruct = sizeof(SYMBOL_INFO);
      MaxNameLen = _countof(buffer);
    }

    char buffer[1024] = {0};
  };

  struct Line : IMAGEHLP_LINE {
    Line() noexcept {
      SizeOfStruct = sizeof(IMAGEHLP_LINE);
    }
  };
};

LONG WINAPI UnHandledExceptionFilter(EXCEPTION_POINTERS* exception_pointers) {
  EXCEPTION_RECORD& exception = *(exception_pointers->ExceptionRecord);

  std::cout << "******  Crashed: ";

  switch (exception.ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION:
      std::cout << "Access violation ";
      std::cout << (exception.ExceptionInformation[0] == 0 ? "reading from" : "writing to") << " address " << std::hex << exception.ExceptionInformation[1];
      break;
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
      std::cout << "Array bounds exceeded";
      break;
    case EXCEPTION_DATATYPE_MISALIGNMENT:
      std::cout << "Data type misalignment";
      break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
      std::cout << "Illegal instruction";
      break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
      std::cout << "Integer divide by zero";
      break;
    case EXCEPTION_STACK_OVERFLOW:
      std::cout << "Stack overflow";
      break;
    default:
      std::cout << "Exception Code:" << std::hex << exception.ExceptionCode;
  }

  std::cout << " at " << std::hex << exception.ExceptionAddress;
  std::cout << "\r\n\r\n";

  SymbolHelper symbols;
  void* traces[64];

  unsigned count = CaptureStackBackTrace(0, _countof(traces), traces, nullptr);
  for (unsigned i = 0; i < count; i++) {
    symbols.Lookup(reinterpret_cast<ULONG_PTR>(traces[i]));
    std::cout << "\r\n";
  }

  return EXCEPTION_CONTINUE_SEARCH;
}

bool InitCrashHandler = (SetUnhandledExceptionFilter(UnHandledExceptionFilter), false);
