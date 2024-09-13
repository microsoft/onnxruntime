// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef __TEST_LOG_H__
#define __TEST_LOG_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <cwchar>
#include <locale>
#include <codecvt>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>  // For MultiByteToWideChar and CP_UTF8
#endif

// Forward declarations
//
class OnnxPrediction;
std::wostream& operator<<(std::wostream& out, OnnxPrediction& pred);

namespace Logger {
// Alias for end of line
//
using LogEndln = void*;

// Used to format the output of the logs.
//
class TestLog {
 public:
  // Flush out all output streams used by logger
  void flush();

  // Print out output of a prediction.
  TestLog& operator<<(OnnxPrediction& pred);

  // Generic log output that appends timing information.
  template <typename T>
  TestLog& operator<<(const T& info);

  // Disable logging
  inline void disable();

  // Enable logging
  inline void enable();

  // Ends the current line so that the next line
  // can start with time information.
  void operator<<(LogEndln info);

  // Minimize log
  inline void minLog();

  // Maintain ring buffer
  // Note: This is only used for minimum logging;
  // if normal logging is being used, this map must
  // be constrained.
  void insert(std::wstring data);

  // Singleton constructor - only one object exists
  // Note: this resource is not thread-safe
  TestLog();

 private:
  bool print_time_info = true;
  bool logging_on = true;
  bool min_log = false;
  std::wofstream logFile;
  std::wstring logFileName;
  std::map<size_t, std::pair<size_t, std::wstring>> ring_buffer;
  static constexpr int logFileLineWidth{128};
  static constexpr int logFileLen{1000};
  std::wstring string_to_wstring(const std::string& str) {
#if defined(_WIN32) || defined(_WIN64)
    // Windows implementation
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
#else
    // Linux implementation
    std::mbstate_t state = std::mbstate_t();
    const char* c_str = str.c_str();
    size_t len = std::mbsrtowcs(nullptr, &c_str, 0, &state);
    std::wstring wstr(len, L'\0');
    std::mbsrtowcs(&wstr[0], &c_str, len, &state);
    return wstr;
#endif
  }
};

// Reference to initialized logger
// Note: this resource is not thread-safe and only
// one exists for the entire process.
//
extern TestLog testLog;

// Object used to mark end of line format for testLog
//
static constexpr LogEndln endl = nullptr;

// Utility function to convert from char to wchar
//
std::wstring towstr(const char* pStr);
}  // namespace Logger

// Inline Functions

// Minimize log
inline void Logger::TestLog::minLog() {
  min_log = true;
}

// Enable logging
inline void Logger::TestLog::enable() {
  logging_on = true;
}

inline void Logger::TestLog::disable() {
  logging_on = false;
}

namespace Logger {

template <typename T>
TestLog& TestLog::operator<<(const T& info) {
  if (!logging_on) {
    return *this;
  }

  // Get the current time
  std::chrono::system_clock::time_point today{std::chrono::system_clock::now()};
  std::time_t tt{std::chrono::system_clock::to_time_t(today)};
#if defined(_WIN32) || defined(_WIN64)
  std::tm tm;
  localtime_s(&tm, &tt);  // Thread-safe on Windows
#else
  std::tm tm = *std::localtime(&tt);  // Thread-safe on Linux
#endif

  // Buffer for formatted time
  char buf[100];
  std::strftime(buf, sizeof(buf), "%c", &tm);

  // Convert multi-byte to wide character string
  std::wstring wstr = string_to_wstring(buf);

  // Use std::wstring_view to avoid copying the wide string
  std::wstring_view temp(wstr);

  // Create a wstringstream to format the output
  std::wstringstream stream;
  if (print_time_info) {
    stream << L"[" << temp << L"]" << L"\t";
  }

  // Append info to the stream
  if constexpr (std::is_same<T, std::string>()) {
    // Convert std::string to std::wstring and append
    std::wstring winfo = string_to_wstring(info);
    stream << winfo;
  } else {
    stream << info;
  }

  // Output the formatted log
  if (min_log) {
    insert(stream.str());
  } else {
    std::wcout << stream.str();
  }

  print_time_info = false;
  return *this;
}
}  // namespace Logger

namespace Logger {
template <typename CharT>
class cache_streambuf : public std::basic_streambuf<CharT> {
 public:
  using Base = std::basic_streambuf<CharT>;
  using char_type = typename Base::char_type;
  using int_type = typename Base::int_type;

  // Get the total number of unique errors found
  inline size_t get_unique_errors();

 protected:
  virtual int_type overflow(int_type ch = Base::traits_type::eof());

 private:
  std::basic_stringstream<char_type> buffer;
  std::unordered_map<std::basic_string<char_type>, size_t> exception_count;
};

using ccstream = cache_streambuf<char>;
using wcstream = cache_streambuf<wchar_t>;

template <typename CharT>
inline size_t cache_streambuf<CharT>::get_unique_errors() {
  return exception_count.size();
}

template <typename CharT>
auto cache_streambuf<CharT>::overflow(int_type ch) -> int_type {
  // If not end of file
  if (!Base::traits_type::eq_int_type(ch, Base::traits_type::eof())) {
    if (ch > 255) {
      if constexpr (std::is_same_v<char_type, char>) {
        std::cout << "Yikes";
      } else {
        std::wcout << L"Yikes";
      }
    }
    if (ch != int_type{'\n'}) {
      buffer << static_cast<char>(ch);
    } else {
      buffer << static_cast<char>(ch);
      exception_count[buffer.str()]++;
      if constexpr (std::is_same_v<char_type, char>) {
        std::cout << buffer.str();
      } else {
        std::wcout << buffer.str();
      }
      buffer = std::basic_stringstream<char_type>{};
    }
  }
  return Base::traits_type::not_eof(ch);
}
}  // namespace Logger

#endif
