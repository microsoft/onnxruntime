// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string.h>
#include <string>
#include <sstream>
#include <assert.h>
#include <stdexcept>
#if defined(_AIX)
#include <sys/stat.h>
#include <iostream>
#endif
#ifdef _WIN32
#include <Windows.h>
#include <time.h>  //strftime
#else
#include <sys/types.h>
#include <dirent.h>
#include <time.h>    //strftime
#include <stddef.h>  //ptrdiff_t
#endif
#include "core/common/path_string.h"
#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

using PATH_CHAR_TYPE = ORTCHAR_T;

template <typename T>
long OrtStrtol(const T* nptr, T** endptr);

template <typename T>
double OrtStrtod(const T* nptr, T** endptr);

/**
 * Convert a C string to ssize_t(or ptrdiff_t)
 * @return the converted integer value.
 */
template <typename T>
ptrdiff_t OrtStrToPtrDiff(const T* nptr, T** endptr);

template <>
inline ptrdiff_t OrtStrToPtrDiff<wchar_t>(const wchar_t* nptr, wchar_t** endptr) {
#ifdef _WIN32
#ifdef _M_AMD64
  return _wcstoi64(nptr, endptr, 10);
#else
  return wcstol(nptr, endptr, 10);
#endif
#else
  return wcstol(nptr, endptr, 10);
#endif
}

template <typename T>
size_t OrtStrftime(T* strDest, size_t maxsize, const T* format, const struct tm* timeptr);

template <>
inline size_t OrtStrftime<char>(char* strDest, size_t maxsize, const char* format, const struct tm* timeptr) {
  return strftime(strDest, maxsize, format, timeptr);
}

template <>
inline size_t OrtStrftime<wchar_t>(wchar_t* strDest, size_t maxsize, const wchar_t* format, const struct tm* timeptr) {
  return wcsftime(strDest, maxsize, format, timeptr);
}

template <>
inline ptrdiff_t OrtStrToPtrDiff<char>(const char* nptr, char** endptr) {
#ifdef _WIN32
#ifdef _M_AMD64
  return _strtoi64(nptr, endptr, 10);
#else
  return strtol(nptr, endptr, 10);
#endif
#else
  return strtol(nptr, endptr, 10);
#endif
}

template <>
inline long OrtStrtol<char>(const char* nptr, char** endptr) {
  return strtol(nptr, endptr, 10);
}

template <>
inline long OrtStrtol<wchar_t>(const wchar_t* nptr, wchar_t** endptr) {
  return wcstol(nptr, endptr, 10);
}

template <>
inline double OrtStrtod<char>(const char* nptr, char** endptr) {
  return strtod(nptr, endptr);
}

template <>
inline double OrtStrtod<wchar_t>(const wchar_t* nptr, wchar_t** endptr) {
  return wcstod(nptr, endptr);
}

namespace onnxruntime {

/**
 * return which dir contains this file path
 * if s equals to '//', the behavior of this function is undefined.
 */
common::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& s, std::basic_string<ORTCHAR_T>& output);
std::basic_string<PATH_CHAR_TYPE> GetLastComponent(const std::basic_string<PATH_CHAR_TYPE>& s);

template <typename T>
int CompareCString(const T* s1, const T* s2);

template <>
inline int CompareCString<char>(const char* s1, const char* s2) {
  return strcmp(s1, s2);
}

template <>
inline int CompareCString<wchar_t>(const wchar_t* s1, const wchar_t* s2) {
  return wcscmp(s1, s2);
}

enum class OrtFileType { TYPE_BLK,
                         TYPE_CHR,
                         TYPE_DIR,
                         TYPE_FIFO,
                         TYPE_LNK,
                         TYPE_REG,
                         TYPE_SOCK,
                         TYPE_UNKNOWN };

template <typename PATH_CHAR_TYPE>
PATH_CHAR_TYPE GetPathSep();

template <typename PATH_CHAR_TYPE>
PATH_CHAR_TYPE GetDot();

template <typename PATH_CHAR_TYPE>
bool HasExtensionOf(const std::basic_string<PATH_CHAR_TYPE>& s1, _In_ const PATH_CHAR_TYPE* s2) {
  typename std::basic_string<PATH_CHAR_TYPE>::size_type pos = s1.rfind(GetDot<PATH_CHAR_TYPE>());
  if (pos == std::basic_string<PATH_CHAR_TYPE>::npos || pos == s1.size() - 1) {
    return false;
  }
  ++pos;
  size_t extension_length = s1.size() - pos;
  return s1.compare(pos, extension_length, s2) == 0;
}

template <>
inline char GetDot<char>() {
  return '.';
}

template <>
inline wchar_t GetDot<wchar_t>() {
  return L'.';
}

#ifdef _WIN32
template <>
inline char GetPathSep<char>() {
  return '\\';
}

template <>
inline wchar_t GetPathSep<wchar_t>() {
  return L'\\';
}
#else
template <>
inline char GetPathSep<char>() {
  return '/';
}

template <>
inline wchar_t GetPathSep<wchar_t>() {
  return L'/';
}
#endif

template <typename PATH_CHAR_TYPE>
std::basic_string<PATH_CHAR_TYPE> ConcatPathComponent(const std::basic_string<PATH_CHAR_TYPE>& left,
                                                      const std::basic_string<PATH_CHAR_TYPE>& right) {
  std::basic_string<PATH_CHAR_TYPE> ret(left);
  ret.append(1, GetPathSep<PATH_CHAR_TYPE>()).append(right);
  return ret;
}

#if defined(_WIN32)
inline OrtFileType DTToFileType(DWORD dwFileAttributes) {
  if (dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    return OrtFileType::TYPE_DIR;
  }
  // TODO: test if it is reg
  return OrtFileType::TYPE_REG;
}
inline std::string FormatErrorCode(DWORD dw) {
  static constexpr DWORD bufferLength = 64 * 1024;
  std::string s(bufferLength, '\0');
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw,
                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)s.data(), bufferLength / sizeof(TCHAR), NULL);
  return s;
}
template <typename T>
void LoopDir(const std::wstring& dir_name, T func) {
  std::wstring pattern = dir_name + L"\\*";
  WIN32_FIND_DATAW ffd;
  std::unique_ptr<void, decltype(&FindClose)> hFind(FindFirstFileW(pattern.c_str(), &ffd), FindClose);
  if (hFind.get() == INVALID_HANDLE_VALUE) {
    DWORD dw = GetLastError();
    std::string s = FormatErrorCode(dw);
    ORT_THROW(s);
  }
  do {
    if (!func(ffd.cFileName, DTToFileType(ffd.dwFileAttributes))) return;
  } while (FindNextFileW(hFind.get(), &ffd) != 0);
  DWORD dwError = GetLastError();
  if (dwError != ERROR_NO_MORE_FILES) {
    DWORD dw = GetLastError();
    std::string s = FormatErrorCode(dw);
    ORT_THROW(s);
  }
}

// TODO: rewrite it with PathFindNextComponentW
inline std::basic_string<PATH_CHAR_TYPE> GetLastComponent(const std::basic_string<PATH_CHAR_TYPE>& s) {
  if (s.empty()) return std::basic_string<PATH_CHAR_TYPE>(1, GetDot<PATH_CHAR_TYPE>());
  std::basic_string<PATH_CHAR_TYPE> input = s;
  typename std::basic_string<PATH_CHAR_TYPE>::size_type pos = input.length();
  PATH_CHAR_TYPE sep = GetPathSep<PATH_CHAR_TYPE>();
  // remove trailing backslash
  for (; pos > 1 && input[pos - 1] == sep; --pos)
    ;
  input.resize(pos);
  for (; pos != 0 && input[pos - 1] != sep; --pos)
    ;
  return input.substr(pos);
}

#elif defined(_AIX)
inline OrtFileType DTToFileTypeAIX(struct stat st) {
  switch (st.st_mode & _S_IFMT) {
    case S_IFBLK:
      return OrtFileType::TYPE_BLK;
    case S_IFCHR:
      return OrtFileType::TYPE_CHR;
    case S_IFDIR:
      return OrtFileType::TYPE_DIR;
    case S_IFIFO:
      return OrtFileType::TYPE_FIFO;
    case S_IFLNK:
      return OrtFileType::TYPE_LNK;
    case S_IFREG:
      return OrtFileType::TYPE_REG;
    /* No Socket type */
    default:
      return OrtFileType::TYPE_UNKNOWN;
  }
}

template <typename T>
void LoopDir(const std::string& dir_name, T func) {
  DIR* dir = opendir(dir_name.c_str());
  struct stat stats;
  if (dir == nullptr) {
    auto e = errno;
    char buf[1024];
    char* msg;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    msg = strerror_r(e, buf, sizeof(buf));
#else
    if (strerror_r(e, buf, sizeof(buf)) != 0) {
      buf[0] = '\0';
    }
    msg = buf;
#endif
    std::ostringstream oss;
    oss << "couldn't open '" << dir_name << "':" << msg;
    std::string s = oss.str();
    ORT_THROW(s);
  }
  ORT_TRY {
    struct dirent* dp;
    while ((dp = readdir(dir)) != nullptr) {
      std::basic_string<PATH_CHAR_TYPE> filename = ConcatPathComponent<PATH_CHAR_TYPE>(dir_name, dp->d_name);
      if (stat(filename.c_str(), &stats) != 0) {
        continue;
      }
      if (!func(dp->d_name, DTToFileTypeAIX(stats))) {
        break;
      }
    }
  }
  ORT_CATCH(const std::exception& ex) {
    closedir(dir);
    ORT_RETHROW;
  }
  closedir(dir);
}
#else
inline OrtFileType DTToFileType(unsigned char t) {
  switch (t) {
    case DT_BLK:
      return OrtFileType::TYPE_BLK;
    case DT_CHR:
      return OrtFileType::TYPE_CHR;
    case DT_DIR:
      return OrtFileType::TYPE_DIR;
    case DT_FIFO:
      return OrtFileType::TYPE_FIFO;
    case DT_LNK:
      return OrtFileType::TYPE_LNK;
    case DT_REG:
      return OrtFileType::TYPE_REG;
    case DT_SOCK:
      return OrtFileType::TYPE_SOCK;
    default:
      return OrtFileType::TYPE_UNKNOWN;
  }
}

template <typename T>
void LoopDir(const std::string& dir_name, T func) {
  DIR* dir = opendir(dir_name.c_str());
  if (dir == nullptr) {
    auto e = errno;
    char buf[1024];
    char* msg;
#if defined(__GLIBC__) && defined(_GNU_SOURCE) && !defined(__ANDROID__)
    msg = strerror_r(e, buf, sizeof(buf));
#else
    if (strerror_r(e, buf, sizeof(buf)) != 0) {
      buf[0] = '\0';
    }
    msg = buf;
#endif
    std::ostringstream oss;
    oss << "couldn't open '" << dir_name << "':" << msg;
    std::string s = oss.str();
    ORT_THROW(s);
  }
  ORT_TRY {
    struct dirent* dp;
    while ((dp = readdir(dir)) != nullptr) {
      if (!func(dp->d_name, DTToFileType(dp->d_type))) {
        break;
      }
    }
  }
  ORT_CATCH(const std::exception& ex) {
    closedir(dir);
    ORT_RETHROW;
  }
  closedir(dir);
}
#endif
template <typename T>
inline T ReplaceFilename(const T& input, const T& new_value) {
  T ret;
  auto status = GetDirNameFromFilePath(input, ret);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return ConcatPathComponent(ret, new_value);
}

}  // namespace onnxruntime
