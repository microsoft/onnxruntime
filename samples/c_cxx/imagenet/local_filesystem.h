// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <stdint.h>
#include <memory>
#include <sstream>
#ifdef _WIN32
#include <Windows.h>
#else
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <onnxruntime/core/session/onnxruntime_c_api.h>
void ReadFileAsString(const ORTCHAR_T* fname, void*& p, size_t& len);

enum class OrtFileType { TYPE_BLK, TYPE_CHR, TYPE_DIR, TYPE_FIFO, TYPE_LNK, TYPE_REG, TYPE_SOCK, TYPE_UNKNOWN };
using TCharString = std::basic_string<ORTCHAR_T>;

#ifdef _WIN32
inline OrtFileType DTToFileType(DWORD dwFileAttributes) {
  if (dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
    return OrtFileType::TYPE_DIR;
  }
  // TODO: test if it is reg
  return OrtFileType::TYPE_REG;
}

inline std::string FormatErrorCode(DWORD dw) {
  char* lpMsgBuf;
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw,
                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
  std::string s(lpMsgBuf);
  LocalFree(lpMsgBuf);
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
    throw std::runtime_error(s);
  }
  do {
    if (!func(ffd.cFileName, DTToFileType(ffd.dwFileAttributes))) return;
  } while (FindNextFileW(hFind.get(), &ffd) != 0);
  DWORD dwError = GetLastError();
  if (dwError != ERROR_NO_MORE_FILES) {
    DWORD dw = GetLastError();
    std::string s = FormatErrorCode(dw);
    throw std::runtime_error(s);
  }
}
#else

inline void ReportSystemError(const char* operation_name, const TCharString& path) {
  auto e = errno;
  char buf[1024];
  const char* msg = "";
  if (e > 0) {
#if defined(__GLIBC__) && defined(_GNU_SOURCE) && !defined(__ANDROID__)
    msg = strerror_r(e, buf, sizeof(buf));
#else
    // for Mac OS X and Android lower than API 23
    if (strerror_r(e, buf, sizeof(buf)) != 0) {
      buf[0] = '\0';
    }
    msg = buf;
#endif
  }
  std::ostringstream oss;
  oss << operation_name << " file \"" << path << "\" failed: " << msg;
  throw std::runtime_error(oss.str());
}

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
void LoopDir(const TCharString& dir_name, T func) {
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
    throw std::runtime_error(s);
  }
  try {
    struct dirent* dp;
    while ((dp = readdir(dir)) != nullptr) {
      if (!func(dp->d_name, DTToFileType(dp->d_type))) {
        break;
      }
    }
  } catch (std::exception& ex) {
    closedir(dir);
    throw;
  }
  closedir(dir);
}
#endif