// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "local_filesystem.h"
#include <assert.h>
#include <mutex>

static std::mutex m;

void ReadFileAsString(const ORTCHAR_T* fname, void*& p, size_t& len) {
  if (!fname) {
    throw std::runtime_error("ReadFileAsString: 'fname' cannot be NULL");
  }

  HANDLE hFile = CreateFileW(fname, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    int err = GetLastError();
    std::ostringstream oss;
    oss << "open file " << fname << " fail, errcode =" << err;
    throw std::runtime_error(oss.str().c_str());
  }
  std::unique_ptr<void, decltype(&CloseHandle)> handler_holder(hFile, CloseHandle);
  LARGE_INTEGER filesize;
  if (!GetFileSizeEx(hFile, &filesize)) {
    int err = GetLastError();
    std::ostringstream oss;
    oss << "GetFileSizeEx file " << fname << " fail, errcode =" << err;
    throw std::runtime_error(oss.str().c_str());
  }
  if (static_cast<ULONGLONG>(filesize.QuadPart) > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("ReadFileAsString: File is too large");
  }
  len = static_cast<size_t>(filesize.QuadPart);
  // check the file file for avoiding allocating a zero length buffer
  if (len == 0) {  // empty file
    p = nullptr;
    len = 0;
    return;
  }
  std::unique_ptr<char[]> buffer(reinterpret_cast<char*>(malloc(len)));
  char* wptr = reinterpret_cast<char*>(buffer.get());
  size_t length_remain = len;
  DWORD bytes_read = 0;
  for (; length_remain > 0; wptr += bytes_read, length_remain -= bytes_read) {
    // read at most 1GB each time
    DWORD bytes_to_read;
    if (length_remain > (1 << 30)) {
      bytes_to_read = 1 << 30;
    } else {
      bytes_to_read = static_cast<DWORD>(length_remain);
    }
    if (ReadFile(hFile, wptr, bytes_to_read, &bytes_read, nullptr) != TRUE) {
      int err = GetLastError();
      p = nullptr;
      len = 0;
      std::ostringstream oss;
      oss << "ReadFile " << fname << " fail, errcode =" << err;
      throw std::runtime_error(oss.str().c_str());
    }
    if (bytes_read != bytes_to_read) {
      p = nullptr;
      len = 0;
      std::ostringstream oss;
      oss << "ReadFile " << fname << " fail: unexpected end";
      throw std::runtime_error(oss.str().c_str());
    }
  }
  p = buffer.release();
  return;
}
