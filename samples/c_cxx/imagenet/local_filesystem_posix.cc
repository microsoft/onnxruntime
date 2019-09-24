// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "local_filesystem.h"
#include <assert.h>
#include <mutex>

static std::mutex m;

void ReadFileAsString(const ORTCHAR_T* fname, void*& p, size_t& len) {
  std::lock_guard<std::mutex> g(m);
  if (!fname) {
    throw std::runtime_error("ReadFileAsString: 'fname' cannot be NULL");
  }
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return ReportSystemError("open", fname);
  }
  struct stat stbuf;
  if (fstat(fd, &stbuf) != 0) {
    return ReportSystemError("fstat", fname);
  }

  if (!S_ISREG(stbuf.st_mode)) {
    throw std::runtime_error("ReadFileAsString: input is not a regular file");
  }
  // TODO:check overflow
  len = static_cast<size_t>(stbuf.st_size);

  if (len == 0) {
    p = nullptr;
  } else {
    char* buffer = reinterpret_cast<char*>(malloc(len));
    char* wptr = reinterpret_cast<char*>(buffer);
    auto length_remain = len;
    do {
      size_t bytes_to_read = length_remain;
      ssize_t bytes_read;
      TEMP_FAILURE_RETRY(bytes_read = read(fd, wptr, bytes_to_read));
      if (bytes_read <= 0) {
        return ReportSystemError("read", fname);
      }
      assert(static_cast<size_t>(bytes_read) <= bytes_to_read);
      wptr += bytes_read;
      length_remain -= bytes_read;
    } while (length_remain > 0);
    p = buffer;
  }
  close(fd);
}
