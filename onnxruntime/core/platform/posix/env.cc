/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <string.h>
#include <thread>
#include <vector>
#include <assert.h>
#include "core/platform/env.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"

// MAC OS X doesn't have this macro
#ifndef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(X) X
#endif

namespace onnxruntime {

namespace {
constexpr int OneMillion = 1000000;

static void ORT_API_CALL DeleteBuffer(void* param) noexcept { ::free(param); }

class UnmapFileParam {
 public:
  void* addr;
  size_t len;
  int fd;
};

static void ORT_API_CALL UnmapFile(void* param) noexcept {
  UnmapFileParam* p = reinterpret_cast<UnmapFileParam*>(param);
  int ret = munmap(p->addr, p->len);
  if (ret != 0) {
    int err = errno;
    LOGS_DEFAULT(INFO) << "munmap failed. error code:" << err;
  }
  (void)close(p->fd);
  delete p;
}

class PosixEnv : public Env {
 public:
  static PosixEnv& Instance() {
    static PosixEnv default_env;
    return default_env;
  }

  int GetNumCpuCores() const override {
    // TODO if you need the number of physical cores you'll need to parse
    // /proc/cpuinfo and grep for "cpu cores".
    //However, that information is not always available(output of 'grep -i core /proc/cpuinfo' is empty)
    return std::thread::hardware_concurrency();
  }

  void SleepForMicroseconds(int64_t micros) const override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= OneMillion) {
        sleep_time.tv_sec = std::min<int64_t>(micros / OneMillion, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64_t>(sleep_time.tv_sec) * OneMillion;
      }
      if (micros < OneMillion) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  PIDType GetSelfPid() const override {
    return getpid();
  }

  static common::Status ReadBinaryFile(int fd, off_t offset, const char* fname, void*& p, size_t len,
                                       OrtCallback& deleter) {
    std::unique_ptr<char[]> buffer(reinterpret_cast<char*>(malloc(len)));
    char* wptr = reinterpret_cast<char*>(buffer.get());
    auto length_remain = len;
    do {
      size_t bytes_to_read = length_remain;
      ssize_t bytes_read;
      TEMP_FAILURE_RETRY(bytes_read =
                             offset > 0 ? pread(fd, wptr, bytes_to_read, offset) : read(fd, wptr, bytes_to_read));
      if (bytes_read <= 0) {
        int err = errno;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "read file '", fname, "' fail, error code = ", err);
      }
      assert(static_cast<size_t>(bytes_read) <= bytes_to_read);
      wptr += bytes_read;
      length_remain -= bytes_read;
    } while (length_remain > 0);
    p = buffer.release();
    deleter.f = DeleteBuffer;
    deleter.param = p;
    return Status::OK();
  }

  common::Status ReadFileAsString(const char* fname, off_t offset, void*& p, size_t& len,
      OrtCallback& deleter) const override {
    if (!fname) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "ReadFileAsString: 'fname' cannot be NULL");
    }

    if (offset < 0) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "ReadFileAsString: offset must be non-negative");
    }
    deleter.f = nullptr;
    deleter.param = nullptr;
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
      return ReportSystemError("open", fname);
    }
    if (len <= 0) {
      struct stat stbuf;
      if (fstat(fd, &stbuf) != 0) {
        return ReportSystemError("fstat", fname);
      }

      if (!S_ISREG(stbuf.st_mode)) {
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                              "ReadFileAsString: input is not a regular file");
      }
      len = static_cast<size_t>(stbuf.st_size);
    }

    if (len == 0) {
      p = nullptr;
    } else {
      long page_size = sysconf(_SC_PAGESIZE);
      off_t offset_to_page = offset % static_cast<off_t>(page_size);
      p = mmap(nullptr, len + offset_to_page, PROT_READ, MAP_SHARED, fd, offset - offset_to_page);
      if (p == MAP_FAILED) {
        auto st = ReadBinaryFile(fd, offset, fname, p, len, deleter);
        (void)close(fd);
        if (!st.IsOK()) {
          return st;
        }
      } else {
        // leave the file open
        deleter.f = UnmapFile;
        deleter.param = new UnmapFileParam{p, len + offset_to_page, fd};
        p = reinterpret_cast<char*>(p) + offset_to_page;
      }
    }

    return common::Status::OK();
  }

  static common::Status ReportSystemError(const char* operation_name, const std::string& path) {
    auto e = errno;
    char buf[1024];
    const char* msg = "";
    if (e > 0) {
#if defined(_GNU_SOURCE) && !defined(__APPLE__)
      msg = strerror_r(e, buf, sizeof(buf));
#else
      // for Mac OS X
      if (strerror_r(e, buf, sizeof(buf)) != 0) {
        buf[0] = '\0';
      }
      msg = buf;
#endif
    }
    std::ostringstream oss;
    oss << operation_name << " file \"" << path << "\" failed: " << msg;
    return common::Status(common::SYSTEM, e, oss.str());
  }

  common::Status FileOpenRd(const std::string& path, /*out*/ int& fd) const override {
    fd = open(path.c_str(), O_RDONLY);
    if (0 > fd) {
      return ReportSystemError("open", path);
    }
    return Status::OK();
  }

  common::Status FileOpenWr(const std::string& path, /*out*/ int& fd) const override {
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (0 > fd) {
      return ReportSystemError("open", path);
    }
    return Status::OK();
  }

  common::Status FileClose(int fd) const override {
    int ret = close(fd);
    if (0 != ret) {
      return ReportSystemError("close", "");
    }
    return Status::OK();
  }

  common::Status LoadDynamicLibrary(const std::string& library_filename, void** handle) const override {
    char* error_str = dlerror();  // clear any old error_str
    *handle = dlopen(library_filename.c_str(), RTLD_NOW | RTLD_LOCAL);
    error_str = dlerror();
    if (!*handle) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to load library " + library_filename + " with error: " + error_str);
    }
    return common::Status::OK();
  }

  common::Status UnloadDynamicLibrary(void* handle) const override {
    if (!handle) {
      return common::Status(common::ONNXRUNTIME, common::FAIL, "Got null library handle");
    }
    char* error_str = dlerror();  // clear any old error_str
    int retval = dlclose(handle);
    error_str = dlerror();
    if (retval != 0) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to unload library with error: " + std::string(error_str));
    }
    return common::Status::OK();
  }

  common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    char* error_str = dlerror();  // clear any old error str
    *symbol = dlsym(handle, symbol_name.c_str());
    error_str = dlerror();
    if (error_str) {
      return common::Status(common::ONNXRUNTIME, common::FAIL,
                            "Failed to get symbol " + symbol_name + " with error: " + error_str);
    }
    // it's possible to get a NULL symbol in our case when Schemas are not custom.
    return common::Status::OK();
  }

  std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override {
    std::string filename;
    if (version.empty()) {
      filename = "lib" + name + ".so";
    } else {
      filename = "lib" + name + ".so" + "." + version;
    }
    return filename;
  }

 private:
  PosixEnv() = default;
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
// REGISTER_FILE_SYSTEM("", PosixFileSystem);
// REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
const Env& Env::Default() {
  return PosixEnv::Instance();
}
#endif

}  // namespace onnxruntime
