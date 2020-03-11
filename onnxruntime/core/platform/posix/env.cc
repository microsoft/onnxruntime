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

#include "core/platform/env.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <ftw.h>
#include <string.h>
#include <thread>
#include <utility>  // for std::forward
#include <vector>
#include <assert.h>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/scoped_resource.h"

namespace onnxruntime {

namespace {

constexpr int OneMillion = 1000000;

class UnmapFileParam {
 public:
  void* addr;
  size_t len;
};

static void UnmapFile(void* param) noexcept {
  UnmapFileParam* p = reinterpret_cast<UnmapFileParam*>(param);
  int ret = munmap(p->addr, p->len);
  if (ret != 0) {
    int err = errno;
    LOGS_DEFAULT(ERROR) << "munmap failed. error code: " << err;
  }
  delete p;
}

struct FileDescriptorTraits {
  using Handle = int;
  static Handle GetInvalidHandleValue() { return -1; }
  static void CleanUp(Handle h) {
    if (close(h) == -1) {
      const int err = errno;
      LOGS_DEFAULT(ERROR) << "Failed to close file descriptor " << h << " - error code: " << err;
    }
  }
};

// Note: File descriptor cleanup may fail but this class doesn't expose a way to check if it failed.
//       If that's important, consider using another cleanup method.
using ScopedFileDescriptor = ScopedResource<FileDescriptorTraits>;

// non-macro equivalent of TEMP_FAILURE_RETRY, described here:
// https://www.gnu.org/software/libc/manual/html_node/Interrupted-Primitives.html
template <typename TFunc, typename... TFuncArgs>
long int TempFailureRetry(TFunc retriable_operation, TFuncArgs&&... args) {
  long int result;
  do {
    result = retriable_operation(std::forward<TFuncArgs>(args)...);
  } while (result == -1 && errno == EINTR);
  return result;
}

int nftw_remove(
    const char* fpath, const struct stat* /*sb*/,
    int /*typeflag*/, struct FTW* /*ftwbuf*/) {
  const auto result = remove(fpath);
  if (result != 0) {
    const int err = errno;
    LOGS_DEFAULT(WARNING) << "remove() failed. Error code: " << err
                          << ", path: " << fpath;
  }
  return result;
}

template <typename T>
struct Freer {
  void operator()(T* p) { ::free(p); }
};

using MallocdStringPtr = std::unique_ptr<char, Freer<char> >;

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

  Status GetFileLength(const ORTCHAR_T* file_path, size_t& length) const override {
    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    if (file_descriptor.Get() < 0) {
      return ReportSystemError("open", file_path);
    }

    struct stat stbuf;
    if (fstat(file_descriptor.Get(), &stbuf) != 0) {
      return ReportSystemError("fstat", file_path);
    }

    if (!S_ISREG(stbuf.st_mode)) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                            "GetFileLength: input is not a regular file");
    }

    length = static_cast<size_t>(stbuf.st_size);
    return Status::OK();
  }

  Status ReadFileIntoBuffer(
      const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
      gsl::span<char> buffer) const override {
    ORT_RETURN_IF_NOT(file_path);
    ORT_RETURN_IF_NOT(offset >= 0);
    ORT_RETURN_IF_NOT(length <= buffer.size());

    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    if (!file_descriptor.IsValid()) {
      return ReportSystemError("open", file_path);
    }

    if (length == 0) return Status::OK();

    if (offset > 0) {
      const FileOffsetType seek_result = lseek(file_descriptor.Get(), offset, SEEK_SET);
      if (seek_result == -1) {
        return ReportSystemError("lseek", file_path);
      }
    }

    size_t total_bytes_read = 0;
    while (total_bytes_read < length) {
      constexpr size_t k_max_bytes_to_read = 1 << 30;  // read at most 1GB each time
      const size_t bytes_remaining = length - total_bytes_read;
      const size_t bytes_to_read = std::min(bytes_remaining, k_max_bytes_to_read);

      const ssize_t bytes_read = TempFailureRetry(read, file_descriptor.Get(), buffer.data() + total_bytes_read, bytes_to_read);

      if (bytes_read == -1) {
        return ReportSystemError("read", file_path);
      }

      if (bytes_read == 0) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL, "ReadFileIntoBuffer - unexpected end of file. ",
            "File: ", file_path, ", offset: ", offset, ", length: ", length);
      }

      total_bytes_read += bytes_read;
    }

    return Status::OK();
  }

  Status MapFileIntoMemory(
      const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
      MappedMemoryPtr& mapped_memory) const override {
    ORT_RETURN_IF_NOT(file_path);
    ORT_RETURN_IF_NOT(offset >= 0);

    ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
    if (!file_descriptor.IsValid()) {
      return ReportSystemError("open", file_path);
    }

    if (length == 0) {
      mapped_memory = MappedMemoryPtr{};
      return Status::OK();
    }

    static const long page_size = sysconf(_SC_PAGESIZE);
    const FileOffsetType offset_to_page = offset % static_cast<FileOffsetType>(page_size);
    const size_t mapped_length = length + offset_to_page;
    const FileOffsetType mapped_offset = offset - offset_to_page;
    void* const mapped_base = mmap(
        nullptr, mapped_length,
        PROT_READ | PROT_WRITE, MAP_PRIVATE,
        file_descriptor.Get(), mapped_offset);

    if (mapped_base == MAP_FAILED) {
      return ReportSystemError("mmap", file_path);
    }

    mapped_memory = MappedMemoryPtr{
        reinterpret_cast<char*>(mapped_base) + offset_to_page,
        OrtCallbackInvoker{OrtCallback{UnmapFile, new UnmapFileParam{mapped_base, mapped_length}}}};

    return Status::OK();
  }

  static common::Status ReportSystemError(const char* operation_name, const std::string& path) {
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
    return common::Status(common::SYSTEM, e, oss.str());
  }

  bool FolderExists(const std::string& path) const override {
    struct stat sb;
    if (stat(path.c_str(), &sb)) {
      return false;
    }
    return S_ISDIR(sb.st_mode);
  }

  common::Status CreateFolder(const std::string& path) const override {
    size_t pos = 0;
    do {
      pos = path.find_first_of("\\/", pos + 1);
      std::string directory = path.substr(0, pos);
      if (FolderExists(directory.c_str())) {
        continue;
      }
      if (mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        return common::Status(common::SYSTEM, errno);
      }
    } while (pos != std::string::npos);
    return Status::OK();
  }

  common::Status DeleteFolder(const std::string& path) const override {
    const auto result = nftw(
        path.c_str(), &nftw_remove, 32, FTW_DEPTH | FTW_PHYS);
    ORT_RETURN_IF_NOT(
        result == 0, "DeleteFolder(): nftw() failed with error: ", result);
    return Status::OK();
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

  common::Status GetCanonicalPath(
      const std::basic_string<ORTCHAR_T>& path,
      std::basic_string<ORTCHAR_T>& canonical_path) const override {
    MallocdStringPtr canonical_path_cstr{realpath(path.c_str(), nullptr)};
    if (!canonical_path_cstr) {
      return ReportSystemError("realpath", path);
    }
    canonical_path.assign(canonical_path_cstr.get());
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

  // \brief returns a provider that will handle telemetry on the current platform
  const Telemetry& GetTelemetryProvider() const override {
    return telemetry_provider_;
  }

  // \brief returns a value for the queried variable name (var_name)
  std::string GetEnvironmentVar(const std::string& var_name) const override {
    char* val = getenv(var_name.c_str());
    return val == NULL ? std::string() : std::string(val);
  }

 private:
  PosixEnv() = default;
  Telemetry telemetry_provider_;
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
