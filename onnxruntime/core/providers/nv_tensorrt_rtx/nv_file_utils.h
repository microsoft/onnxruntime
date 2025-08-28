#pragma once
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace file_utils {

class MappedFileView {
 public:
  void Unmap() {
#if defined(_WIN32)
    if (mapped_) UnmapViewOfFile(mapped_);
    if (hMap_) CloseHandle(hMap_);
    if (hFile_) CloseHandle(hFile_);
    mapped_ = nullptr;
    hMap_ = nullptr;
    hFile_ = nullptr;
#else
    if (mapped_ && size_) munmap(mapped_, size_);
    if (fd_ != -1) close(fd_);
    mapped_ = nullptr;
    fd_ = -1;
#endif
    data_ = nullptr;
    size_ = 0;
  }

  const void* data() { return data_; }

  size_t size() { return size_; }

  bool empty() { return size_ == 0 || data_ == nullptr; }

  MappedFileView(const std::string& path) {
    data_ = nullptr;
    size_ = 0;

    if (!std::filesystem::exists(path)) return;
#if defined(_WIN32)
    hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile_ == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile_, &fileSize)) {
      CloseHandle(hFile_);
      return;
    }
    size_t temp_size = static_cast<size_t>(fileSize.QuadPart);
    hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap_) {
      CloseHandle(hFile_);
      return;
    }
    mapped_ = MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0);
    if (!mapped_) {
      CloseHandle(hMap_);
      CloseHandle(hFile_);
      return;
    }
#else
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) return view;
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close(fd_);
      return;
    }
    size_t temp_size = sb.st_size;
    if (size_ > 0) {
      mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (mapped_ == MAP_FAILED) {
        close(fd_);
        return;
      }
    }
#endif
    size_ = temp_size;
    data_ = static_cast<const char*>(mapped_);
  }

  ~MappedFileView() { Unmap(); }

 private:
  const char* data_ = nullptr;
  size_t size_ = 0;
#if defined(_WIN32)
  HANDLE hFile_ = nullptr;
  HANDLE hMap_ = nullptr;
#else
  int fd_ = -1;
#endif
  void* mapped_ = nullptr;
};

inline std::vector<char> ReadFile(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    LOGS_DEFAULT(INFO) << "TensorRT RTX could not find the file and will create a new one " << path << std::endl;
    return {};
  }
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    ORT_THROW("Failed to open file: " + path);
  }
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (size > 0 && !file.read(buffer.data(), size)) {
    ORT_THROW("Failed to read file: " + path);
  }
  return buffer;
}

inline void WriteFile(const std::string& path, const void* data, size_t size) {
  if (std::filesystem::exists(path)) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
      ORT_THROW("Failed to open file for writing: " + path);
    }
    file.write(static_cast<const char*>(data), size);
  } else {
    LOGS_DEFAULT(INFO) << "TensorRT RTX a new file cache was written to " << path << std::endl;
    // Create new file
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file) {
      ORT_THROW("Failed to create file: " + path);
    }
    file.write(static_cast<const char*>(data), size);
  }
}

inline void WriteFile(const std::string& path, const std::vector<char>& data) { WriteFile(path, data.data(), data.size()); }

inline std::string VerifyPathAndMakeAbsolute(const std::string& path) {
  std::filesystem::path p(path);
  std::filesystem::path abs_path = std::filesystem::absolute(p);

  if (std::filesystem::exists(abs_path)) {
    // Path exists, check if it's writable
    std::ofstream test(abs_path.string(), std::ios::app | std::ios::binary);
    if (test.is_open()) {
      return abs_path.string();
    }
  } else {
    // Path does not exist, check if parent directory is writable
    auto parent = abs_path.parent_path();
    if (parent.empty()) parent = std::filesystem::current_path();
    std::ofstream test(abs_path.string(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (test.is_open()) {
      test.close();
      std::filesystem::remove(abs_path);  // Clean up test file
      return abs_path.string();
    }
  }
  LOGS_DEFAULT(INFO) << "TensorRT RTX the given path '" << path << "' could no be verified and written to as absolute path: '" << abs_path.string() << "'" << std::endl;
  return "";
}

}  // namespace file_utils
}  // namespace onnxruntime