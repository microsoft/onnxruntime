#include "file_util.h"

#include <gtest/gtest.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#include <Windows.h>
#include <fcntl.h>
#endif

namespace onnxruntime {
namespace test {

void DeleteFileFromDisk(const ORTCHAR_T* path) {
#ifdef _WIN32
  ASSERT_EQ(TRUE, DeleteFileW(path));
#else
  ASSERT_EQ(0, unlink(path));
#endif
}
void CreateTestFile(int& out, std::basic_string<ORTCHAR_T>& filename_template) {
  if (filename_template.empty())
    ORT_THROW_EX(std::runtime_error, "file name template can't be empty");

  ORTCHAR_T* filename = const_cast<ORTCHAR_T*>(filename_template.c_str());
#ifdef _WIN32
  ASSERT_EQ(0, _wmktemp_s(filename, filename_template.length() + 1));
  int fd;
  int err = _wsopen_s(&fd, filename, _O_CREAT | _O_EXCL | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYRW, _S_IREAD | _S_IWRITE);
  if (err != 0)
    ORT_THROW_EX(std::runtime_error, "open temp file failed");
#else
  int fd = mkstemp(filename);
  if (fd < 0) {
    ORT_THROW_EX(std::runtime_error, "open temp file failed");
  }
#endif
  out = fd;
}
void CreateTestFile(FILE*& out, std::basic_string<ORTCHAR_T>& filename_template) {
  if (filename_template.empty())
    ORT_THROW_EX(std::runtime_error, "file name template can't be empty");

  ORTCHAR_T* filename = const_cast<ORTCHAR_T*>(filename_template.c_str());
#ifdef _WIN32
  ASSERT_EQ(0, _wmktemp_s(filename, filename_template.length() + 1));
  FILE* fp = nullptr;
  ASSERT_EQ(0, _wfopen_s(&fp, filename, ORT_TSTR("wb")));
#else
  int fd = mkstemp(filename);
  if (fd < 0) {
    ORT_THROW_EX(std::runtime_error, "open temp file failed");
  }
  FILE* fp = fdopen(fd, "w");
#endif
  out = fp;
}
}  // namespace test
}  // namespace onnxruntime
