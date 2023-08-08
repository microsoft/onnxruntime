// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"

#include <fstream>
#include <random>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // for sysconf() and _SC_PAGESIZE
#else
#include <Windows.h>
#endif

#include "core/common/gsl.h"

#include "gtest/gtest.h"

#include "core/common/span_utils.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {

namespace {
using PathString = std::basic_string<ORTCHAR_T>;

struct TempFilePath {
  TempFilePath(const PathString& base)
      : path{
            [&base]() {
              PathString path_template = base + ORT_TSTR("XXXXXX");
              int fd;
              CreateTestFile(fd, path_template);
#ifdef _WIN32
              _close(fd);
#else
              close(fd);
#endif
              return path_template;
            }()} {
  }

  ~TempFilePath() {
    DeleteFileFromDisk(path.c_str());
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TempFilePath);

  const PathString path;
};

std::vector<char> GenerateData(size_t length, uint32_t seed = 0) {
  auto engine = std::default_random_engine{seed};
  auto dist = std::uniform_int_distribution<int>{
      std::numeric_limits<char>::min(), std::numeric_limits<char>::max()};
  std::vector<char> result{};
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result.push_back(static_cast<char>(dist(engine)));
  }
  return result;
}

void WriteDataToFile(gsl::span<const char> data, const PathString& path) {
#ifndef _WIN32
  std::ofstream out{path, std::ios_base::out | std::ios_base::trunc};
#else
  std::ofstream out{path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary};
#endif
  out.write(data.data(), data.size());
}

std::vector<std::pair<FileOffsetType, size_t>> GenerateValidOffsetLengthPairs(size_t begin, size_t end, size_t interval = 1) {
  std::vector<std::pair<FileOffsetType, size_t>> offset_length_pairs;
  for (size_t range_begin = begin; range_begin < end; range_begin += interval) {
    for (size_t range_end = range_begin; range_end <= end; range_end += interval) {
      offset_length_pairs.emplace_back(static_cast<FileOffsetType>(range_begin), range_end - range_begin);
    }
  }
  return offset_length_pairs;
}
}  // namespace

TEST(FileIoTest, ReadFileIntoBuffer) {
  TempFilePath tmp(ORT_TSTR("read_test_"));
  const auto expected_data = GenerateData(32);
  WriteDataToFile(gsl::make_span(expected_data), tmp.path);

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(0, expected_data.size());

  std::vector<char> buffer(expected_data.size());
  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    auto buffer_span = gsl::make_span(buffer.data(), length);
    auto status = Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), offset, length, buffer_span);
    ASSERT_TRUE(status.IsOK())
        << "ReadFileIntoBuffer failed for offset " << offset << " and length " << length
        << " with error: " << status.ErrorMessage();

    auto expected_data_span = gsl::make_span(expected_data.data() + offset, length);

    ASSERT_TRUE(SpanEq(buffer_span, expected_data_span));
  }

  // invalid - negative offset
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), -1, 0, gsl::make_span(buffer)).IsOK());

  // invalid - length too long
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), 0, expected_data.size() + 1, gsl::make_span(buffer)).IsOK());

  // invalid - buffer too short
  ASSERT_FALSE(Env::Default().ReadFileIntoBuffer(tmp.path.c_str(), 0, 3, gsl::make_span(buffer.data(), 2)).IsOK());
}

#ifndef _WIN32  // not implemented on Windows
TEST(FileIoTest, MapFileIntoMemory) {
  static const auto page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);

  TempFilePath tmp(ORT_TSTR("map_file_test_"));
  const auto expected_data = GenerateData(page_size * 3 / 2);
  WriteDataToFile(gsl::make_span(expected_data), tmp.path);

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(0, expected_data.size(), page_size / 10);

  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    Env::MappedMemoryPtr mapped_memory{};
    auto status = Env::Default().MapFileIntoMemory(tmp.path.c_str(), offset, length, mapped_memory);
    ASSERT_TRUE(status.IsOK())
        << "MapFileIntoMemory failed for offset " << offset << " and length " << length
        << " with error: " << status.ErrorMessage();

    auto mapped_span = gsl::make_span(mapped_memory.get(), length);

    auto expected_data_span = gsl::make_span(expected_data.data() + offset, length);

    ASSERT_TRUE(SpanEq(mapped_span, expected_data_span));
  }

  {
    Env::MappedMemoryPtr mapped_memory{};

    // invalid - negative offset
    ASSERT_FALSE(Env::Default().MapFileIntoMemory(tmp.path.c_str(), -1, 0, mapped_memory).IsOK());
  }
}
#else
TEST(FileIoTest, MapFileIntoMemory) {
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  static const auto page_size = sysinfo.dwPageSize;
  static const auto allocation_granularity = sysinfo.dwAllocationGranularity;
  ASSERT_GT(page_size, static_cast<DWORD>(0));

  TempFilePath tmp(ORT_TSTR("map_file_test_"));
  const auto expected_data = GenerateData(page_size * 3 / 2);
  WriteDataToFile(gsl::make_span(expected_data), tmp.path);

  const auto offsets_and_lengths = GenerateValidOffsetLengthPairs(
      0, expected_data.size(), page_size / 10);

  for (const auto& offset_and_length : offsets_and_lengths) {
    const auto offset = offset_and_length.first;
    const auto length = offset_and_length.second;

    // The offset must be a multiple of the allocation granularity
    if (offset % allocation_granularity != 0) {
      continue;
    }

    Env::MappedMemoryPtr mapped_memory{};
    auto status = Env::Default().MapFileIntoMemory(
        tmp.path.c_str(), offset, length, mapped_memory);
    ASSERT_TRUE(status.IsOK())
        << "MapFileIntoMemory failed for offset " << offset << " and length " << length
        << " with error: " << status.ErrorMessage();

    auto mapped_span = gsl::make_span(mapped_memory.get(), length);

    auto expected_data_span = gsl::make_span(expected_data.data() + offset, length);

    ASSERT_TRUE(SpanEq(mapped_span, expected_data_span));
  }

  {
    Env::MappedMemoryPtr mapped_memory{};

    // invalid - offset is not a multiple of the allocation granularity
    ASSERT_FALSE(Env::Default().MapFileIntoMemory(
                                   tmp.path.c_str(), allocation_granularity * 3 / 2, page_size / 10, mapped_memory)
                     .IsOK());
  }

  {
    Env::MappedMemoryPtr mapped_memory{};

    // invalid - negative offset
    ASSERT_FALSE(Env::Default().MapFileIntoMemory(tmp.path.c_str(), -1, 0, mapped_memory).IsOK());
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
