// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/FileWriter.hpp"
#include "MILBlob/Blob/StorageFormat.hpp"

#include <cstdio>
#include <stdexcept>

// ORT_EDIT: Exclude mmap on Windows. Not used in this file anyway.
#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#endif

using namespace MILBlob;
using namespace MILBlob::Blob;
using namespace MILBlob::Util;

namespace {
std::ios_base::openmode GetWriterMode(bool truncate) {
  std::ios_base::openmode result = (std::ios::in | std::ios::out | std::ios::binary);
  if (truncate) {
    result |= std::ios::trunc;
  }
  return result;
}
}  // anonymous namespace

FileWriter::~FileWriter() = default;

FileWriter::FileWriter(const std::string& filePath, bool truncateFile) {
  m_fileStream.open(filePath, GetWriterMode(truncateFile));
  if (!m_fileStream) {
    // If file does not exists, ios::in does not create one
    // Let's create a file and re-open with required flags
    m_fileStream.open(filePath, std::ofstream::binary | std::ios::out);
    m_fileStream.close();
    m_fileStream.open(filePath, GetWriterMode(truncateFile));
  }
  MILVerifyIsTrue(m_fileStream,
                  std::runtime_error,
                  "[MIL FileWriter]: Unable to open " + filePath + " file stream for writing");
}

uint64_t FileWriter::GetNextAlignedOffset() {
  m_fileStream.seekg(0, std::ios::end);
  uint64_t offset = static_cast<uint64_t>(m_fileStream.tellg());
  if (offset % DefaultStorageAlignment == 0) {
    return offset;
  }
  auto pad = DefaultStorageAlignment - (offset % DefaultStorageAlignment);
  return offset + pad;
}

uint64_t FileWriter::GetFileSize() {
  m_fileStream.seekg(0, std::ios::end);
  return static_cast<uint64_t>(m_fileStream.tellg());
}

uint64_t FileWriter::AppendData(Span<const uint8_t> data) {
  auto offset = GetNextAlignedOffset();
  m_fileStream.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  m_fileStream.write(reinterpret_cast<const char*>(data.Data()), static_cast<std::streamsize>(data.Size()));
  MILVerifyIsTrue(m_fileStream.good(),
                  std::runtime_error,
                  "[MIL FileWriter]: Unknown error occured while writing data to the file.");
  return offset;
}

void FileWriter::WriteData(Span<const uint8_t> data, uint64_t offset) {
  MILVerifyIsTrue(offset % DefaultStorageAlignment == 0,
                  std::runtime_error,
                  "[MIL FileWriter]: Provided offset not aligned. offset=" + std::to_string(offset) +
                      " alignment=" + std::to_string(DefaultStorageAlignment) + ".");
  m_fileStream.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  m_fileStream.write(reinterpret_cast<const char*>(data.Data()), static_cast<std::streamsize>(data.Size()));
  MILVerifyIsTrue(m_fileStream.good(),
                  std::runtime_error,
                  "[MIL FileWriter]: Unknown error occured while writing data to the file.");
}

void FileWriter::ReadData(uint64_t offset, Util::Span<uint8_t> destData) {
  m_fileStream.seekg(static_cast<std::streamsize>(offset), std::ios::beg);
  m_fileStream.read(reinterpret_cast<char*>(destData.Data()), static_cast<std::streamsize>(destData.Size()));
  MILVerifyIsTrue(m_fileStream.good(),
                  std::runtime_error,
                  "[MIL FileWriter]: Unknown error occured while reading data from the file.");
}
