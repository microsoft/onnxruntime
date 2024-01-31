// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Span.hpp"

#include <fstream>
#include <string>
#include <type_traits>

namespace MILBlob {
namespace Blob {
/**
 * Utility for interfacing with files
 */
class FileWriter final {
public:
    FileWriter() = delete;
    FileWriter(const FileWriter&) = delete;
    FileWriter(FileWriter&&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    FileWriter& operator=(FileWriter&&) = delete;

    FileWriter(const std::string& filePath, bool truncateFile);
    ~FileWriter();

    /**
     * Appends given data to file at next aligned offset
     * @throws std::runtime_error if error occurs while writing to file stream
     */
    uint64_t AppendData(Util::Span<const uint8_t> data);

    /**
     * Writes data to given offset
     * @throws std::runtime_error if error occurs while writing to file stream or offset is not aligned
     */
    void WriteData(Util::Span<const uint8_t> data, uint64_t offset);

    /**
     * Returns next available aligned offset for writing
     */
    uint64_t GetNextAlignedOffset();

    /**
     * Returns size in byte of file currently open
     */
    uint64_t GetFileSize();

    /**
     * Reads data from current stream from given offset and writes into destData
     * @throws std:runtime_error if error occurs during reading data
     */
    void ReadData(uint64_t offset, Util::Span<uint8_t> destData);

private:
    std::fstream m_fileStream;
};

}  // namespace Blob
}  // namespace MILBlob
