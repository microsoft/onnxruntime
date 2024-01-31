// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Bf16.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/Span.hpp"
#include <memory>
#include <string>

namespace MILBlob {
namespace Blob {

/**
 * Utility for writing MIL Blob Storage format
 * details of new file format: MIL/Blob/StorageFormat.hpp
 */
class StorageWriter final {
public:
    StorageWriter() = delete;
    StorageWriter(const StorageWriter&) = delete;
    StorageWriter(StorageWriter&&) = delete;
    StorageWriter& operator=(const StorageWriter&) = delete;
    StorageWriter& operator=(StorageWriter&&) = delete;

    StorageWriter(const std::string& filePath, bool truncateFile = true);
    ~StorageWriter();

    /**
     * Writes data to the next available aligned location into opened file stream
     * Writes blob_metadata followed by data (both at next aligned offset specified by MILBlob::Blob::DefaultAlignment)
     * @throws std::runtime_error if error occurs while writing data to file
     */
    template <typename T>
    uint64_t WriteData(Util::Span<const T> data);

    /**
     * Returns the file path of the blob storage file.
     */
    std::string GetFilePath() const;

private:
    class Impl;
    const std::unique_ptr<Impl> m_impl;
};

template <>
uint64_t StorageWriter::WriteData<int8_t>(Util::Span<const int8_t>);
template <>
uint64_t StorageWriter::WriteData<uint8_t>(Util::Span<const uint8_t>);
template <>
uint64_t StorageWriter::WriteData<Bf16>(Util::Span<const Bf16>);
template <>
uint64_t StorageWriter::WriteData<Fp16>(Util::Span<const Fp16>);
template <>
uint64_t StorageWriter::WriteData<float>(Util::Span<const float>);
template <>
uint64_t StorageWriter::WriteData<int16_t>(Util::Span<const int16_t>);
template <>
uint64_t StorageWriter::WriteData<uint16_t>(Util::Span<const uint16_t>);

}  // namespace Blob
}  // namespace MILBlob
