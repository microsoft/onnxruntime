// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Bf16.hpp"
#include "MILBlob/Blob/FileWriter.hpp"
#include "MILBlob/Blob/StorageFormat.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Fp16.hpp"
#include "MILBlob/Util/Span.hpp"
#include "MILBlob/Util/SpanCast.hpp"

#include <string>
#include <unordered_map>

using namespace MILBlob;
using namespace MILBlob::Blob;

namespace {
template <typename T>
Util::Span<uint8_t> CastAndMakeSpan(T& x)
{
    return Util::Span<uint8_t>(reinterpret_cast<uint8_t*>(&x), sizeof(x));
}
}  // anonymous namespace

class StorageWriter::Impl final {
public:
    Impl(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl& operator=(Impl&&) = delete;

    ~Impl() = default;

    Impl(const std::string& filePath, bool truncateFile)
        : m_filePath(filePath)
        , m_fileWriter(std::make_unique<FileWriter>(filePath, truncateFile))
    {
        if (truncateFile) {
            m_fileWriter->WriteData(CastAndMakeSpan(m_header), 0);
        } else {
            auto fileSize = m_fileWriter->GetFileSize();
            if (fileSize == 0) {
                // File exists and is empty
                m_fileWriter->WriteData(CastAndMakeSpan(m_header), 0);
            } else if (static_cast<size_t>(fileSize) >= sizeof(m_header)) {
                m_fileWriter->ReadData(0, CastAndMakeSpan(m_header));
                if (m_header.version != 2) {
                    // File exists and header is incorrect
                    // File is not empty, please use truncate option
                    throw std::runtime_error(
                        "[MIL StorageWriter]: Incorrect file header, please use truncateFile=true");
                }
            } else {
                // File is not empty, please use truncate option
                throw std::runtime_error("[MIL StorageWriter]: Incorrect file header, please use truncateFile=true");
            }
        }
    }

    template <typename T>
    uint64_t WriteData(Util::Span<const T> data);

    std::string GetFilePath() const
    {
        return m_filePath;
    }

private:
    std::string m_filePath;
    std::unique_ptr<FileWriter> m_fileWriter;
    storage_header m_header;
};

template <typename T>
uint64_t StorageWriter::Impl::WriteData(Util::Span<const T> data)
{
    // 1. Write data
    blob_metadata metadata;
    metadata.mil_dtype = BlobDataTypeTraits<typename std::remove_const<T>::type>::DataType;
    metadata.sizeInBytes = data.Size() * sizeof(T);
    // Get offset for data
    auto metadataOffset = m_fileWriter->GetNextAlignedOffset();
    // metadata is 64 bit aligned.
    auto dataOffset = metadataOffset + sizeof(metadata);
    MILVerifyIsTrue(dataOffset % DefaultStorageAlignment == 0,
                    std::runtime_error,
                    "[MIL StorageWriter]: dataOffset is expected to be 64 bits aligned.");
    metadata.offset = dataOffset;
    // We don't expect m_fileWriter to produce different offset for metadata and data
    auto actualMetadataOffset = m_fileWriter->AppendData(CastAndMakeSpan(metadata));
    MILVerifyIsTrue(metadataOffset == actualMetadataOffset,
                    std::runtime_error,
                    "[MIL StorageWriter]: Metadata written to different offset than expected.");
    auto actualDataOffset = m_fileWriter->AppendData(Util::SpanCast<const uint8_t>(data));
    MILVerifyIsTrue(dataOffset == actualDataOffset,
                    std::runtime_error,
                    "[MIL StorageWriter]: Metadata written to different offset than expected.");

    // 2. Update count in header
    m_header.count++;
    // Write header with new count
    m_fileWriter->WriteData(CastAndMakeSpan(m_header), 0);
    // return offset in file to blob_metadata
    return metadataOffset;
}

// --------------------------------------------------------------------------------------

StorageWriter::~StorageWriter() = default;

StorageWriter::StorageWriter(const std::string& filePath, bool truncateFile)
    : m_impl(std::make_unique<Impl>(filePath, truncateFile))
{}

template <>
uint64_t StorageWriter::WriteData<int8_t>(Util::Span<const int8_t> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<uint8_t>(Util::Span<const uint8_t> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<Bf16>(Util::Span<const Bf16> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<Fp16>(Util::Span<const Fp16> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<float>(Util::Span<const float> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<int16_t>(Util::Span<const int16_t> data)
{
    return m_impl->WriteData(data);
}

template <>
uint64_t StorageWriter::WriteData<uint16_t>(Util::Span<const uint16_t> data)
{
    return m_impl->WriteData(data);
}

std::string StorageWriter::GetFilePath() const
{
    return m_impl->GetFilePath();
}
