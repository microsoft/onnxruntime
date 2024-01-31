// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Util/Span.hpp"

#include <functional>
#include <memory>
#include <string>

namespace MILBlob {
namespace Blob {

/**
 * Memory-mapped file reader.
 */
class MMapFileReader {
public:
    MMapFileReader() = delete;
    MMapFileReader(const MMapFileReader&) = delete;
    MMapFileReader(MMapFileReader&&) = delete;
    MMapFileReader& operator=(const MMapFileReader&) = delete;
    MMapFileReader& operator=(MMapFileReader&&) = delete;

    /**
     * Maps the file specified into virtual memory space.
     * @throws std::runtime_error if the file cannot be loaded or mapping fails.
     */
    MMapFileReader(const std::string& filename);

    /** Unmaps the loaded file from virtual memory space. */
    ~MMapFileReader();

    uint64_t GetLength() const;

    /**
     * Provides a read-only Span of bytes at the requested offset and length.
     * @throws std::range_error if offset or length are invalid.
     */
    Util::Span<const uint8_t> ReadData(uint64_t offset, uint64_t length) const;

    /**
     * Interprets mapped data as a C++ struct at the provided offset.
     */
    template <typename T>
    const T& ReadStruct(uint64_t offset) const
    {
        auto region = ReadData(offset, sizeof(T));
        return *reinterpret_cast<const T*>(region.Data());
    }

    /** Returns true if the underlying file is encrypted. */
    bool IsEncrypted() const;

protected:
    std::unique_ptr<void, std::function<void(void*)>> m_mmap;

    Util::Span<uint8_t> m_dataSpan;

    bool m_isEncrypted;
};

}  // namespace Blob
}  // namespace MILBlob
