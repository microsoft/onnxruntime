// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/MMapFileReader.hpp"

#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace MILBlob;
using namespace MILBlob::Blob;

MMapFileReader::~MMapFileReader() = default;

MMapFileReader::MMapFileReader(const std::string& filename) : m_isEncrypted(false)
{
    // verify file exists and find its length
    struct stat fileInfo;
    if (stat(filename.c_str(), &fileInfo) != 0) {
        throw std::runtime_error("Could not open " + filename);
    }

    // mmap works in size_t units to be compatible with virtual address space units
    auto fileLength = static_cast<size_t>(fileInfo.st_size);

    // wrap fopen/fclose in exception-safe type
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(filename.c_str(), "r"), fclose);

    MILVerifyIsTrue(f != nullptr, std::runtime_error, "Unable to read file " + filename);

    // wrap mmap/munmap in exception-safe type
    std::unique_ptr<void, std::function<void(void*)>> mmapPtr(
        mmap(nullptr, fileLength, PROT_READ, MAP_PRIVATE, fileno(f.get()), 0 /*offset*/),
        [length = fileLength](void* ptr) { munmap(ptr, length); });

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast) -- MAP_FAILED is (void*) -1.
    MILVerifyIsTrue(mmapPtr.get() != nullptr && mmapPtr.get() != MAP_FAILED,
                    std::runtime_error,
                    "Unable to mmap file " + filename);

    m_dataSpan = Util::Span<uint8_t>(reinterpret_cast<uint8_t*>(mmapPtr.get()), fileLength);

    // Keep mmaping alive
    m_mmap = std::move(mmapPtr);
}

uint64_t MMapFileReader::GetLength() const
{
    return m_dataSpan.Size();
}

Util::Span<const uint8_t> MMapFileReader::ReadData(uint64_t offset, uint64_t length) const
{
    return m_dataSpan.Slice(static_cast<size_t>(offset), static_cast<size_t>(length));
}

bool MMapFileReader::IsEncrypted() const
{
    return m_isEncrypted;
}
