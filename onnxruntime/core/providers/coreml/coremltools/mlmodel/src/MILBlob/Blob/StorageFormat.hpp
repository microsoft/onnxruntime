// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Blob/BlobDataType.hpp"

namespace MILBlob {
namespace Blob {

//
// ---: Blob Storage File Format :---
// Default file format for CoreML (iOS15 onwards)
//
// ---: File sturcture :---
// File is structued as below:
// 1. Storage header: `struct storage_header`
// 2. Followed by pair: `struct blob_metadata` and `raw_data`
// Each entry i.e. blob_metadata and raw data is 64 bytes aligned.
//
// Example file structure:
// |<storage_header>|<blob_metadata 0>|<data 0>|...|<optional padding for
// alignment>|<blob_metadata k>|<data k>|
//
// Example (file structure and associated mil_program usage):
// |storage_header>|<blob_metadata_0>,<data_0>|...|<blob_metadata_k>,<data_k>| //  file structure
// |               |64               ,128     |   |256              ,320     | //  byte offset
//
// Example usage in MIL program:
// a = const(BlobFile(file_path="weights/file.wt", offset=64))
// b = const(BlobFile(file_path="weights/file.wt", offset=256))
//
// Reference: https://quip-apple.com/V5zFA91jmjL3
//

// Default alignment being used for reading-writing Blob Storage format.
constexpr uint64_t DefaultStorageAlignment = 64;
// Default sentinel for validation for metadata
constexpr uint64_t BlobMetadataSentinel = 0xDEADBEEF;

/**
 * blob_metadata: stores information of blob present in weight file
 */
struct blob_metadata {
    uint32_t sentinel = BlobMetadataSentinel;  // for validating correctness of metadata.

    BlobDataType mil_dtype;       // data type of the blob data.
    uint64_t sizeInBytes;         // size of the blob data in bytes.
    uint64_t offset;              // offset in file for blob data.

    // Reserve fields
    uint64_t reserved_0;
    uint64_t reserved_1;
    uint64_t reserved_2;
    uint64_t reserved_3;
    uint64_t reserved_4;
};

/**
 * storage_header: Header for MIL Blob Storage format
 *  - stores count of number of blobs present in current weight file
 *  - stores version (this format currently only supports version=2)
 *        version=1 in file header is Espresso `blob_v1` format
 */
struct storage_header {
    uint32_t count = 0;    // Number of blob data.
    uint32_t version = 2;  // default=2

    uint64_t reserved_0 = 0;
    uint64_t reserved_1 = 0;
    uint64_t reserved_2 = 0;
    uint64_t reserved_3 = 0;
    uint64_t reserved_4 = 0;
    uint64_t reserved_5 = 0;
    uint64_t reserved_6 = 0;
};

// storage_header and blob_metadata are 64 bytes aligned.
// This allows first metadata to be aligned by default
// and data following blob_metadata aligned by defaul as well.
static_assert(sizeof(blob_metadata) == sizeof(uint64_t) * 8, "blob_metadata must be of size 64 bytes");
static_assert(sizeof(storage_header) == sizeof(uint64_t) * 8, "storage_header must be of size 64 bytes");

}  // namespace Blob
}  // namespace MILBlob
