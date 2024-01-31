// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Blob/MMapFileReader.hpp"
#include "MILBlob/Blob/MMapFileReaderFactory.hpp"

namespace MILBlob::Blob {

std::unique_ptr<MMapFileReader> MakeMMapFileReader(const std::string& filePath)
{
    return std::make_unique<MMapFileReader>(filePath);
}

}  // namespace MILBlob::Blob
