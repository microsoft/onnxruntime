// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <string>

namespace MILBlob::Blob {

class MMapFileReader;

/**
 * MakeMMapFileReader: Returns MMapedFileReader for file present at given filePath
 */
std::unique_ptr<MMapFileReader> MakeMMapFileReader(const std::string& filePath);

}  // namespace MILBlob::Blob
