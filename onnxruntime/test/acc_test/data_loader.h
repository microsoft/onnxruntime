// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
#include <vector>
#include <filesystem>

#include "basic_utils.h"
#include "model_io_utils.h"

namespace acctest {

bool LoadIODataFromDisk(const std::vector<std::filesystem::path>& dataset_paths,
                        const std::vector<IOInfo>& io_infos,
                        const char* data_file_prefix,
                        std::vector<std::unique_ptr<char[]>>& dataset_data);
}  // namespace acctest
