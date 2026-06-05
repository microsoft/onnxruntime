// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file parser.h
/// \brief Model package JSON parser (internal).

#pragma once

#include <filesystem>
#include <string>

#include "model_package_internal.h"

namespace model_package {

/// Parse a model package from a directory.
/// Reads manifest.json, metadata.json per component, variant.json per variant.
///
/// \param[in]  package_root  Path to the model package root directory.
/// \param[out] out_package   On success, filled with the parsed package info.
/// \param[out] out_error     On failure, filled with an error message.
/// \return true on success, false on error.
bool ParsePackage(const std::filesystem::path& package_root,
                  PackageInfo& out_package,
                  std::string& out_error);

}  // namespace model_package
