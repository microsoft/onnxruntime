// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file manifest_parser.h
/// \brief Internal parser that reads a model package from disk into the
///        in-memory representation defined in model_package_impl.h.

#pragma once

#include "model_package_impl.h"

namespace model_package_v2 {

/// Parse the manifest at `<package_root>/manifest.json` and all referenced
/// external component files, then populate `*pkg`. Caller owns `pkg`.
ModelPackageStatus* ParsePackage(const std::filesystem::path& package_root,
                                 const ModelPackageOpenOptions& opts,
                                 ModelPackage* pkg);

}  // namespace model_package_v2
