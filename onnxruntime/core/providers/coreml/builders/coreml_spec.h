// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// TODO come up with a more intuitive way of limiting this to Apple platform builds
// E.g., putting CoreML EP files that should be enabled iff `defined(__APPLE__)` in a separate directory.
#if !defined(__APPLE__)
#error "This file should only be included when building on Apple platforms."
#endif

#include "coreml/Model.pb.h"

namespace COREML_SPEC = CoreML::Specification;
