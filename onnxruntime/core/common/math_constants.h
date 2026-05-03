// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Portable math constant definitions.
// POSIX constants like M_PI and M_SQRT2 are not part of the C++ standard and
// are not provided by all standard library implementations (e.g. libc++).
// This header provides them when missing, enabling compilation with any
// conforming C++ standard library.

#pragma once

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
