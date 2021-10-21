// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace utils {

/**
 * Gets the random seed value used by onnxruntime.
 *
 * The random seed value can be override with SetRandomSeed().
 *
 * @return The random seed value.
 */
int64_t GetRandomSeed();

/**
 * Sets the random seed value to be used by onnxruntime.
 *
 * If not called manually, the current clock will be used.
 *
 * @param seed The random seed value to use.
 */
void SetRandomSeed(int64_t seed);

}  // namespace test
}  // namespace onnxruntime
