// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace utils {

/**
 * Gets the static random seed value. All calls to GetStaticRandomSeed()
 * throughout the lifetime of the process will return the same value.
 *
 * @param default_seed If the random seed is not set, return this value instead of random if it's positive.
 * @return The static random seed value.
 */
uint32_t GetStaticRandomSeed(uint32_t default_seed = 0);

/**
 * Sets the static random seed value.
 *
 * If called, this should be called before calling GetStaticRandomSeed().
 * Not calling this is also fine. In that case, the value will be generated.
 *
 * @param seed The random seed value to use.
 */
void SetStaticRandomSeed(uint32_t seed);

}  // namespace test
}  // namespace onnxruntime
