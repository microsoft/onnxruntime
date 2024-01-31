// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#define MILVerifyImpl(condition, ex_type, ...) \
    do {                                       \
        if (!(condition)) {                    \
            throw ex_type(__VA_ARGS__);        \
        }                                      \
    } while (0)

#if defined(NDEBUG)
#define MILVerifyDebugImpl(condition, ex_type, ...)
#else
#define MILVerifyDebugImpl(condition, ex_type, ...) MILVerifyImpl(condition, ex_type, __VA_ARGS__)
#endif

// MILVerifyIsNotNull verifies a pointer is not null. Upon failure, it throws the exception
// with the provided arguments.
#define MILVerifyIsNotNull(pointer, ex_type, ...) MILVerifyImpl(pointer != nullptr, ex_type, __VA_ARGS__)

// MILVerifyIsTrue verifies condition is true. Upon failure, it throws the exception
// with the provided arguments.
#define MILVerifyIsTrue(condition, ex_type, ...) MILVerifyImpl(condition, ex_type, __VA_ARGS__)

// MILDebugVerifyIsTrue verifies condition is true in debug builds only. Upon failure,
// it throws the exception with the provided arguments.
#define MILDebugVerifyIsTrue(condition, ex_type, ...) MILVerifyDebugImpl(condition, ex_type, __VA_ARGS__)
