// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is adapted from microsoft/DeepSpeed
// type_shim.h

/* Taken from NVIDIA/apex commit 855808f3fc268e9715d613f3c2e56469d8c986d8 */
#include <ATen/ATen.h>

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)                   \
    switch (TYPE) {                                                              \
        case at::ScalarType::Double: {                                           \
            using scalar_t_##LEVEL = double;                                     \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Float: {                                            \
            using scalar_t_##LEVEL = float;                                      \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        case at::ScalarType::Half: {                                             \
            using scalar_t_##LEVEL = at::Half;                                   \
            __VA_ARGS__;                                                         \
            break;                                                               \
        }                                                                        \
        default: AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }
