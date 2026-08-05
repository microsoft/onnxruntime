// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/elementwise_impl/impl.cuh"

ELEMENTWISE_KERNEL_IMPL(functor::ReLU, float);
ELEMENTWISE_KERNEL_IMPL(functor::ReLU, half);
ELEMENTWISE_KERNEL_IMPL(functor::ReLU, BFloat16);
