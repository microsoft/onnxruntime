// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/elementwise_impl/impl.cuh"

ELEMENTWISE_KERNEL_IMPL(functor::FastGeLU, float);
ELEMENTWISE_KERNEL_IMPL(functor::FastGeLU, half);
ELEMENTWISE_KERNEL_IMPL(functor::FastGeLU, BFloat16);
