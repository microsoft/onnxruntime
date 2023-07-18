// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/elementwise_impl/impl.cuh"

ELEMENTWISE_KERNEL_IMPL(functor::GeLU, float);
ELEMENTWISE_KERNEL_IMPL(functor::GeLU, half);
ELEMENTWISE_KERNEL_IMPL(functor::GeLU, BFloat16);
