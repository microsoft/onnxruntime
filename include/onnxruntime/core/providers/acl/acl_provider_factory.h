// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \deprecated The ACL execution provider has been removed. This function is retained for compatibility.
 * \param options Session options, which are left unchanged.
 * \param enable_fast_math Ignored.
 * \return An ORT_FAIL status explaining that the ACL execution provider has been removed.
 */
ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_ACL, _In_ OrtSessionOptions* options,
                          bool enable_fast_math)
    ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
