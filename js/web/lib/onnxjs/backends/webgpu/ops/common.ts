// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * constant value for a workgroup size.
 *
 * We definitely can do further optimization in future, but for now we use 64.
 *
 * rule of thumb: Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload
 *                needs something different.
 *
 * from: https://surma.dev/things/webgpu/
 **/
export const WORKGROUP_SIZE = 64;
