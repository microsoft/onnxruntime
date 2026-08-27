// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { env } from 'onnxruntime-common';

// Reading the device is synchronous and may return undefined before WebGPU initialization.
//
// {type-tests}|pass
const device: unknown | undefined = env.webgpu.device;
void device;

// {type-tests}|fail|1|2322
const devicePromise: Promise<unknown> = env.webgpu.device;
void devicePromise;

// Applications can request additional features for an ORT-created device.
//
// {type-tests}|pass
env.webgpu.requiredFeatures = ['bgra8unorm-storage'];
