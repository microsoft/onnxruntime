// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type { InferenceSession } from 'onnxruntime-common';

export const serializeStorageBufferBucketLimits = (options: InferenceSession.WebGpuExecutionProviderOption): string => {
  const limits = options.storageBufferBucketLimits;
  if (!limits || typeof limits !== 'object' || Array.isArray(limits) || Object.keys(limits).length === 0) {
    throw new Error('storageBufferBucketLimits must be a non-empty object');
  }
  if (options.storageBufferCacheMode && options.storageBufferCacheMode !== 'bucket') {
    throw new Error("storageBufferBucketLimits requires storageBufferCacheMode 'bucket'");
  }
  return Object.entries(limits)
    .map(([key, limit]) => {
      const size = Number(key);
      if (!/^[1-9][0-9]*$/.test(key) || !Number.isSafeInteger(size) || size % 16 !== 0) {
        throw new Error('storageBufferBucketLimits sizes must be positive safe integers divisible by 16');
      }
      if (!Number.isSafeInteger(limit) || limit < 0) {
        throw new Error('storageBufferBucketLimits limits must be non-negative safe integers');
      }
      return `${key}:${limit}`;
    })
    .join(',');
};
