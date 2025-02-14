// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import type { Env } from 'onnxruntime-common';
// import { getInstance } from './wasm-factory';

// let envConfigured = false;
// let currentAdapter: GPUAdapter | undefined;
// let currentDevice: GPUDevice | undefined;

// const createNewAdapter = async (env: Env): Promise<GPUAdapter> => {
//   const powerPreference = env.webgpu.powerPreference;
//   if (powerPreference !== undefined && powerPreference !== 'low-power' && powerPreference !== 'high-performance') {
//     throw new Error(`Invalid powerPreference setting: "${powerPreference}"`);
//   }
//   const forceFallbackAdapter = env.webgpu.forceFallbackAdapter;
//   if (forceFallbackAdapter !== undefined && typeof forceFallbackAdapter !== 'boolean') {
//     throw new Error(`Invalid forceFallbackAdapter setting: "${forceFallbackAdapter}"`);
//   }
//   const adapter = await navigator.gpu.requestAdapter({ powerPreference, forceFallbackAdapter });
//   if (!adapter) {
//     throw new Error('Failed to get GPU adapter. ' + 'You may need a browser that supports WebGPU to run this code.');
//   }
//   return adapter;
// };

// const createNewDevice = async (adapter: GPUAdapter): Promise<GPUDevice> => {
//   const requiredFeatures: GPUFeatureName[] = [];
//   const deviceDescriptor: GPUDeviceDescriptor = {
//     requiredLimits: {
//       maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
//       maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
//       maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
//       maxBufferSize: adapter.limits.maxBufferSize,
//       maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
//       maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
//       maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
//       maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
//     },
//     requiredFeatures,
//   };

//   // Try requiring WebGPU features
//   const requireFeatureIfAvailable = (feature: GPUFeatureName) =>
//     adapter.features.has(feature) && requiredFeatures.push(feature) && true;
//   // Try chromium-experimental-timestamp-query-inside-passes and fallback to timestamp-query
//   if (!requireFeatureIfAvailable('chromium-experimental-timestamp-query-inside-passes' as GPUFeatureName)) {
//     requireFeatureIfAvailable('timestamp-query');
//   }
//   requireFeatureIfAvailable('shader-f16');
//   // Try subgroups
//   if (requireFeatureIfAvailable('subgroups' as GPUFeatureName)) {
//     // If subgroups feature is available, also try subgroups-f16
//     requireFeatureIfAvailable('subgroups-f16' as GPUFeatureName);
//   }

//   return adapter.requestDevice(deviceDescriptor);
// };

// export const webgpuEpInit = (adapter: GPUAdapter, device: GPUDevice) => {
//   currentAdapter = adapter;
//   currentDevice = device;

//   getInstance().webgpuInit!(adapter, device);
// };

// export const ensureAdapterAndDevice = async (env: Env): Promise<[GPUDevice, GPUAdapter]> => {
//   // perform WebGPU availability check
//   if (typeof navigator === 'undefined' || !navigator.gpu) {
//     throw new Error('WebGPU is not supported in current environment');
//   }

//   if (BUILD_DEFS.USE_WEBGPU_EP && !envConfigured) {
//     Object.defineProperty(env.webgpu, 'adapter', {
//       enumerable: true,
//       configurable: false,
//       get: () => currentAdapter,
//     });
//     Object.defineProperty(env.webgpu, 'device', {
//       enumerable: true,
//       configurable: false,
//       get: () => currentDevice,
//       set: (_device: GPUDevice) => {
//         throw new Error('Currently setting GPU device is not supported yet.');
//       },
//     });
//     envConfigured = true;
//   }

//   // check if user has provided a GPUAdapter object
//   let adapter = env.webgpu?.adapter;
//   if (!BUILD_DEFS.USE_WEBGPU_EP && adapter) {
//     // if adapter is set, validate it.
//     // this validation is only for JSEP, for backward compatibility.
//     //
//     // In latest API, adapter is not setable by user.
//     if (
//       typeof adapter.limits !== 'object' ||
//       typeof adapter.features !== 'object' ||
//       typeof adapter.requestDevice !== 'function'
//     ) {
//       throw new Error('Invalid GPU adapter set in `env.webgpu.adapter`. It must be a GPUAdapter object.');
//     }
//   } else {
//     adapter = await createNewAdapter(env);
//   }

//   // Now Adapter is available. Request a device.

//   let device: GPUDevice | null = await env.webgpu?.device;
//   if (device) {
//     // if device is set, validate it.
//     if (
//       typeof device.limits !== 'object' ||
//       typeof device.features !== 'object' ||
//       typeof device.createBuffer !== 'function' ||
//       typeof device.createComputePipelineAsync !== 'function'
//     ) {
//       throw new Error('Invalid GPU device set in `env.webgpu.device`. It must be a GPUDevice object.');
//     }
//   } else {
//     device = await createNewDevice(adapter);
//   }

//   if (BUILD_DEFS.USE_WEBGPU_EP) {
//     void device.lost.then(async (info) => {
//       if (info.reason === 'destroyed') {
//         currentAdapter = undefined;
//         currentDevice = undefined;

//         const adapter = await createNewAdapter(env);
//         const device = await createNewDevice(adapter);

//         webgpuEpInit(adapter, device);
//       }
//     });
//   }

//   return [device, adapter];
// };

// export const configureWebGpu =  (env: Env): void => {
//   Object.defineProperty(env.webgpu, 'adapter', {
//     enumerable: true,
//     configurable: false,
//     get: () => currentAdapter,
//   });
//   Object.defineProperty(env.webgpu, 'device', {
//     enumerable: true,
//     configurable: false,
//     get: () => currentDevice,
//     set: (_device: GPUDevice) => {
//       throw new Error('Currently setting GPU device is not supported yet.');
//     },
//   });

// };
