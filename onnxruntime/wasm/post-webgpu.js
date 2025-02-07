// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

//
// This file contains the post-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--post-js` option.
//
// This file will only be used in build with flag `--use_webgpu`.

const webgpuActiveDevices = new WeakMap();
let webgpuNextDeviceId = 1;
let webgpuSetDefaultDevice = undefined;
let webgpuCurrentDeviceId = undefined;

/**
 * This function is called only once when initializing the WebGPU backend.
 *
 * @param {(gpuDevice: GPUDevice) => void} setDefaultDevice A callback function to set the default device.
 */
Module["webgpuInit"] = (setDefaultDevice) => {
  webgpuSetDefaultDevice = setDefaultDevice;
};

/**
 * This function is called only when a custom device is used, during preparation of session options.
 *
 * @param {GPUDevice} device the user provided device object.
 * @returns {undefined|[number, number, number]} a tuple of device id, instance handle, and device handle.
 */
Module["webgpuRegisterDevice"] = (device) => {
  if (webgpuCurrentDeviceId !== undefined) {
    throw new Error("another WebGPU EP inference session is being created.");
  }

  if (device) {
    let deviceInfo = webgpuActiveDevices.get(device);
    if (!deviceInfo) {
      const instanceHandle = _wgpuCreateInstance(0);
      const deviceHandle = WebGPU.importJsDevice(device, instanceHandle);
      deviceInfo = [webgpuNextDeviceId++, instanceHandle, deviceHandle];
      webgpuActiveDevices.set(device, deviceInfo);
    }

    // The current device ID is a temporary storage for the device ID to be used in the session that is being created.
    //
    // Soon after `webgpuRegisterDevice` (this function) is called, `webgpuOnCreateSession` will be called so that the
    // value of `webgpuCurrentDeviceId` is used and reset then.
    webgpuCurrentDeviceId = deviceInfo[0];
    return deviceInfo;
  } else {
    webgpuCurrentDeviceId = 0;
    return undefined;
  }
};

const webgpuActiveSessions = new Map();
Module["webgpuOnCreateSession"] = (sessionHandle) => {
  if (webgpuCurrentDeviceId === undefined) {
    // do nothing if webgpuCurrentDeviceId is undefined.
    // this means no WebGPU EP is being created.
    return;
  }

  const deviceId = webgpuCurrentDeviceId;
  webgpuCurrentDeviceId = undefined;

  const deviceHandle = _OrtGetWebGpuDevice(deviceId);
  webgpuActiveSessions.set(sessionHandle, deviceHandle);

  if (deviceId === 0) {
    const device = WebGPU.getJsObject(deviceHandle);
    webgpuSetDefaultDevice(device);
  }
};

Module["webgpuOnReleaseSession"] = (sessionHandle) => {
  webgpuActiveSessions.delete(sessionHandle);
};

const gpuBufferMetadataSymbol = Symbol("gpuBufferMetadata");

Module["webgpuRegisterBuffer"] = (buffer, sessionHandle, bufferHandle) => {
  const metadata = buffer[gpuBufferMetadataSymbol];
  if (bufferHandle) {
    // This is a buffer that was created by ORT. Metadata is [bufferHandle, NaN]

    buffer[gpuBufferMetadataSymbol] = [bufferHandle, NaN];
    return bufferHandle;
  } else {
    // This is a buffer that was created by the user. Metadata is [bufferHandle, refCount]

    if (metadata) {
      metadata[1]++;
      return metadata[0];
    }

    const deviceHandle = webgpuActiveSessions.get(sessionHandle);
    if (deviceHandle === undefined) {
      throw new Error("Invalid session handle passed to webgpuRegisterBuffer");
    }

    const bufferHandle = WebGPU.importJsBuffer(buffer, deviceHandle);
    buffer[gpuBufferMetadataSymbol] = [1];
    return bufferHandle;
  }
};

Module["webgpuUnregisterBuffer"] = (buffer) => {
  const metadata = buffer[gpuBufferMetadataSymbol];
  if (!metadata) {
    throw new Error("Buffer is not registered");
  }
  metadata[1]--;
  if (metadata[1] === 0) {
    // For buffers created by ORT, metadata[1] will always be NaN. This function will not release the buffer.
    // Instead, the buffer will be released when user calls `Tensor.dispose()` in JavaScript.
    _wgpuBufferRelease(metadata[0]);
    delete buffer[gpuBufferMetadataSymbol];
  }
};

Module["webgpuGetBuffer"] = (bufferHandle) => {
  return WebGPU.getJsObject(bufferHandle);
};

Module["webgpuCreateDownloader"] = (gpuBuffer, bufferSize, sessionHandle) => {
  const deviceHandle = webgpuActiveSessions.get(sessionHandle);
  if (deviceHandle === undefined) {
    throw new Error("Invalid session handle passed to webgpuRegisterBuffer");
  }

  const buffer = gpuBuffer;
  const device = WebGPU.getJsObject(deviceHandle);
  const originalSize = bufferSize;
  const size = Math.ceil(Number(originalSize) / 16) * 16;

  return async () => {
    const gpuReadBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        buffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        size /* size */
      );
      device.queue.submit([commandEncoder.finish()]);

      await gpuReadBuffer.mapAsync(GPUMapMode.READ);

      const arrayBuffer = gpuReadBuffer.getMappedRange();
      return arrayBuffer.slice(0, originalSize);
    } finally {
      gpuReadBuffer.destroy();
    }
  };
};
