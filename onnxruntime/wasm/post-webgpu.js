// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This file contains the post-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--post-js` option.
//
// This file will only be used in build with flag `--use_webgpu`.

/**
 * This function is called only once when initializing the WebGPU backend.
 *
 * @param {(gpuDevice: GPUDevice) => void} setDefaultDevice A callback function to set the default device.
 */
Module["webgpuInit"] = (setDefaultDevice) => {
  /**
   * a map from GPUDevice to [deviceId, instanceHandle, deviceHandle]
   *
   * only stores custom devices (ie. devices created by the user, not the default device created by ORT)
   *
   * key is the GPUDevice object.
   *
   * value is a tuple of 3 elements:
   * - deviceId: a unique ID for the device. Must be positive integer.
   * - instanceHandle: the instance handle(pointer) of the device.
   * - deviceHandle: the device handle(pointer) of the device.
   *
   * @type {WeakMap<GPUDevice, [number, number, number]>}
   */
  const webgpuActiveDevices = new WeakMap();
  /**
   * a number that is used to assign a unique ID to the next custom device.
   */
  let webgpuNextDeviceId = 1;
  /**
   * a function to set the default device.
   *
   * @type {(gpuDevice: GPUDevice) => void}
   */
  const webgpuSetDefaultDevice = setDefaultDevice;
  /**
   * the current device that is being used to create a WebGPU EP inference session.
   *
   * the value of this variable is only valid during the creation of a WebGPU EP inference session.
   *
   * @type {GPUDevice|undefined}
   */
  let webgpuCurrentDevice = undefined;
  /**
   * the current device ID that is being used to create a WebGPU EP inference session.
   *
   * the value of this variable is only valid during the creation of a WebGPU EP inference session.
   *
   * @type {number|undefined}
   */
  let webgpuCurrentDeviceId = undefined;

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
      webgpuCurrentDevice = device;
      webgpuCurrentDeviceId = deviceInfo[0];
      return deviceInfo;
    } else {
      webgpuCurrentDevice = undefined;
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

    if (sessionHandle) {
      // when session created successfully
      const deviceHandle = _OrtGetWebGpuDevice(deviceId);
      webgpuActiveSessions.set(sessionHandle, deviceHandle);

      if (deviceId === 0) {
        const device = webgpuCurrentDevice ?? WebGPU.getJsObject(deviceHandle);
        webgpuSetDefaultDevice(device);
      }
    }
    webgpuCurrentDevice = undefined;
  };

  Module["webgpuOnReleaseSession"] = (sessionHandle) => {
    webgpuActiveSessions.delete(sessionHandle);
  };

  const gpuBufferMetadataSymbol = Symbol("gpuBufferMetadata");

  Module["webgpuRegisterBuffer"] = (buffer, sessionHandle, bufferHandle) => {
    if (bufferHandle) {
      // This is a buffer that was created by ORT. Metadata is [bufferHandle, NaN]

      buffer[gpuBufferMetadataSymbol] = [bufferHandle, NaN];
      return bufferHandle;
    } else {
      // This is a buffer that was created by the user. Metadata is [bufferHandle, refCount]

      const metadata = buffer[gpuBufferMetadataSymbol];
      if (metadata) {
        metadata[1]++;
        return metadata[0];
      }

      const deviceHandle = webgpuActiveSessions.get(sessionHandle);
      if (deviceHandle === undefined) {
        throw new Error(
          "Invalid session handle passed to webgpuRegisterBuffer"
        );
      }

      const bufferHandle = WebGPU.importJsBuffer(buffer, deviceHandle);
      buffer[gpuBufferMetadataSymbol] = [bufferHandle, 1];
      return bufferHandle;
    }
  };

  Module["webgpuUnregisterBuffer"] = (buffer) => {
    const metadata = buffer[gpuBufferMetadataSymbol];
    if (!metadata) {
      throw new Error("Buffer is not registered");
    }
    metadata[1]--;
    // For buffers created by ORT, metadata[1] will always be NaN. This function will not release the buffer.
    // Instead, the buffer will be released when user calls `Tensor.dispose()` in JavaScript.
    if (metadata[1] === 0) {
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
      // prettier-ignore
      //
      // the line above is used to force prettier to skip formatting the next statement.
      // this is because prettier will remove the quotes around the property names, but we need to keep them
      // because otherwise closure compiler may rename them and break the code.
      const gpuReadBufferDescriptor = {
        "size": size,
        "usage": 9 /* GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ */,
      };
      const gpuReadBuffer = device.createBuffer(gpuReadBufferDescriptor);
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

  // Setup a callback function for loading external buffers (model weights).
  Module.webgpuUploadExternalBuffer = (bufferHandle, data) => {
    const srcArrayBuffer = data.buffer;
    const srcOffset = data.byteOffset;
    const srcLength = data.byteLength;
    const size = Math.ceil(Number(srcLength) / 16) * 16;

    const gpuBuffer = WebGPU.getJsObject(bufferHandle);

    // get current device
    if (!webgpuCurrentDevice) {
      const deviceHandle = _OrtGetWebGpuDevice(webgpuCurrentDeviceId);
      webgpuCurrentDevice = WebGPU.getJsObject(deviceHandle);
    }

    // create gpu buffer

    // prettier-ignore
    //
    // the line above is used to force prettier to skip formatting the next statement.
    // this is because prettier will remove the quotes around the property names, but we need to keep them
    // because otherwise closure compiler may rename them and break the code.
    const gpuBufferForUploadingDescriptor = {
      "mappedAtCreation": true,
      "size": size,
      "usage": 6 /* GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC */,
    };
    const gpuBufferForUploading = webgpuCurrentDevice.createBuffer(
      gpuBufferForUploadingDescriptor
    );

    // copy (upload) data
    const arrayBuffer = gpuBufferForUploading.getMappedRange();
    new Uint8Array(arrayBuffer).set(
      new Uint8Array(srcArrayBuffer, srcOffset, srcLength)
    );
    gpuBufferForUploading.unmap();

    // GPU copy
    const commandEncoder = webgpuCurrentDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      gpuBufferForUploading,
      0,
      gpuBuffer,
      0,
      size
    );
    webgpuCurrentDevice.queue.submit([commandEncoder.finish()]);
    gpuBufferForUploading.destroy();
  };
};
