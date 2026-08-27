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
 * @param {GPUDevice|undefined} configuredDevice The application-created environment device, if set.
 * @param {readonly string[]} requiredFeatures Features required by the environment.
 * @param {(gpuDevice: GPUDevice) => void} setDefaultDevice A callback function to publish the effective device.
 */
Module["webgpuInit"] = (
  configuredDevice,
  requiredFeatures,
  setDefaultDevice,
) => {
  /**
   * a function to set the default device.
   *
   * @type {(gpuDevice: GPUDevice) => void}
   */
  const webgpuSetDefaultDevice = setDefaultDevice;
  /** @type {GPUDevice|undefined} */
  let webgpuDefaultDevice = configuredDevice;
  /**
   * the current device that is being used to create a WebGPU EP inference session.
   *
   * the value of this variable is only valid during the creation of a WebGPU EP inference session.
   *
   * @type {GPUDevice|undefined}
   */
  let webgpuCurrentDevice = undefined;
  /** Whether a WebGPU EP inference session is currently being created. */
  let webgpuCreatingSession = false;

  const configureDefaultContext = (device) => {
    let instanceHandle = 0;
    let deviceHandle = 0;
    if (device) {
      instanceHandle = _OrtCreateWebGpuInstance();
      if (!instanceHandle) {
        throw new Error(
          "Failed to create the WebGPU instance for the application-created device.",
        );
      }
      deviceHandle = WebGPU.importJsDevice(device, instanceHandle);
    }

    const features = requiredFeatures.join(",");
    const featuresLength = lengthBytesUTF8(features) + 1;
    const featuresOffset = _malloc(featuresLength);
    stringToUTF8(features, featuresOffset, featuresLength);
    const errorCode = _OrtConfigureWebGpuDefaultContext(
      instanceHandle,
      deviceHandle,
      featuresOffset,
    );
    _free(featuresOffset);
    return errorCode;
  };

  let errorCode = configureDefaultContext(configuredDevice);
  if (errorCode !== 0) {
    return errorCode;
  }
  if (configuredDevice) {
    webgpuSetDefaultDevice(configuredDevice);
  }

  /**
   * This function is called only when a custom device is used, during preparation of session options.
   *
   * @param {GPUDevice} device the user provided device object.
   * @returns {number} ORT error code.
   */
  Module["webgpuRegisterDevice"] = (device) => {
    if (webgpuCreatingSession) {
      throw new Error("another WebGPU EP inference session is being created.");
    }

    if (device) {
      if (webgpuDefaultDevice && webgpuDefaultDevice !== device) {
        throw new Error(
          "A different WebGPU device is already used by this ONNX Runtime environment.",
        );
      }
      if (!webgpuDefaultDevice) {
        errorCode = configureDefaultContext(device);
        if (errorCode !== 0) {
          return errorCode;
        }
        webgpuDefaultDevice = device;
        webgpuSetDefaultDevice(device);
      }
      webgpuCurrentDevice = device;
    } else {
      webgpuCurrentDevice = webgpuDefaultDevice;
    }
    webgpuCreatingSession = true;
    return 0;
  };

  const webgpuActiveSessions = new Map();
  Module["webgpuOnCreateSession"] = (sessionHandle) => {
    if (!webgpuCreatingSession) {
      return;
    }

    webgpuCreatingSession = false;

    if (sessionHandle) {
      // when session created successfully
      const deviceHandle = _OrtGetWebGpuDevice(0);
      webgpuActiveSessions.set(sessionHandle, deviceHandle);

      if (!webgpuDefaultDevice) {
        webgpuDefaultDevice = WebGPU.getJsObject(deviceHandle);
        webgpuSetDefaultDevice(webgpuDefaultDevice);
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
          "Invalid session handle passed to webgpuRegisterBuffer",
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
          size /* size */,
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
      const deviceHandle = _OrtGetWebGpuDevice(0);
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
      gpuBufferForUploadingDescriptor,
    );

    // copy (upload) data
    const arrayBuffer = gpuBufferForUploading.getMappedRange();
    new Uint8Array(arrayBuffer).set(
      new Uint8Array(srcArrayBuffer, srcOffset, srcLength),
    );
    gpuBufferForUploading.unmap();

    // GPU copy
    const commandEncoder = webgpuCurrentDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      gpuBufferForUploading,
      0,
      gpuBuffer,
      0,
      size,
    );
    webgpuCurrentDevice.queue.submit([commandEncoder.finish()]);
    gpuBufferForUploading.destroy();
  };

  return 0;
};
