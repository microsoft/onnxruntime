// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WebGpuBackend} from '../backend-webgpu';
import {LOG_DEBUG} from '../log';

import {GpuData, GpuDataId, GpuDataType} from './types';

/**
 * manages GpuDataId -> GpuBuffer
 */
export interface GpuDataManager {
  /**
   * copy data from CPU to GPU.
   */
  upload(id: GpuDataId, data: Uint8Array): void;
  /**
   * copy data from GPU to GPU.
   */
  memcpy(sourceId: GpuDataId, destinationId: GpuDataId): void;
  /**
   * create new data on GPU.
   */
  create(size: number, usage?: number): GpuData;
  /**
   * get GPU data by ID.
   */
  get(id: GpuDataId): GpuData|undefined;
  /**
   * release the data on GPU by ID.
   *
   * @return size of the data released
   */
  release(id: GpuDataId): number;
  /**
   * copy data from GPU to CPU synchronously.
   */
  downloadSync(id: GpuDataId): ArrayBufferLike;
  /**
   * copy data from GPU to CPU.
   */
  download(id: GpuDataId): Promise<ArrayBufferLike>;

  /**
   * refresh the buffers that marked for release.
   *
   * when release() is called, the buffer is not released immediately. this is because we need to wait for the commands
   * to be submitted to the GPU. this function is called after the commands are submitted so that the buffers can be
   * actually released.
   */
  refreshPendingBuffers(): void;

  /**
   * destroy all gpu buffers. Call this when the session.release is called.
   */
  dispose(): void;
}

interface StorageCacheValue {
  gpuData: GpuData;
  originalSize: number;
}

/**
 * normalize the buffer size so that it fits the 128-bits (16 bytes) alignment.
 */
const calcNormalizedBufferSize = (size: number) => Math.ceil(size / 16) * 16;

let guid = 0;
const createNewGpuDataId = () => guid++;

class GpuDataManagerImpl implements GpuDataManager {
  // GPU Data ID => GPU Data ( storage buffer )
  storageCache: Map<GpuDataId, StorageCacheValue>;

  // pending buffers for uploading ( data is unmapped )
  private buffersForUploadingPending: GPUBuffer[];
  // pending buffers for computing
  private buffersPending: GPUBuffer[];

  // The reusable storage buffers for computing.
  private freeBuffers: Map<number, GPUBuffer[]>;

  constructor(private backend: WebGpuBackend) {
    this.storageCache = new Map();
    this.freeBuffers = new Map();
    this.buffersForUploadingPending = [];
    this.buffersPending = [];
  }

  upload(id: GpuDataId, data: Uint8Array): void {
    const srcArrayBuffer = data.buffer;
    const srcOffset = data.byteOffset;
    const srcLength = data.byteLength;
    const size = calcNormalizedBufferSize(srcLength);

    // get destination gpu buffer
    const gpuDataCache = this.storageCache.get(id);
    if (!gpuDataCache) {
      throw new Error('gpu data for uploading does not exist');
    }
    if (gpuDataCache.originalSize !== srcLength) {
      throw new Error(`inconsistent data size. gpu data size=${gpuDataCache.originalSize}, data size=${srcLength}`);
    }

    // create gpu buffer
    const gpuBufferForUploading = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {mappedAtCreation: true, size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC});

    // copy (upload) data
    const arrayBuffer = gpuBufferForUploading.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    gpuBufferForUploading.unmap();


    // GPU copy
    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    commandEncoder.copyBufferToBuffer(gpuBufferForUploading, 0, gpuDataCache.gpuData.buffer, 0, size);

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.upload(id=${id})`);

    this.buffersForUploadingPending.push(gpuBufferForUploading);
  }

  memcpy(sourceId: GpuDataId, destinationId: GpuDataId): void {
    // get source gpu buffer
    const sourceGpuDataCache = this.storageCache.get(sourceId);
    if (!sourceGpuDataCache) {
      throw new Error('source gpu data for memcpy does not exist');
    }
    // get destination gpu buffer
    const destinationGpuDataCache = this.storageCache.get(destinationId);
    if (!destinationGpuDataCache) {
      throw new Error('destination gpu data for memcpy does not exist');
    }
    if (sourceGpuDataCache.originalSize !== destinationGpuDataCache.originalSize) {
      throw new Error('inconsistent source and destination gpu data size');
    }
    const size = calcNormalizedBufferSize(sourceGpuDataCache.originalSize);

    // GPU copy
    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    commandEncoder.copyBufferToBuffer(
        sourceGpuDataCache.gpuData.buffer, 0, destinationGpuDataCache.gpuData.buffer, 0, size);
  }

  // eslint-disable-next-line no-bitwise
  create(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GpuData {
    const bufferSize = calcNormalizedBufferSize(size);

    let gpuBuffer;
    // Currently, only storage buffers are reused.
    // eslint-disable-next-line no-bitwise
    if ((usage & GPUBufferUsage.STORAGE) === GPUBufferUsage.STORAGE) {
      let buffers = this.freeBuffers.get(bufferSize);
      if (!buffers) {
        buffers = [];
        this.freeBuffers.set(bufferSize, buffers);
      }
      if (buffers.length > 0) {
        gpuBuffer = buffers.pop() as GPUBuffer;
      } else {
        // create gpu buffer
        gpuBuffer = this.backend.device.createBuffer({size: bufferSize, usage});
      }
    } else {
      // create gpu buffer
      gpuBuffer = this.backend.device.createBuffer({size: bufferSize, usage});
    }

    const gpuData = {id: createNewGpuDataId(), type: GpuDataType.default, buffer: gpuBuffer};
    this.storageCache.set(gpuData.id, {gpuData, originalSize: size});

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.create(size=${size}) => id=${gpuData.id}`);
    return gpuData;
  }

  get(id: GpuDataId): GpuData|undefined {
    return this.storageCache.get(id)?.gpuData;
  }

  release(id: GpuDataId): number {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('releasing data does not exist');
    }

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.release(id=${id}), gpuDataId=${cachedData.gpuData.id}`);

    this.storageCache.delete(id);
    this.buffersPending.push(cachedData.gpuData.buffer);
    // cachedData.gpuData.buffer.destroy();

    return cachedData.originalSize;
  }

  downloadSync(id: GpuDataId): ArrayBufferLike {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }
    const bufferSize = calcNormalizedBufferSize(cachedData.originalSize);

    const pixelsSize = bufferSize / 4;
    const valsGPU = new ArrayBuffer(bufferSize);
    const canvasWidth = 256, canvasHeight = 256;
    const stagingDeviceStorage = new OffscreenCanvas(canvasWidth, canvasHeight);
    const stagingHostStorage = new OffscreenCanvas(canvasWidth, canvasHeight);

    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();

    const context = stagingDeviceStorage.getContext('webgpu');
    if (context) {
      context.configure({
        device: this.backend.device,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.COPY_DST,
        alphaMode: 'premultiplied',
      });
      const texture = context.getCurrentTexture();

      const bytesPerRow = canvasWidth * 4;
      const readDataGPUToCPU = (width: number, height: number, offset: number) => {
        commandEncoder.copyBufferToTexture(
            {
              buffer: cachedData.gpuData.buffer,
              bytesPerRow,
              offset,
            },
            {
              texture,
            },
            {
              width,
              height,
            });
        this.backend.flush();

        const gl = stagingHostStorage.getContext('webgl2');
        if (gl) {
          const glTex = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, glTex);
          gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, stagingDeviceStorage);
          const fb = gl.createFramebuffer();
          gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, glTex, 0);

          const id = new Uint8ClampedArray(valsGPU, offset, width * height * 4);
          gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, id);
        } else {
          throw new Error(
              'Cannot copy data from GPU to CPU synchronously, ' +
              'OffscreenCanvas does not support webgl2 drawing context.');
        }
      };

      const fullyReadCount = Math.floor(pixelsSize / (canvasWidth * canvasHeight));
      let width = canvasWidth, height = canvasHeight, offset = 0;
      for (let i = 0; i < fullyReadCount; i++) {
        // Read the buffer data, which fully fill the whole canvas.
        readDataGPUToCPU(width, height, offset);
        offset += canvasWidth * canvasHeight * 4;
      }

      const remainSize = pixelsSize % (canvasWidth * canvasHeight);
      height = Math.floor(remainSize / canvasWidth);
      if (height > 0) {
        // Read the buffer data, which fully fill certain rows of canvas.
        readDataGPUToCPU(width, height, offset);
        offset += height * (canvasWidth * 4);
      }

      width = remainSize % canvasWidth;
      if (width > 0) {
        // Read the buffer data, which not fully fill one row of canvas.
        readDataGPUToCPU(width, 1, offset);
      }
    } else {
      throw new Error(
          'Cannot copy data from GPU to CPU synchronously, OffscreenCanvas does not support webgpu drawing context.');
    }

    return valsGPU;
  }

  async download(id: GpuDataId): Promise<ArrayBufferLike> {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }

    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    const bufferSize = calcNormalizedBufferSize(cachedData.originalSize);
    const gpuReadBuffer = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {size: bufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    commandEncoder.copyBufferToBuffer(
        cachedData.gpuData.buffer /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
        0 /* destination offset */, bufferSize /* size */
    );
    this.backend.flush();

    return new Promise<ArrayBuffer>((resolve) => {
      gpuReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const data = gpuReadBuffer.getMappedRange().slice(0);
        gpuReadBuffer.destroy();
        resolve(data);
      });
    });
  }

  refreshPendingBuffers(): void {
    for (const buffer of this.buffersForUploadingPending) {
      // upload buffer is only useful in the session creation time. So we don't need to reuse them in session running.
      buffer.destroy();
    }
    this.buffersForUploadingPending = [];
    for (const buffer of this.buffersPending) {
      // eslint-disable-next-line no-bitwise
      if ((buffer.usage & GPUBufferUsage.STORAGE) === GPUBufferUsage.STORAGE) {
        // Put the pending buffer to freeBuffers list instead of really destroying it for buffer reusing.
        this.freeBuffers.get(buffer.size)!.push(buffer);
      } else {
        buffer.destroy();
      }
    }
    this.buffersPending = [];
  }

  dispose() {
    this.freeBuffers.forEach((buffers) => {
      buffers.forEach(buffer => {
        buffer.destroy();
      });
    });

    this.storageCache.forEach((storage) => {
      storage.gpuData.buffer.destroy();
    });

    this.storageCache = new Map();
    this.freeBuffers = new Map();
  }
}

export const createGpuDataManager = (...args: ConstructorParameters<typeof GpuDataManagerImpl>): GpuDataManager =>
    new GpuDataManagerImpl(...args);
