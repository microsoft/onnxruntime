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
}

interface StorageCacheValue {
  gpuData: GpuData;
  originalSize: number;
}

interface DownloadCacheValue {
  data: Promise<ArrayBufferLike>;
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

  // GPU Data ID => GPU Data ( read buffer )
  downloadCache: Map<GpuDataId, DownloadCacheValue>;

  // pending buffers for uploading ( data is unmapped )
  private buffersForUploadingPending: GPUBuffer[];
  // pending buffers for computing
  private buffersPending: GPUBuffer[];

  constructor(private backend: WebGpuBackend /* , private reuseBuffer: boolean */) {
    this.storageCache = new Map();
    this.downloadCache = new Map();
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
    this.backend.getCommandEncoder().copyBufferToBuffer(
        sourceGpuDataCache.gpuData.buffer, 0, destinationGpuDataCache.gpuData.buffer, 0, size);
  }

  // eslint-disable-next-line no-bitwise
  create(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GpuData {
    // !!!
    // !!! IMPORTANT: TODO: whether we should keep the storage buffer every time, or always create new ones.
    // !!!                  This need to be figured out by performance test results.
    // !!!

    const bufferSize = calcNormalizedBufferSize(size);

    // create gpu buffer
    const gpuBuffer = this.backend.device.createBuffer({size: bufferSize, usage});

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

    const downloadingData = this.downloadCache.get(id);
    if (downloadingData) {
      this.downloadCache.delete(id);
    }

    return cachedData.originalSize;
  }

  async download(id: GpuDataId): Promise<ArrayBufferLike> {
    const downloadData = this.downloadCache.get(id);
    if (downloadData) {
      return downloadData.data;
    }

    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }

    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    const gpuReadBuffer = this.backend.device.createBuffer(
        // eslint-disable-next-line no-bitwise
        {size: cachedData.originalSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    commandEncoder.copyBufferToBuffer(
        cachedData.gpuData.buffer /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
        0 /* destination offset */, cachedData.originalSize /* size */
    );
    this.backend.flush();

    const readDataPromise = new Promise<ArrayBuffer>((resolve) => {
      gpuReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const data = gpuReadBuffer.getMappedRange().slice(0);
        gpuReadBuffer.destroy();
        resolve(data);
      });
    });

    this.downloadCache.set(id, {data: readDataPromise});

    return readDataPromise;
  }

  refreshPendingBuffers(): void {
    for (const buffer of this.buffersForUploadingPending) {
      buffer.destroy();
    }
    for (const buffer of this.buffersPending) {
      buffer.destroy();
    }
  }
}

export const createGpuDataManager = (...args: ConstructorParameters<typeof GpuDataManagerImpl>): GpuDataManager =>
    new GpuDataManagerImpl(...args);
