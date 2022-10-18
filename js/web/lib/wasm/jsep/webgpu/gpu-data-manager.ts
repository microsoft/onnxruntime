// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WebGpuBackend} from '../backend-webgpu';

import {GpuData, GpuDataId, GpuDataType} from './types';

/**
 * manages GpuDataId -> GpuBuffer
 */
export interface GpuDataManager {
  /**
   * upload data to GPU.
   */
  upload(id: GpuDataId, data: Uint8Array): void;
  /**
   * create new data on GPU.
   */
  create(size: number): GpuData;
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
   * download the data from GPU.
   */
  download(id: GpuDataId): Promise<ArrayBufferLike>;
}

interface StorageCacheValue {
  gpuData: GpuData;
  originalSize: number;
}

interface DownloadCacheValue {
  gpuData: GpuData;
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

  constructor(private backend: WebGpuBackend /* , private reuseBuffer: boolean */) {
    this.storageCache = new Map();
    this.downloadCache = new Map();
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
    const gpuBufferForUploading =
        this.backend.device.createBuffer({mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE});

    // copy (upload) data
    const arrayBuffer = gpuBufferForUploading.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    gpuBufferForUploading.unmap();


    // GPU copy
    this.backend.getCommandEncoder().copyBufferToBuffer(gpuBufferForUploading, 0, gpuDataCache.gpuData.buffer, 0, size);
    this.backend.flush();

    gpuBufferForUploading.destroy();
  }

  create(size: number): GpuData {
    // !!!
    // !!! IMPORTANT: TODO: whether we should keep the storage buffer every time, or always create new ones.
    // !!!                  This need to be figured out by performance test results.
    // !!!

    const bufferSize = calcNormalizedBufferSize(size);

    // create gpu buffer
    const gpuBuffer =
        // eslint-disable-next-line no-bitwise
        this.backend.device.createBuffer({size: bufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    const gpuData = {id: createNewGpuDataId(), type: GpuDataType.default, buffer: gpuBuffer};
    this.storageCache.set(gpuData.id, {gpuData, originalSize: size});
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

    this.storageCache.delete(id);
    cachedData.gpuData.buffer.destroy();

    const downloadingData = this.downloadCache.get(id);
    if (downloadingData) {
      void downloadingData.data.then(() => {
        downloadingData.gpuData.buffer.destroy();
      });
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

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    return gpuReadBuffer.getMappedRange();

    // TODO: release gpuReadBuffer
  }
}

export const createGpuDataManager = (...args: ConstructorParameters<typeof GpuDataManagerImpl>): GpuDataManager =>
    new GpuDataManagerImpl(...args);
