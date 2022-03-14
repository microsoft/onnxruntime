// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Guid} from 'guid-typescript';
import {sizeof, Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {GpuData, GpuDataId, GpuDataType} from './types';

/**
 * manages GpuDataId -> GpuBuffer
 */
export interface GpuDataManager {
  uploadData(tensor: Tensor, gpuDataType: GpuDataType): GpuData;
  createData(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): GpuData;
  releaseData(tensorId: Tensor.Id): void;
  downloadData(tensorId: Tensor.Id): Promise<ArrayBufferLike>;
}

interface DefaultCacheValue {
  gpuData: GpuData;
  size: number;
}

interface DownloadCacheValue {
  gpuData: GpuData;
  data: Promise<ArrayBufferLike>;
}

class GpuDataManagerImpl implements GpuDataManager {
  defaultCache: Map<GpuDataId, DefaultCacheValue>;
  downloadCache: Map<GpuDataId, DownloadCacheValue>;
  constructor(private device: GPUDevice) {
    this.defaultCache = new Map();
    this.downloadCache = new Map();
  }

  uploadData(tensor: Tensor, gpuDataType: GpuDataType): GpuData {
    if (gpuDataType !== GpuDataType.default) {
      throw new Error('we only support default GPU data type now');
    }

    const cachedData = this.defaultCache.get(tensor.dataId);
    if (cachedData) {
      return cachedData.gpuData;
    }

    const src = tensor.numberData;
    const srcArrayBuffer = src.buffer;
    const srcOffset = src.byteOffset;
    const srcLength = src.byteLength;

    // create gpu buffer
    const gpuBuffer =
        this.device.createBuffer({mappedAtCreation: true, size: srcLength, usage: GPUBufferUsage.STORAGE});

    // copy (upload) data
    const arrayBuffer = gpuBuffer.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    gpuBuffer.unmap();

    const gpuData = {id: tensor.dataId, type: GpuDataType.default, buffer: gpuBuffer};
    this.defaultCache.set(gpuData.id, {gpuData, size: srcLength});
    return gpuData;
  }

  createData(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): GpuData {
    if (gpuDataType !== GpuDataType.default) {
      throw new Error('we only support default GPU data type now');
    }

    // !!!
    // !!! IMPORTANT: TODO: whether we should keep the storage buffer every time, or always create new ones.
    // !!!                  This need to be figured out by performance test results.
    // !!!

    const elemCount = ShapeUtil.size(dims);
    const bufferLength = sizeof(type) * elemCount;

    // create gpu buffer
    const gpuBuffer =
        // eslint-disable-next-line no-bitwise
        this.device.createBuffer({size: bufferLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

    const gpuData = {id: Guid.create(), type: GpuDataType.default, buffer: gpuBuffer};
    this.defaultCache.set(gpuData.id, {gpuData, size: bufferLength});
    return gpuData;
  }

  releaseData(tensorId: Tensor.Id): void {
    const cachedData = this.defaultCache.get(tensorId);
    if (!cachedData) {
      throw new Error('releasing data does not exist');
    }

    this.defaultCache.delete(tensorId);
    cachedData.gpuData.buffer.destroy();
  }

  async downloadData(tensorId: Tensor.Id): Promise<ArrayBufferLike> {
    const downloadData = this.downloadCache.get(tensorId);
    if (downloadData) {
      return downloadData.data;
    }

    const cachedData = this.defaultCache.get(tensorId);
    if (!cachedData) {
      throw new Error('data does not exist');
    }

    const commandEncoder = this.device.createCommandEncoder();
    const gpuReadBuffer =
        // eslint-disable-next-line no-bitwise
        this.device.createBuffer({size: cachedData.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    commandEncoder.copyBufferToBuffer(
        cachedData.gpuData.buffer /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
        0 /* destination offset */, cachedData.size /* size */
    );
    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    return gpuReadBuffer.getMappedRange();
  }
}

export const createGpuDataManager = (device: GPUDevice): GpuDataManager => new GpuDataManagerImpl(device);
