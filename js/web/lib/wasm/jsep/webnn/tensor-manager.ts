// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WebNNBackend } from '../backend-webnn';
import { LOG_DEBUG } from '../log';

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn.d.ts" />

export type TensorId = number;

/**
 * Manages TensorId to MLTensor mapping.
 */
export interface TensorManager {
  /**
   * Reserve a new TensorId.
   */
  reserveTensorId(): TensorId;
  /**
   * Release a TensorId.
   */
  releaseTensorId(tensorId: TensorId): void;
  /**
   * Ensure a MLTensor is created for the TensorId.
   */
  ensureTensor(
    tensorId: TensorId,
    dataType: MLOperandDataType,
    shape: readonly number[],
    copyOld: boolean,
  ): Promise<MLTensor>;
  /**
   * Upload data to a MLTensor.
   */
  upload(tensorId: TensorId, data: Uint8Array): void;
  /**
   * Download data from a MLTensor.
   */
  download(tensorId: TensorId): Promise<ArrayBuffer>;
  download(tensorId: TensorId, dstTensor: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  /**
   * Release all tensors for a MLContext.
   */
  releaseTensorsForContext(mlContext: MLContext): void;
  /**
   * Register an externally created MLTensor with a given MLContext and return a TensorId.
   */
  registerTensor(mlContext: MLContext, mlTensor: MLTensor, dataType: MLOperandDataType, shape: number[]): TensorId;
}

let tensorGuid = 1;
const createNewTensorId = (): TensorId => tensorGuid++;

export type MLTensorEntry = [MLTensor, MLOperandDataType, readonly number[]];

/**
 * TensorTracker tracks the MLTensor and pending upload data.
 *
 * We need to track the MLTensor and pending upload data because we delay the creation of MLTensor until
 * we know the data type and shape. This is because future implementations of WebNN will only support creating
 * MLTensors with dataTypes and shape.
 */
class TensorTracker {
  private tensorEntry?: MLTensorEntry;
  private activeUpload?: Uint8Array;
  private tensorCache: MLTensorEntry[];

  constructor(
    private mlContext?: MLContext,
    tensorEntry?: MLTensorEntry,
  ) {
    this.tensorEntry = tensorEntry;
    this.tensorCache = tensorEntry ? [tensorEntry] : [];
  }

  public get tensor(): MLTensor | undefined {
    return this.tensorEntry?.[0];
  }

  public get context(): MLContext {
    if (!this.mlContext) {
      throw new Error('MLContext has not been set.');
    }
    return this.mlContext;
  }

  public set context(mlContext: MLContext) {
    if (this.mlContext && this.mlContext !== mlContext) {
      throw new Error('MLTensor in use in a different MLContext.');
    }
    this.mlContext = mlContext;
  }

  public destroy(): void {
    for (const [mlTensor] of this.tensorCache) {
      mlTensor.destroy();
    }
    this.tensorCache = [];
    this.tensorEntry = undefined;
  }

  public trySelectTensor(context: MLContext, tryMLTensor: MLTensor): boolean {
    for (const [mlTensor, dataType, shape] of this.tensorCache) {
      if (tryMLTensor === mlTensor) {
        if (this.context !== context) {
          throw new Error('MLTensor cannot be registered with a different MLContext.');
        }
        this.tensorEntry = [mlTensor, dataType, shape];
        return true;
      }
    }
    return false;
  }

  public async ensureTensor(
    dataType: MLOperandDataType,
    shape: readonly number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    if (this.tensorEntry) {
      const [mlTensor, existingDataType, existingShape] = this.tensorEntry;
      if (existingDataType === dataType && existingShape.every((v, i) => v === shape[i])) {
        return mlTensor;
      }
    }

    for (const [mlTensor, existingDataType, existingShape] of this.tensorCache) {
      if (existingDataType === dataType && existingShape.every((v, i) => v === shape[i])) {
        if (copyOld && this.tensorEntry) {
          // WebNN does not support copyTensorToTensor, so we need to read and write the tensors.
          LOG_DEBUG(
            'verbose',
            () => `[WebNN] Slowdown may occur, having to copy existing tensor {dataType: ${dataType}, shape: ${shape}}`,
          );
          const data = await this.context.readTensor(this.tensorEntry[0]);
          this.context.writeTensor(mlTensor, data);
        }
        this.tensorEntry = [mlTensor, existingDataType, existingShape];
        return mlTensor;
      }
    }
    LOG_DEBUG('verbose', () => `[WebNN] MLContext.createTensor {dataType: ${dataType}, shape: ${shape}}`);
    // eslint-disable-next-line no-bitwise
    const usage = MLTensorUsage.READ | MLTensorUsage.WRITE;
    const tensor = await this.context.createTensor({
      dataType,
      shape,
      // Assign both shape and dimensions while transitioning to new API.
      dimensions: shape,
      usage,
    });
    this.tensorEntry = [tensor, dataType, shape];
    this.tensorCache.push(this.tensorEntry);

    if (this.activeUpload) {
      this.mlContext?.writeTensor(tensor, this.activeUpload);
      this.activeUpload = undefined;
    }

    return tensor;
  }

  public upload(data: Uint8Array): void {
    if (!this.tensorEntry) {
      this.activeUpload = new Uint8Array(data);
      return;
    }
    this.mlContext?.writeTensor(this.tensorEntry[0], data);
  }

  public async download(dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    if (this.activeUpload) {
      if (dstBuffer) {
        if (dstBuffer instanceof ArrayBuffer) {
          new Uint8Array(dstBuffer).set(this.activeUpload);
        } else {
          new Uint8Array(dstBuffer.buffer, dstBuffer.byteOffset, dstBuffer.byteLength).set(this.activeUpload);
        }

        return;
      } else {
        return this.activeUpload.buffer;
      }
    }
    if (!this.tensorEntry) {
      throw new Error('Tensor has not been created.');
    }
    if (dstBuffer) {
      return this.context.readTensor(this.tensorEntry[0], dstBuffer);
    }
    return this.context.readTensor(this.tensorEntry[0]);
  }
}

class TensorManagerImpl implements TensorManager {
  private tensorsById = new Map<TensorId, TensorTracker>();
  private tensorIdsByContext = new Map<MLContext, Set<TensorId>>();

  constructor(private backend: WebNNBackend) {}

  public reserveTensorId(): TensorId {
    const tensorId = createNewTensorId();
    this.tensorsById.set(tensorId, new TensorTracker());
    return tensorId;
  }

  public releaseTensorId(tensorId: TensorId): void {
    const tensorTracker = this.tensorsById.get(tensorId);
    if (!tensorTracker) {
      return;
    }
    tensorTracker.destroy();
    this.tensorsById.delete(tensorId);
    for (const [mlContext, tensors] of this.tensorIdsByContext) {
      if (tensors.has(tensorId)) {
        tensors.delete(tensorId);
        if (tensors.size === 0) {
          this.tensorIdsByContext.delete(mlContext);
        }
        break;
      }
    }
  }

  public async ensureTensor(
    tensorId: TensorId,
    dataType: MLOperandDataType,
    shape: number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    LOG_DEBUG(
      'verbose',
      () =>
        `[WebNN] TensorManager.ensureTensor {tensorId: ${tensorId}, dataType: ${
          dataType
        }, shape: ${shape}, copyOld: ${copyOld}}`,
    );
    const tensor = this.tensorsById.get(tensorId);
    if (!tensor) {
      throw new Error('Tensor not found.');
    }
    tensor.context = this.backend.currentContext;
    if (!this.tensorIdsByContext.has(this.backend.currentContext)) {
      this.tensorIdsByContext.set(this.backend.currentContext, new Set());
    }
    this.tensorIdsByContext.get(this.backend.currentContext)?.add(tensorId);
    return tensor.ensureTensor(dataType, shape, copyOld);
  }

  public upload(tensorId: TensorId, data: Uint8Array): void {
    this.tensorsById.get(tensorId)!.upload(data);
  }

  public async download(tensorId: TensorId): Promise<ArrayBuffer>;
  public async download(tensorId: TensorId, dstBuffer: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  async download(tensorId: TensorId, dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    LOG_DEBUG(
      'verbose',
      () => `[WebNN] TensorManager.download {tensorId: ${tensorId}, dstBuffer: ${dstBuffer?.byteLength}}`,
    );
    return this.tensorsById.get(tensorId)!.download(dstBuffer);
  }

  public releaseTensorsForContext(mlContext: MLContext): void {
    const tensors = this.tensorIdsByContext.get(mlContext);
    if (!tensors) {
      return;
    }
    for (const tensorId of tensors) {
      this.tensorsById.get(tensorId)!.destroy();
      this.tensorsById.delete(tensorId);
    }
    this.tensorIdsByContext.delete(mlContext);
  }

  public registerTensor(
    mlContext: MLContext,
    mlTensor: MLTensor,
    dataType: MLOperandDataType,
    shape: readonly number[],
  ): TensorId {
    for (const [tensorId, tensorTracker] of this.tensorsById) {
      if (tensorTracker.trySelectTensor(mlContext, mlTensor)) {
        return tensorId;
      }
    }
    const tensorId = createNewTensorId();
    this.tensorsById.set(tensorId, new TensorTracker(mlContext, [mlTensor, dataType, shape]));
    let tensors = this.tensorIdsByContext.get(mlContext);
    if (!tensors) {
      tensors = new Set();
      this.tensorIdsByContext.set(mlContext, tensors);
    }
    tensors.add(tensorId);
    return tensorId;
  }
}

export const createTensorManager = (...args: ConstructorParameters<typeof TensorManagerImpl>): TensorManager =>
  new TensorManagerImpl(...args);
