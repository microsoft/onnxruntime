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
   * Release all tensors for a given session.
   */
  releaseTensorsForSession(session: number): void;
  /**
   * Register an externally created MLTensor with a given MLContext and return a TensorId.
   */
  registerTensor(mlContext: MLContext, mlTensor: MLTensor, dataType: MLOperandDataType, shape: number[]): TensorId;
}

let tensorGuid = 1;
const createNewTensorId = (): TensorId => tensorGuid++;

/**
 * Map from MLOperandDataType to size in bits. Using bits instead of bytes to avoid possible precision loss on int4 and uint4.
 */
const webnnDataTypeToSize = new Map<MLOperandDataType, number>([
  ['float32', 32],
  ['float16', 16],
  ['int32', 32],
  ['uint32', 32],
  ['int64', 64],
  ['uint64', 64],
  ['int8', 8],
  ['uint8', 8],
  ['int4', 4],
  ['uint4', 4],
]);

/**
 * Calculate the byte length of a tensor with the given data type and shape.
 */
const calculateByteLength = (dataType: MLOperandDataType, shape: readonly number[]): number => {
  const size = webnnDataTypeToSize.get(dataType);
  if (!size) {
    throw new Error('Unsupported data type.');
  }
  return shape.length > 0 ? Math.ceil((shape.reduce((a, b) => a * b) * size) / 8) : 0;
};

/**
 * TensorWrapper wraps an MLTensor and provides a way to track the last session that used it.
 */
class TensorWrapper {
  // The id of the last session that used this tensor.
  public sessionId: number;

  private mlContext: MLContext;
  private mlTensor: MLTensor;
  private dataType: MLOperandDataType;
  private tensorShape: readonly number[];

  constructor(descriptor: {
    sessionId: number;
    context: MLContext;
    tensor: MLTensor;
    dataType: MLOperandDataType;
    shape: readonly number[];
  }) {
    this.sessionId = descriptor.sessionId;
    this.mlContext = descriptor.context;
    this.mlTensor = descriptor.tensor;
    this.dataType = descriptor.dataType;
    this.tensorShape = descriptor.shape;
  }

  public get tensor(): MLTensor {
    return this.mlTensor;
  }

  public get type(): MLOperandDataType {
    return this.dataType;
  }

  public get shape(): readonly number[] {
    return this.tensorShape;
  }

  public get byteLength(): number {
    return calculateByteLength(this.dataType, this.tensorShape);
  }

  public destroy(): void {
    LOG_DEBUG('verbose', () => '[WebNN] TensorWrapper.destroy');
    this.mlTensor.destroy();
  }

  public write(data: Uint8Array): void {
    this.mlContext.writeTensor(this.mlTensor, data);
  }

  public async read(): Promise<ArrayBuffer>;
  public async read(dstBuffer: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  async read(dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    if (dstBuffer) {
      return this.mlContext.readTensor(this.mlTensor, dstBuffer);
    }
    return this.mlContext.readTensor(this.mlTensor);
  }

  public sameTypeAndShape(dataType: MLOperandDataType, shape: readonly number[]): boolean {
    return (
      this.dataType === dataType &&
      this.tensorShape.length === shape.length &&
      this.tensorShape.every((v, i) => v === shape[i])
    );
  }
}

/**
 * TensorTracker tracks the MLTensor and pending upload data.
 *
 * We need to track the MLTensor and pending upload data because we delay the creation of MLTensor until
 * we know the data type and shape. This is because WebNN only support creating MLTensors with dataTypes and shape.
 */
class TensorIdTracker {
  private activeUpload?: Uint8Array;

  constructor(
    private tensorManager: TensorManagerImpl,
    private wrapper?: TensorWrapper,
  ) {}

  public get tensorWrapper(): TensorWrapper | undefined {
    return this.wrapper;
  }

  public releaseTensor(): void {
    if (this.tensorWrapper) {
      this.tensorManager.releaseTensor(this.tensorWrapper);
      this.wrapper = undefined;
    }
  }

  public async ensureTensor(
    dataType: MLOperandDataType,
    shape: readonly number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    if (this.wrapper) {
      if (this.wrapper.sameTypeAndShape(dataType, shape)) {
        return this.wrapper.tensor;
      } else {
        if (copyOld) {
          if (this.wrapper.byteLength !== calculateByteLength(dataType, shape)) {
            throw new Error('Unable to copy data to tensor with different size.');
          }
          this.activeUpload = new Uint8Array(await this.wrapper.read());
        }
        this.tensorManager.releaseTensor(this.wrapper);
      }
    }

    // eslint-disable-next-line no-bitwise
    const usage = typeof MLTensorUsage == 'undefined' ? undefined : MLTensorUsage.READ | MLTensorUsage.WRITE;
    this.wrapper = await this.tensorManager.getCachedTensor(dataType, shape, usage, true, true);

    if (copyOld && this.activeUpload) {
      this.wrapper.write(this.activeUpload);
      this.activeUpload = undefined;
    }

    return this.wrapper.tensor;
  }

  public upload(data: Uint8Array): void {
    if (this.wrapper) {
      if (data.byteLength === this.wrapper.byteLength) {
        this.wrapper.write(data);
        return;
      } else {
        LOG_DEBUG('verbose', () => 'Data size does not match tensor size. Releasing tensor.');
        this.releaseTensor();
      }
    }

    if (this.activeUpload) {
      this.activeUpload.set(data);
    } else {
      this.activeUpload = new Uint8Array(data);
    }
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
    if (!this.wrapper) {
      throw new Error('Tensor has not been created.');
    }
    if (!dstBuffer) {
      return this.wrapper.read();
    }
    return this.wrapper.read(dstBuffer);
  }
}

class TensorManagerImpl implements TensorManager {
  private tensorTrackersById: Map<TensorId, TensorIdTracker> = new Map();
  private freeTensors: TensorWrapper[] = [];
  private externalTensors: Set<TensorWrapper> = new Set();

  constructor(private backend: WebNNBackend) {}

  public reserveTensorId(): TensorId {
    const tensorId = createNewTensorId();
    this.tensorTrackersById.set(tensorId, new TensorIdTracker(this));
    return tensorId;
  }

  public releaseTensorId(tensorId: TensorId): void {
    const tensorTracker = this.tensorTrackersById.get(tensorId);
    if (!tensorTracker) {
      return;
    }
    this.tensorTrackersById.delete(tensorId);
    if (tensorTracker.tensorWrapper) {
      this.releaseTensor(tensorTracker.tensorWrapper);
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
    const tensor = this.tensorTrackersById.get(tensorId);
    if (!tensor) {
      throw new Error('Tensor not found.');
    }
    return tensor.ensureTensor(dataType, shape, copyOld);
  }

  public upload(tensorId: TensorId, data: Uint8Array): void {
    const tensor = this.tensorTrackersById.get(tensorId);
    if (!tensor) {
      throw new Error('Tensor not found.');
    }
    tensor.upload(data);
  }

  public async download(tensorId: TensorId): Promise<ArrayBuffer>;
  public async download(tensorId: TensorId, dstBuffer: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  async download(tensorId: TensorId, dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    LOG_DEBUG(
      'verbose',
      () => `[WebNN] TensorManager.download {tensorId: ${tensorId}, dstBuffer: ${dstBuffer?.byteLength}}`,
    );
    const tensorTracker = this.tensorTrackersById.get(tensorId);
    if (!tensorTracker) {
      throw new Error('Tensor not found.');
    }
    return tensorTracker.download(dstBuffer);
  }

  public releaseTensorsForSession(sessionId: number): void {
    for (const tensor of this.freeTensors) {
      if (tensor.sessionId === sessionId) {
        tensor.destroy();
      }
    }
    this.freeTensors = this.freeTensors.filter((tensor) => tensor.sessionId !== sessionId);
  }

  public registerTensor(
    mlContext: MLContext,
    mlTensor: MLTensor,
    dataType: MLOperandDataType,
    shape: readonly number[],
  ): TensorId {
    const tensorId = createNewTensorId();
    // Defaulting to READ | WRITE if usage is not provided.
    // eslint-disable-next-line no-bitwise
    const wrapper = new TensorWrapper({
      sessionId: this.backend.currentSessionId,
      context: mlContext,
      tensor: mlTensor,
      dataType,
      shape,
    });
    this.tensorTrackersById.set(tensorId, new TensorIdTracker(this, wrapper));
    this.externalTensors.add(wrapper);
    return tensorId;
  }

  /**
   * Get or create an MLTensor with the given data type and shape.
   */
  public async getCachedTensor(
    dataType: MLOperandDataType,
    shape: readonly number[],
    usage: MLTensorUsageFlags | undefined,
    writable: boolean,
    readable: boolean,
  ): Promise<TensorWrapper> {
    const sessionId = this.backend.currentSessionId;
    for (const [index, tensor] of this.freeTensors.entries()) {
      if (tensor.sameTypeAndShape(dataType, shape)) {
        LOG_DEBUG('verbose', () => `[WebNN] Reusing tensor {dataType: ${dataType}, shape: ${shape}}`);
        const wrapper = this.freeTensors.splice(index, 1)[0];
        wrapper.sessionId = sessionId;
        return wrapper;
      }
    }
    const context = this.backend.currentContext;
    LOG_DEBUG('verbose', () => `[WebNN] MLContext.createTensor {dataType: ${dataType}, shape: ${shape}}`);
    const tensor = await context.createTensor({
      dataType,
      shape,
      dimensions: shape,
      usage,
      writable,
      readable,
    });
    return new TensorWrapper({ sessionId, context, tensor, dataType, shape });
  }

  /**
   * Release tensor for reuse unless external.
   */
  public releaseTensor(tensorWrapper: TensorWrapper) {
    if (this.externalTensors.has(tensorWrapper)) {
      this.externalTensors.delete(tensorWrapper);
    }
    this.freeTensors.push(tensorWrapper);
  }
}

export const createTensorManager = (...args: ConstructorParameters<typeof TensorManagerImpl>): TensorManager =>
  new TensorManagerImpl(...args);
