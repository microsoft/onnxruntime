// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from 'onnxruntime-common';

import {OrtWasmModule} from '../binding/ort-wasm';
import {DataType, getTensorElementSize} from '../wasm-common';

import {WebGpuBackend} from './backend-webgpu';
import {LOG_DEBUG} from './log';
import {TensorView} from './tensor-view';
import {ShapeUtil} from './util';
import {ComputeContext, ComputeContextInputsOutputsMapping, ProgramInfo} from './webgpu/types';

/* eslint-disable no-bitwise */

class TensorViewImpl implements TensorView {
  constructor(
      private module: OrtWasmModule, public readonly dataType: number, public readonly data: number,
      public readonly dims: readonly number[]) {}

  getFloat32Array(): Float32Array {
    if (this.dataType !== DataType.float) {
      throw new Error('Invalid data type');
    }
    const elementCount = ShapeUtil.size(this.dims);
    return elementCount === 0 ? new Float32Array() :
                                new Float32Array(this.module.HEAP8.buffer, this.data, elementCount);
  }

  getBigInt64Array(): BigInt64Array {
    if (this.dataType !== DataType.int64) {
      throw new Error('Invalid data type');
    }
    const elementCount = ShapeUtil.size(this.dims);
    return elementCount === 0 ? new BigInt64Array() :
                                new BigInt64Array(this.module.HEAP8.buffer, this.data, elementCount);
  }

  getInt32Array(): Int32Array {
    if (this.dataType !== DataType.int32) {
      throw new Error('Invalid data type');
    }
    const elementCount = ShapeUtil.size(this.dims);
    return elementCount === 0 ? new Int32Array() : new Int32Array(this.module.HEAP8.buffer, this.data, elementCount);
  }

  reshape(newDims: readonly number[]): TensorView {
    if (ShapeUtil.size(newDims) !== ShapeUtil.size(this.dims)) {
      throw new Error('Invalid new shape');
    }
    return new TensorViewImpl(this.module, this.dataType, this.data, newDims);
  }
}

class ComputeContextImpl implements ComputeContext {
  readonly opKernelContext: number;
  readonly inputs: readonly TensorView[];
  readonly outputCount: number;
  get kernelCustomData(): {[key: string]: unknown} {
    return this.backend.currentKernelCustomData;
  }
  get customDataBuffer(): Uint8Array {
    return this.module.HEAPU8.subarray(this.customDataOffset, this.customDataOffset + this.customDataSize);
  }
  private customDataOffset = 0;
  private customDataSize = 0;
  constructor(private module: OrtWasmModule, private backend: WebGpuBackend, contextDataOffset: number) {
    const heapU32 = module.HEAPU32;

    // extract context data
    let dataIndex = (contextDataOffset >> 2);
    this.opKernelContext = heapU32[dataIndex++];
    const inputCount = heapU32[dataIndex++];
    this.outputCount = heapU32[dataIndex++];
    this.customDataOffset = heapU32[dataIndex++];
    this.customDataSize = heapU32[dataIndex++];

    const inputs: TensorView[] = [];
    for (let i = 0; i < inputCount; i++) {
      const dataType = heapU32[dataIndex++];
      const data = heapU32[dataIndex++];
      const dim = heapU32[dataIndex++];
      const dims: number[] = [];
      for (let d = 0; d < dim; d++) {
        dims.push(heapU32[dataIndex++]);
      }
      inputs.push(new TensorViewImpl(module, dataType, data, dims));
    }
    this.inputs = inputs;
  }

  compute(program: ProgramInfo, inputsOutputsMapping?: ComputeContextInputsOutputsMapping): TensorView[] {
    // prepare inputs. inputs should always be valid data.
    const mappedInputs =
        inputsOutputsMapping?.inputs?.map(i => typeof i === 'number' ? this.inputs[i] : i) ?? this.inputs;
    // prepare outputs.
    const outputIndices = inputsOutputsMapping?.outputs ?? [];
    const createKernelOutput = (index: number, dataType: number, dims: readonly number[]): TensorView =>
        new TensorViewImpl(this.module, dataType, this.output(index, dims), dims);
    const createTemporaryOutput = (dataType: number, dims: readonly number[]): TensorView => {
      const elementSize = getTensorElementSize(dataType);
      if (!elementSize) {
        throw new Error(`Unsupported data type: ${dataType}`);
      }
      const bufferSize = elementSize * ShapeUtil.size(dims);
      return new TensorViewImpl(this.module, dataType, this.backend.gpuDataManager.create(bufferSize).id, dims);
    };
    return this.backend.run(program, mappedInputs, outputIndices, createKernelOutput, createTemporaryOutput);
  }

  output(index: number, dims: readonly number[]): number {
    const stack = this.module.stackSave();
    try {
      const data = this.module.stackAlloc((1 + dims.length) * 4 /* sizeof(size_t) */);
      let offset = data >> 2;
      this.module.HEAPU32[offset++] = dims.length;
      for (let i = 0; i < dims.length; i++) {
        this.module.HEAPU32[offset++] = dims[i];
      }
      return this.module._JsepOutput(this.opKernelContext, index, data);
    } catch (e) {
      throw new Error(
          `Failed to generate kernel's output[${index}] with dims [${dims}]. ` +
          'If you are running with pre-allocated output, please make sure the output type/dims are correct. ' +
          `Error: ${e}`);
    } finally {
      this.module.stackRestore(stack);
    }
  }
}

export const init = async(module: OrtWasmModule, env: Env): Promise<void> => {
  const init = module.jsepInit;
  if (init && navigator.gpu) {
    if (!env.wasm.simd) {
      throw new Error(
          'Not supported for WebGPU=ON and SIMD=OFF. Please set `env.wasm.simd` to true when using WebGPU EP');
    }
    const backend = new WebGpuBackend();
    await backend.initialize(env);

    init(
        // backend
        backend,

        // jsepAlloc()
        (size: number) => backend.alloc(size),

        // jsepFree()
        (ptr: number) => backend.free(ptr),

        // jsepCopy(src, dst, size, isSourceGpu)
        (src: number, dst: number, size: number, isSourceGpu = false) => {
          if (isSourceGpu) {
            LOG_DEBUG('verbose', () => `[WebGPU] jsepCopyGpuToGpu: src=${src}, dst=${dst}, size=${size}`);
            backend.memcpy(src, dst);
          } else {
            LOG_DEBUG('verbose', () => `[WebGPU] jsepCopyCpuToGpu: dataOffset=${src}, gpuDataId=${dst}, size=${size}`);
            const data = module.HEAPU8.subarray(src, src + size);
            backend.upload(dst, data);
          }
        },

        // jsepCopyAsync(src, dst, size)
        async(gpuDataId: number, dataOffset: number, size: number):
            Promise<void> => {
              LOG_DEBUG(
                  'verbose',
                  () => `[WebGPU] jsepCopyGpuToCpu: gpuDataId=${gpuDataId}, dataOffset=${dataOffset}, size=${size}`);

              await backend.download(gpuDataId, () => module.HEAPU8.subarray(dataOffset, dataOffset + size));
            },

        // jsepCreateKernel
        (name: string, kernel: number, attribute: unknown) => backend.createKernel(
            name, kernel, attribute,
            env.debug || backend.isQueryEnabled() ? module.UTF8ToString(module._JsepGetNodeName(kernel)) : `${kernel}`),

        // jsepReleaseKernel
        (kernel: number) => backend.releaseKernel(kernel),

        // jsepRun
        (kernel: number, contextDataOffset: number, sessionHandle: number, errors: Array<Promise<string|null>>) => {
          LOG_DEBUG(
              'verbose',
              () => `[WebGPU] jsepRun: sessionHandle=${sessionHandle}, kernel=${kernel}, contextDataOffset=${
                  contextDataOffset}`);
          const context = new ComputeContextImpl(module, backend, contextDataOffset);
          return backend.computeKernel(kernel, context, errors);
        });
  }
};
