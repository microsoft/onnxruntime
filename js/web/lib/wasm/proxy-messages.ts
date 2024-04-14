// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type {Env, InferenceSession, Tensor} from 'onnxruntime-common';

export type SerializableTensorMetadata =
    [dataType: Tensor.Type, dims: readonly number[], data: Tensor.DataType, location: 'cpu'];

export type GpuBufferMetadata = {
  gpuBuffer: Tensor.GpuBufferType;
  download?: () => Promise<Tensor.DataTypeMap[Tensor.GpuBufferDataTypes]>;
  dispose?: () => void;
};

export type UnserializableTensorMetadata =
    [dataType: Tensor.Type, dims: readonly number[], data: GpuBufferMetadata, location: 'gpu-buffer']|
    [dataType: Tensor.Type, dims: readonly number[], data: Tensor.DataType, location: 'cpu-pinned'];

export type TensorMetadata = SerializableTensorMetadata|UnserializableTensorMetadata;

export type SerializableSessionMetadata = [sessionHandle: number, inputNames: string[], outputNames: string[]];

export type SerializableModeldata = [modelDataOffset: number, modelDataLength: number];

interface MessageError {
  err?: string;
}

interface MessageInitWasm extends MessageError {
  type: 'init-wasm';
  in ?: Env.WebAssemblyFlags;
}

interface MessageInitOrt extends MessageError {
  type: 'init-ort';
  in ?: Env;
}

interface MessageCreateSessionAllocate extends MessageError {
  type: 'create_allocate';
  in ?: {model: Uint8Array};
  out?: SerializableModeldata;
}

interface MessageCreateSessionFinalize extends MessageError {
  type: 'create_finalize';
  in ?: {modeldata: SerializableModeldata; options?: InferenceSession.SessionOptions};
  out?: SerializableSessionMetadata;
}

interface MessageCreateSession extends MessageError {
  type: 'create';
  in ?: {model: Uint8Array; options?: InferenceSession.SessionOptions};
  out?: SerializableSessionMetadata;
}

interface MessageReleaseSession extends MessageError {
  type: 'release';
  in ?: number;
}

interface MessageRun extends MessageError {
  type: 'run';
  in ?: {
    sessionId: number; inputIndices: number[]; inputs: SerializableTensorMetadata[]; outputIndices: number[];
    options: InferenceSession.RunOptions;
  };
  out?: SerializableTensorMetadata[];
}

interface MesssageEndProfiling extends MessageError {
  type: 'end-profiling';
  in ?: number;
}

interface MessageIsOrtEnvInitialized extends MessageError {
  type: 'is-ort-env-initialized';
  out?: boolean;
}

export type OrtWasmMessage = MessageInitWasm|MessageInitOrt|MessageCreateSessionAllocate|MessageCreateSessionFinalize|
    MessageCreateSession|MessageReleaseSession|MessageRun|MesssageEndProfiling|MessageIsOrtEnvInitialized;
