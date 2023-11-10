// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

/**
 *  tuple elements are: ORT element type; dims; tensor data
 */
export type SerializableTensor = [Tensor.Type, readonly number[], Tensor.DataType];

/**
 *  tuple elements are: InferenceSession handle; input names; output names
 */
export type SerializableSessionMetadata = [number, string[], string[]];

/**
 *  tuple elements are: modeldata.offset, modeldata.length
 */
export type SerializableModeldata = [number, number];

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
    sessionId: number; inputIndices: number[]; inputs: SerializableTensor[]; outputIndices: number[];
    options: InferenceSession.RunOptions;
  };
  out?: SerializableTensor[];
}

interface MesssageEndProfiling extends MessageError {
  type: 'end-profiling';
  in ?: number;
}

export type OrtWasmMessage = MessageInitWasm|MessageInitOrt|MessageCreateSessionAllocate|MessageCreateSessionFinalize|
    MessageCreateSession|MessageReleaseSession|MessageRun|MesssageEndProfiling;
