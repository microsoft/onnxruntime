import type {
  SessionHandler,
  InferenceSession,
  Tensor,
} from 'onnxruntime-common';

export interface SupportedBackend {
  name: string;
}

export interface ValueMetadata {
  name: string;
  isTensor: boolean;
  type: number;
  shape: number[];
  symbolicDimensions: string[];
}

type FeedsType = SessionHandler.FeedsType;
type FetchesType = SessionHandler.FetchesType;
type ReturnType = SessionHandler.ReturnType;
type SessionOptions = InferenceSession.SessionOptions & {
  ortExtLibPath?: string;
};
type RunOptions = InferenceSession.RunOptions;

export interface InferenceSessionImpl {
  loadModel(modelPath: string, options: SessionOptions): Promise<void>;
  loadModel(
    buffer: ArrayBuffer,
    byteOffset: number,
    byteLength: number,
    options: SessionOptions
  ): Promise<void>;

  readonly inputMetadata: ValueMetadata[];
  readonly outputMetadata: ValueMetadata[];

  run(
    feeds: FeedsType,
    fetches: FetchesType,
    options: RunOptions
  ): Promise<ReturnType>;

  endProfiling(): void;

  dispose(): void;
}

export declare interface OrtApi {
  createInferenceSession(): InferenceSessionImpl;

  listSupportedBackends(): SupportedBackend[];

  initOrtOnce(logLevel: number, tensorConstructor: typeof Tensor): void;

  version: string;
}
