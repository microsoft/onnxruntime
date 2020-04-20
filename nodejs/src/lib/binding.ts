import {InferenceSession as InferenceSessionInterface} from './inference-session';
import {OnnxValue} from './onnx-value';

/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
  export interface InferenceSession {
    loadModel(modelPath: string): void;
    loadModel(buffer: ArrayBuffer, byteOffset: number, byteLength: number): void;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(feeds: InferenceSession.FeedsType, fetches: InferenceSession.FetchesType,
        options: InferenceSession.RunOptions): InferenceSession.ReturnType;
  }

  export namespace InferenceSession {
    type FeedsType = {[name: string]: OnnxValue};
    type FetchesType = {[name: string]: OnnxValue | null};
    type ReturnType = {[name: string]: OnnxValue};
    type RunOptions = InferenceSessionInterface.RunOptions;
  }

  export interface InferenceSessionConstructor {
    new(options: InferenceSessionInterface.SessionOptions): InferenceSession;
  }
}

// export native binding
export const binding =
    require(`../bin/napi-v4/cpu/onnxruntime_binding.node`) as {InferenceSession: Binding.InferenceSessionConstructor};
