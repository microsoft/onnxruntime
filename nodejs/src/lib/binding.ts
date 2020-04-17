import {InferenceSession as InferenceSessionInterface} from './inference-session';
import {OnnxValue} from './onnx-value';

export declare namespace Binding {
  type FeedsType = {[name: string]: OnnxValue};
  type FetchesType = {[name: string]: OnnxValue | null};
  type ReturnType = {[name: string]: OnnxValue};

  export interface InferenceSession {
    loadModel(modelPath: string): void;
    loadModel(buffer: ArrayBuffer, byteOffset: number, byteLength: number): void;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(feeds: FeedsType, fetches: FetchesType, options: InferenceSessionInterface.RunOptions): ReturnType;
  }

  export interface InferenceSessionConstructor {
    new(options: InferenceSessionInterface.SessionOptions): InferenceSession;
  }
}

// construct binding file path
const GPU_ENABLED = false;  // TODO: handle GPU

export const binding =
    require(`../bin/napi-v4/${GPU_ENABLED ? 'gpu' : 'cpu'}/onnxruntime_binding${GPU_ENABLED ? '_gpu' : ''}.node`) as
    {InferenceSession: Binding.InferenceSessionConstructor};
