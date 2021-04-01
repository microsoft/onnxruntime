// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {resolveBackend, SessionHandler} from './backend';
import {OnnxValue} from './onnx-value';
import {Tensor} from './tensor';

export declare namespace InferenceSession {
  //#region input/output types

  type OnnxValueMapType = {readonly [name: string]: OnnxValue};
  type NullableOnnxValueMapType = {readonly [name: string]: OnnxValue | null};

  /**
   * A feeds (model inputs) is an object that use input names as keys and OnnxValue as corresponding values.
   */
  type FeedsType = OnnxValueMapType;

  /**
   * A fetches (model outputs) could be one of the following:
   *
   * - Omitted. Use model's output names definition.
   * - An array of string indicating the output names.
   * - An object that use output names as keys and OnnxValue or null as corresponding values.
   *
   * REMARK: different from input argument, in output, OnnxValue is optional. If an OnnxValue is present it will be
   * used as a pre-allocated value by the inference engine; if omitted, inference engine will allocate buffer
   * internally.
   */
  type FetchesType = readonly string[]|NullableOnnxValueMapType;

  type ReturnType = {readonly [name: string]: OnnxValue};

  //#endregion

  //#region session options

  /**
   * A set of configurations for session behavior.
   */
  export interface SessionOptions {
    /**
     * An array of execution provider options.
     *
     * An execution provider option can be a string indicating the name of the execution provider,
     * or an object of corresponding type.
     */
    executionProviders?: readonly SessionOptions.ExecutionProviderConfig[];

    /**
     * The intra OP threads number.
     */
    intraOpNumThreads?: number;

    /**
     * The inter OP threads number.
     */
    interOpNumThreads?: number;

    /**
     * The optimization level.
     */
    graphOptimizationLevel?: 'disabled'|'basic'|'extended'|'all';

    /**
     * Whether enable CPU memory arena.
     */
    enableCpuMemArena?: boolean;

    /**
     * Whether enable memory pattern.
     */
    enableMemPattern?: boolean;

    /**
     * Execution mode.
     */
    executionMode?: 'sequential'|'parallel';

    /**
     * Log ID.
     */
    logId?: string;

    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/logging/severity.h
     */
    logSeverityLevel?: 0|1|2|3|4;
  }

  export namespace SessionOptions {
    //#region execution providers

    // currently we only have CPU and CUDA EP support. in future to support more.
    interface ExecutionProviderOptionMap {
      cpu: CpuExecutionProviderOption;
      cuda: CudaExecutionProviderOption;
      webgl: WebGLExecutionProviderOption;
    }

    type ExecutionProviderName = keyof ExecutionProviderOptionMap;
    type ExecutionProviderConfig = ExecutionProviderOptionMap[ExecutionProviderName]|ExecutionProviderName;

    export interface CpuExecutionProviderOption {
      readonly name: 'cpu';
      useArena?: boolean;
    }
    export interface CudaExecutionProviderOption {
      readonly name: 'cuda';
      deviceId?: number;
    }
    export interface WebGLExecutionProviderOption {
      readonly name: 'webgl';
      // TODO: add flags
    }
    //#endregion
  }

  //#endregion

  //#region run options

  /**
   * A set of configurations for inference run behavior
   */
  export interface RunOptions {
    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/logging/severity.h
     */
    logSeverityLevel?: 0|1|2|3|4;

    /**
     * A tag for the Run() calls using this
     */
    tag?: string;
  }

  //#endregion

  //#region value metadata

  // eslint-disable-next-line @typescript-eslint/no-empty-interface
  interface ValueMetadata {
    // TBD
  }

  //#endregion
}

type SessionOptions = InferenceSession.SessionOptions;
type RunOptions = InferenceSession.RunOptions;
type FeedsType = InferenceSession.FeedsType;
type FetchesType = InferenceSession.FetchesType;
type ReturnType = InferenceSession.ReturnType;

/**
 * Represent a runtime instance of an ONNX model.
 */
export interface InferenceSession {
  //#region run()

  /**
   * Execute the model asynchronously with the given feeds and options.
   *
   * @param feeds Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param options Optional. A set of options that controls the behavior of model inference.
   */
  run(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;

  /**
   * Execute the model asynchronously with the given feeds, fetches and options.
   *
   * @param feeds Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param fetches Representation of the model output. See type description of `InferenceSession.OutputType` for
   * detail.
   * @param options Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;

  //#endregion

  //#region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: readonly string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: readonly string[];

  // /**
  //  * Get input metadata of the loaded model.
  //  */
  // readonly inputMetadata: ReadonlyArray<Readonly<InferenceSession.ValueMetadata>>;

  // /**
  //  * Get output metadata of the loaded model.
  //  */
  // readonly outputMetadata: ReadonlyArray<Readonly<InferenceSession.ValueMetadata>>;

  //#endregion
}

export interface InferenceSessionFactory {
  //#region create()

  /**
   * Create a new inference session and load model asynchronously from an ONNX model file.
   *
   * @param path The file path of the model to load.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(path: string, options?: SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from an array bufer.
   *
   * @param buffer An ArrayBuffer representation of an ONNX model.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, options?: SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from segment of an array bufer.
   *
   * @param buffer An ArrayBuffer representation of an ONNX model.
   * @param byteOffset The beginning of the specified portion of the array buffer.
   * @param byteLength The length in bytes of the array buffer.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, byteOffset: number, byteLength?: number, options?: SessionOptions):
      Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from a Uint8Array.
   *
   * @param buffer A Uint8Array representation of an ONNX model.
   * @param options specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: Uint8Array, options?: SessionOptions): Promise<InferenceSession>;

  //#endregion
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
// export const InferenceSession: InferenceSessionFactory = impl;
export class InferenceSession implements InferenceSession {
  private constructor(handler: SessionHandler) {
    this.#handler = handler;
  }
  run(feeds: FeedsType, options?: RunOptions): Promise<ReturnType>;
  run(feeds: FeedsType, fetches: FetchesType, options?: RunOptions): Promise<ReturnType>;
  async run(feeds: FeedsType, arg1?: FetchesType|RunOptions, arg2?: RunOptions): Promise<ReturnType> {
    const fetches: {[name: string]: OnnxValue|null} = {};
    let options: RunOptions = {};
    // check inputs
    if (typeof feeds !== 'object' || feeds === null || feeds instanceof Tensor || Array.isArray(feeds)) {
      throw new TypeError(
          '\'feeds\' must be an object that use input names as keys and OnnxValue as corresponding values.');
    }

    let isFetchesEmpty = true;
    // determine which override is being used
    if (typeof arg1 === 'object') {
      if (arg1 === null) {
        throw new TypeError('Unexpected argument[1]: cannot be null.');
      }
      if (arg1 instanceof Tensor) {
        throw new TypeError('\'fetches\' cannot be a Tensor');
      }

      if (Array.isArray(arg1)) {
        if (arg1.length === 0) {
          throw new TypeError('\'fetches\' cannot be an empty array.');
        }
        isFetchesEmpty = false;
        // output names
        for (const name of arg1) {
          if (typeof name !== 'string') {
            throw new TypeError('\'fetches\' must be a string array or an object.');
          }
          if (this.outputNames.indexOf(name) === -1) {
            throw new RangeError(`'fetches' contains invalid output name: ${name}.`);
          }
          fetches[name] = null;
        }

        if (typeof arg2 === 'object' && arg2 !== null) {
          options = arg2;
        } else if (typeof arg2 !== 'undefined') {
          throw new TypeError('\'options\' must be an object.');
        }
      } else {
        // decide whether arg1 is fetches or options
        // if any output name is present and its value is valid OnnxValue, we consider it fetches
        let isFetches = false;
        const arg1Keys = Object.getOwnPropertyNames(arg1);
        for (const name of this.outputNames) {
          if (arg1Keys.indexOf(name) !== -1) {
            const v = (arg1 as InferenceSession.NullableOnnxValueMapType)[name];
            if (v === null || v instanceof Tensor) {
              isFetches = true;
              isFetchesEmpty = false;
              fetches[name] = v;
            }
          }
        }

        if (isFetches) {
          if (typeof arg2 === 'object' && arg2 !== null) {
            options = arg2;
          } else if (typeof arg2 !== 'undefined') {
            throw new TypeError('\'options\' must be an object.');
          }
        } else {
          options = arg1 as RunOptions;
        }
      }
    } else if (typeof arg1 !== 'undefined') {
      throw new TypeError('Unexpected argument[1]: must be \'fetches\' or \'options\'.');
    }

    // check if all inputs are in feed
    for (const name of this.inputNames) {
      if (typeof feeds[name] === 'undefined') {
        throw new Error(`input '${name}' is missing in 'feeds'.`);
      }
    }

    // if no fetches is specified, we use the full output names list
    if (isFetchesEmpty) {
      for (const name of this.outputNames) {
        fetches[name] = null;
      }
    }

    // feeds, fetches and options are prepared

    // promise start here
    //
    //
    const results = await this.#handler.run(feeds, fetches, options);
    const returnValue: {[name: string]: OnnxValue} = {};
    for (const key in results) {
      returnValue[key] = new Tensor(results[key].type, results[key].data, results[key].dims);
    }
    return returnValue;
  }

  static create(path: string, options?: SessionOptions): Promise<InferenceSession>;
  static create(buffer: ArrayBufferLike, options?: SessionOptions): Promise<InferenceSession>;
  static create(buffer: ArrayBufferLike, byteOffset: number, byteLength?: number, options?: SessionOptions):
      Promise<InferenceSession>;
  static create(buffer: Uint8Array, options?: SessionOptions): Promise<InferenceSession>;
  static async create(
      arg0: string|ArrayBufferLike|Uint8Array, arg1?: SessionOptions|number, arg2?: number,
      arg3?: SessionOptions): Promise<InferenceSession> {
    // either load from a file or buffer
    let loadFromFilePath = false;
    let filePath: string;
    let buffer: ArrayBufferLike;
    let byteOffset = -1;
    let byteLength = -1;
    let uint8Array: Uint8Array;
    let options: SessionOptions = {};

    if (typeof arg0 === 'string') {
      loadFromFilePath = true;
      filePath = arg0;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
    } else if (arg0 instanceof Uint8Array) {
      uint8Array = arg0;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
    } else if (
        arg0 instanceof ArrayBuffer ||
        (typeof SharedArrayBuffer !== 'undefined' && arg0 instanceof SharedArrayBuffer)) {
      buffer = arg0;
      byteOffset = 0;
      byteLength = arg0.byteLength;
      if (typeof arg1 === 'object' && arg1 !== null) {
        options = arg1;
      } else if (typeof arg1 === 'number') {
        byteOffset = arg1;
        if (!Number.isSafeInteger(byteOffset)) {
          throw new RangeError('\'byteOffset\' must be an integer.');
        }
        if (byteOffset < 0 || byteOffset >= buffer.byteLength) {
          throw new RangeError(`'byteOffset' is out of range [0, ${buffer.byteLength}).`);
        }
        byteLength = arg0.byteLength - byteOffset;
        if (typeof arg2 === 'number') {
          byteLength = arg2;
          if (!Number.isSafeInteger(byteLength)) {
            throw new RangeError('\'byteLength\' must be an integer.');
          }
          if (byteLength <= 0 || byteOffset + byteLength > buffer.byteLength) {
            throw new RangeError(`'byteLength' is out of range (0, ${buffer.byteLength - byteOffset}].`);
          }
          if (typeof arg3 === 'object' && arg3 !== null) {
            options = arg3;
          } else if (typeof arg3 !== 'undefined') {
            throw new TypeError('\'options\' must be an object.');
          }
        } else if (typeof arg2 !== 'undefined') {
          throw new TypeError('\'byteLength\' must be a number.');
        }
      } else if (typeof arg1 !== 'undefined') {
        throw new TypeError('\'options\' must be an object.');
      }
      uint8Array = new Uint8Array(buffer, byteOffset, byteLength);
    } else {
      throw new TypeError('Unexpected argument[0]: must be \'path\' or \'buffer\'.');
    }

    // promise start here
    //
    //
    const backend = await resolveBackend(options);
    const handler = await (
        loadFromFilePath ? backend.createSessionHandler(filePath!, options) :
                           backend.createSessionHandler(uint8Array!, options));
    return new InferenceSession(handler);
  }

  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];

  #handler: SessionHandler;
}
