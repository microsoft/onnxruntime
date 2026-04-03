// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { InferenceSession as InferenceSessionImpl } from './inference-session-impl.js';
import { OnnxModelOptions } from './onnx-model.js';
import { OnnxValue, OnnxValueDataLocation } from './onnx-value.js';
import type { Tensor } from './tensor.js';
import { TryGetGlobalType } from './type-helper.js';

/* eslint-disable @typescript-eslint/no-redeclare */

export declare namespace InferenceSession {
  // #region input/output types

  type OnnxValueMapType = { readonly [name: string]: OnnxValue };
  type NullableOnnxValueMapType = { readonly [name: string]: OnnxValue | null };

  /**
   * A feeds (model inputs) is an object that uses input names as keys and OnnxValue as corresponding values.
   */
  type FeedsType = OnnxValueMapType;

  /**
   * A fetches (model outputs) could be one of the following:
   *
   * - Omitted. Use model's output names definition.
   * - An array of string indicating the output names.
   * - An object that use output names as keys and OnnxValue or null as corresponding values.
   *
   * @remark
   * different from input argument, in output, OnnxValue is optional. If an OnnxValue is present it will be
   * used as a pre-allocated value by the inference engine; if omitted, inference engine will allocate buffer
   * internally.
   */
  type FetchesType = readonly string[] | NullableOnnxValueMapType;

  /**
   * A inferencing return type is an object that uses output names as keys and OnnxValue as corresponding values.
   */
  type ReturnType = OnnxValueMapType;

  // #endregion

  // #region session options

  /**
   * A set of configurations for session behavior.
   */
  export interface SessionOptions extends OnnxModelOptions {
    /**
     * An array of execution provider options.
     *
     * An execution provider option can be a string indicating the name of the execution provider,
     * or an object of corresponding type.
     */
    executionProviders?: readonly ExecutionProviderConfig[];

    /**
     * The intra OP threads number.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native).
     */
    intraOpNumThreads?: number;

    /**
     * The inter OP threads number.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native).
     */
    interOpNumThreads?: number;

    /**
     * The free dimension override.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    freeDimensionOverrides?: { readonly [dimensionName: string]: number };

    /**
     * The optimization level.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'layout' | 'all';

    /**
     * Whether enable CPU memory arena.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    enableCpuMemArena?: boolean;

    /**
     * Whether enable memory pattern.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    enableMemPattern?: boolean;

    /**
     * Execution mode.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    executionMode?: 'sequential' | 'parallel';

    /**
     * Optimized model file path.
     *
     * If this setting is specified, the optimized model will be dumped. In browser, a blob will be created
     * with a pop-up window.
     */
    optimizedModelFilePath?: string;

    /**
     * Whether enable profiling.
     *
     * This setting is a placeholder for a future use.
     */
    enableProfiling?: boolean;

    /**
     * File prefix for profiling.
     *
     * This setting is a placeholder for a future use.
     */
    profileFilePrefix?: string;

    /**
     * Log ID.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logId?: string;

    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logSeverityLevel?: 0 | 1 | 2 | 3 | 4;

    /**
     * Log verbosity level.
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    logVerbosityLevel?: number;

    /**
     * Specify string as a preferred data location for all outputs, or an object that use output names as keys and a
     * preferred data location as corresponding values.
     *
     * This setting is available only in ONNXRuntime Web for WebGL and WebGPU EP.
     */
    preferredOutputLocation?: OnnxValueDataLocation | { readonly [outputName: string]: OnnxValueDataLocation };

    /**
     * Whether enable graph capture.
     * This setting is available only in ONNXRuntime Web for WebGPU EP.
     */
    enableGraphCapture?: boolean;

    /**
     * Store configurations for a session. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/
     * onnxruntime_session_options_config_keys.h
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     *
     * @example
     * ```js
     * extra: {
     *   session: {
     *     set_denormal_as_zero: "1",
     *     disable_prepacking: "1"
     *   },
     *   optimization: {
     *     enable_gelu_approximation: "1"
     *   }
     * }
     * ```
     */
    extra?: Record<string, unknown>;
  }

  // #region execution providers

  // Currently, we have the following backends to support execution providers:
  // Backend Node.js binding: supports 'cpu', 'dml' (win32), 'coreml' (macOS) and 'cuda' (linux).
  // Backend WebAssembly: supports 'cpu', 'wasm', 'webgpu' and 'webnn'.
  // Backend ONNX.js: supports 'webgl'.
  // Backend React Native: supports 'cpu', 'xnnpack', 'coreml' (iOS), 'nnapi' (Android).
  interface ExecutionProviderOptionMap {
    coreml: CoreMLExecutionProviderOption;
    cpu: CpuExecutionProviderOption;
    cuda: CudaExecutionProviderOption;
    dml: DmlExecutionProviderOption;
    nnapi: NnapiExecutionProviderOption;
    tensorrt: TensorRtExecutionProviderOption;
    wasm: WebAssemblyExecutionProviderOption;
    webgl: WebGLExecutionProviderOption;
    webgpu: WebGpuExecutionProviderOption;
    webnn: WebNNExecutionProviderOption;
    qnn: QnnExecutionProviderOption;
    xnnpack: XnnpackExecutionProviderOption;
  }

  type ExecutionProviderName = keyof ExecutionProviderOptionMap;
  type ExecutionProviderConfig =
    | ExecutionProviderOptionMap[ExecutionProviderName]
    | ExecutionProviderOption
    | ExecutionProviderName
    | string;

  export interface ExecutionProviderOption {
    readonly name: string;
    /**
     * Additional provider-specific options as key-value pairs.
     * This allows passing custom configuration to execution providers.
     */
    [key: string]: unknown;
  }
  export interface CpuExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'cpu';
    useArena?: boolean;
  }
  export interface CudaExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'cuda';
    deviceId?: number;
    /** GPU memory limit for CUDA (BFC Arena) */
    gpuMemLimit?: number;
    /** BFC Arena extension strategy */
    arenaExtendStrategy?: 'kNextPowerOfTwo' | 'kSameAsRequested';
    /** CUDNN convolution algorithm search */
    cudnnConvAlgoSearch?: 'EXHAUSTIVE' | 'HEURISTIC' | 'DEFAULT';
    /** Flag specifying if copying can use the default stream */
    doCopyInDefaultStream?: boolean;
    /** Flag specifying if maximum workspace can be used in CUDNN convolution algorithm search */
    cudnnConvUseMaxWorkspace?: boolean;
    /** Flag specifying if the CUDA graph is to be captured for the model */
    enableCudaGraph?: boolean;
    /** Flag specifying if TunableOp is enabled */
    tunableOpEnable?: boolean;
    /** Flag specifying if TunableOp tuning is enabled */
    tunableOpTuningEnable?: boolean;
    /** Max tuning duration time limit for TunableOp (milliseconds) */
    tunableOpMaxTuningDurationMs?: number;
    /** Flag specifying if SkipLayerNorm is in strict mode */
    enableSkipLayerNormStrictMode?: boolean;
    /** Make the CUDA EP NHWC preferred */
    preferNhwc?: boolean;
    /** Flag specifying if EP level unified stream is used */
    useEpLevelUnifiedStream?: boolean;
    /** Use TF32 */
    useTf32?: boolean;
  }
  export interface DmlExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'dml';
    deviceId?: number;
  }
  export interface TensorRtExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'tensorrt';
    deviceId?: number;
    /** Maximum iterations for TensorRT parser to get capability */
    trtMaxPartitionIterations?: number;
    /** Minimum size of TensorRT subgraphs */
    trtMinSubgraphSize?: number;
    /** Maximum workspace size for TensorRT (0 means max device memory size) */
    trtMaxWorkspaceSize?: number;
    /** Enable TensorRT FP16 precision */
    trtFp16Enable?: boolean;
    /** Enable TensorRT BF16 precision */
    trtBf16Enable?: boolean;
    /** Enable TensorRT INT8 precision */
    trtInt8Enable?: boolean;
    /** TensorRT INT8 calibration table name */
    trtInt8CalibrationTableName?: string;
    /** Use native TensorRT generated calibration table */
    trtInt8UseNativeCalibrationTable?: boolean;
    /** Enable DLA */
    trtDlaEnable?: boolean;
    /** DLA core number */
    trtDlaCore?: number;
    /** Dump TRT subgraph */
    trtDumpSubgraphs?: boolean;
    /** Enable engine caching */
    trtEngineCacheEnable?: boolean;
    /** Specify engine cache path */
    trtEngineCachePath?: string;
    /** Enable engine decryption */
    trtEngineDecryptionEnable?: boolean;
    /** Specify engine decryption library path */
    trtEngineDecryptionLibPath?: string;
    /** Force building TensorRT engine sequentially */
    trtForceSequentialEngineBuild?: boolean;
    /** Enable context memory sharing between subgraphs */
    trtContextMemorySharingEnable?: boolean;
    /** Force Pow + Reduce ops in layer norm to FP32 */
    trtLayerNormFp32Fallback?: boolean;
    /** Enable TensorRT timing cache */
    trtTimingCacheEnable?: boolean;
    /** Specify timing cache path */
    trtTimingCachePath?: string;
    /** Force the TensorRT cache to be used even if device profile does not match */
    trtForceTimingCache?: boolean;
    /** Enable detailed build step logging on TensorRT EP with timing for each engine build */
    trtDetailedBuildLog?: boolean;
    /** Build engine using heuristics to reduce build time */
    trtBuildHeuristicsEnable?: boolean;
    /** Control if sparsity can be used by TRT */
    trtSparsityEnable?: boolean;
    /** Set the builder optimization level (0-5, default 3) */
    trtBuilderOptimizationLevel?: number;
    /** Set maximum number of auxiliary streams per inference stream (-1 = heuristics) */
    trtAuxiliaryStreams?: number;
    /** Specify the tactics to be used by adding (+) or removing (-) tactics from the default */
    trtTacticSources?: string;
    /** Specify extra TensorRT plugin library paths */
    trtExtraPluginLibPaths?: string;
    /** Specify the range of the input shapes to build the engine with (min shapes) */
    trtProfileMinShapes?: string;
    /** Specify the range of the input shapes to build the engine with (max shapes) */
    trtProfileMaxShapes?: string;
    /** Specify the range of the input shapes to build the engine with (optimal shapes) */
    trtProfileOptShapes?: string;
    /** Enable CUDA graph in ORT TRT */
    trtCudaGraphEnable?: boolean;
    /** Specify the preview features to be enabled */
    trtPreviewFeatures?: string;
    /** Dump EP context node model */
    trtDumpEpContextModel?: boolean;
    /** Specify file name to dump EP context node model */
    trtEpContextFilePath?: string;
    /** Specify EP context embed mode (0 = engine cache path, 1 = engine binary data) */
    trtEpContextEmbedMode?: number;
    /** Enable weight-stripped engine build */
    trtWeightStrippedEngineEnable?: boolean;
    /** Folder path for the ONNX model containing the weights (for weight-stripped engines) */
    trtOnnxModelFolderPath?: string;
    /** Specify engine cache prefix */
    trtEngineCachePrefix?: string;
    /** Enable hardware compatibility */
    trtEngineHwCompatible?: boolean;
    /** Exclude specific ops from running on TRT */
    trtOpTypesToExclude?: string;
    /** Save initializers locally instead of to disk */
    trtLoadUserInitializer?: boolean;
  }
  export interface WebAssemblyExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'wasm';
  }
  export interface WebGLExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'webgl';
    // TODO: add flags
  }
  export interface XnnpackExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'xnnpack';
  }
  export interface WebGpuExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'webgpu';

    /**
     * Specify the preferred layout when running layout sensitive operators.
     *
     * @default 'NCHW'
     */
    preferredLayout?: 'NCHW' | 'NHWC';

    /**
     * Specify a list of node names that should be executed on CPU even when WebGPU EP is used.
     */
    forceCpuNodeNames?: readonly string[];

    /**
     * Specify the validation mode for WebGPU execution provider.
     * - 'disabled': Disable all validation.
     * When used in Node.js, disable validation may cause process crash if WebGPU errors occur. Be cautious when using
     * this mode.
     * When used in web, this mode is equivalent to 'wgpuOnly'.
     * - 'wgpuOnly': Perform WebGPU internal validation only.
     * - 'basic': Perform basic validation including WebGPU internal validation. This is the default mode.
     * - 'full': Perform full validation. This mode may have performance impact. Use it for debugging purpose.
     *
     * @default 'basic'
     */
    validationMode?: 'disabled' | 'wgpuOnly' | 'basic' | 'full';

    /**
     * Specify an optional WebGPU device to be used by the WebGPU execution provider.
     */
    device?: TryGetGlobalType<'GPUDevice'>;
  }

  // #region WebNN options

  interface WebNNExecutionProviderName extends ExecutionProviderOption {
    readonly name: 'webnn';
  }

  /**
   * Represents a set of options for creating a WebNN MLContext.
   *
   * @see https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions
   */
  export interface WebNNContextOptions {
    deviceType?: 'cpu' | 'gpu' | 'npu';
    numThreads?: number;
    powerPreference?: 'default' | 'low-power' | 'high-performance';
  }

  /**
   * Represents a set of options for WebNN execution provider without MLContext.
   */
  export interface WebNNOptionsWithoutMLContext extends WebNNExecutionProviderName, WebNNContextOptions {
    context?: never;
  }

  /**
   * Represents a set of options for WebNN execution provider with MLContext.
   *
   * When MLContext is provided, the deviceType is also required so that the WebNN EP can determine the preferred
   * channel layout.
   *
   * @see https://www.w3.org/TR/webnn/#dom-ml-createcontext
   */
  export interface WebNNOptionsWithMLContext
    extends WebNNExecutionProviderName,
      Omit<WebNNContextOptions, 'deviceType'>,
      Required<Pick<WebNNContextOptions, 'deviceType'>> {
    context: TryGetGlobalType<'MLContext'>;
  }

  /**
   * Represents a set of options for WebNN execution provider with MLContext which is created from GPUDevice.
   *
   * @see https://www.w3.org/TR/webnn/#dom-ml-createcontext-gpudevice
   */
  export interface WebNNOptionsWebGpu extends WebNNExecutionProviderName {
    context: TryGetGlobalType<'MLContext'>;
    gpuDevice: TryGetGlobalType<'GPUDevice'>;
  }

  /**
   * Options for WebNN execution provider.
   */
  export type WebNNExecutionProviderOption =
    | WebNNOptionsWithoutMLContext
    | WebNNOptionsWithMLContext
    | WebNNOptionsWebGpu;

  // #endregion

  export interface QnnExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'qnn';
    /**
     * Specify the QNN backend type. E.g., 'cpu' or 'htp'.
     * Mutually exclusive with `backendPath`.
     *
     * @default 'htp'
     */
    backendType?: string;
    /**
     * Specify a path to the QNN backend library.
     * Mutually exclusive with `backendType`.
     */
    backendPath?: string;
    /**
     * Specify whether to enable HTP FP16 precision.
     *
     * @default true
     */
    enableFp16Precision?: boolean;
  }
  export interface CoreMLExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'coreml';
    /**
     * The bit flags for CoreML execution provider.
     *
     * ```
     * COREML_FLAG_USE_CPU_ONLY = 0x001
     * COREML_FLAG_ENABLE_ON_SUBGRAPH = 0x002
     * COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004
     * COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008
     * COREML_FLAG_CREATE_MLPROGRAM = 0x010
     * COREML_FLAG_USE_CPU_AND_GPU = 0x020
     * ```
     *
     * See include/onnxruntime/core/providers/coreml/coreml_provider_factory.h for more details.
     *
     * This flag is available only in ONNXRuntime (Node.js binding).
     */
    coreMlFlags?: number;
    /**
     * Specify whether to use CPU only in CoreML EP.
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    useCPUOnly?: boolean;
    useCPUAndGPU?: boolean;
    /**
     * Specify whether to enable CoreML EP on subgraph.
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    enableOnSubgraph?: boolean;
    /**
     * Specify whether to only enable CoreML EP for Apple devices with ANE (Apple Neural Engine).
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    onlyEnableDeviceWithANE?: boolean;
  }
  export interface NnapiExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'nnapi';
    useFP16?: boolean;
    useNCHW?: boolean;
    cpuDisabled?: boolean;
    cpuOnly?: boolean;
  }
  // #endregion

  // #endregion

  // #region run options

  /**
   * A set of configurations for inference run behavior
   */
  export interface RunOptions {
    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logSeverityLevel?: 0 | 1 | 2 | 3 | 4;

    /**
     * Log verbosity level.
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    logVerbosityLevel?: number;

    /**
     * Terminate all incomplete OrtRun calls as soon as possible if true
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    terminate?: boolean;

    /**
     * A tag for the Run() calls using this
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    tag?: string;

    /**
     * Set a single run configuration entry. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/
     * onnxruntime_run_options_config_keys.h
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     *
     * @example
     *
     * ```js
     * extra: {
     *   memory: {
     *     enable_memory_arena_shrinkage: "1",
     *   }
     * }
     * ```
     */
    extra?: Record<string, unknown>;
  }

  // #endregion

  // #region value metadata

  /**
   * The common part of the value metadata type for both tensor and non-tensor values.
   */
  export interface ValueMetadataBase {
    /**
     * The name of the specified input or output.
     */
    readonly name: string;
  }

  /**
   * Represents the metadata of a non-tensor value.
   */
  export interface NonTensorValueMetadata extends ValueMetadataBase {
    /**
     * Get a value indicating whether the value is a tensor.
     */
    readonly isTensor: false;
  }

  /**
   * Represents the metadata of a tensor value.
   */
  export interface TensorValueMetadata extends ValueMetadataBase {
    /**
     * Get a value indicating whether the value is a tensor.
     */
    readonly isTensor: true;
    /**
     * Get the data type of the tensor.
     */
    readonly type: Tensor.Type;
    /**
     * Get the shape of the tensor.
     *
     * If the shape is not defined, the value will an empty array. Otherwise, it will be an array representing the shape
     * of the tensor. Each element in the array can be a number or a string. If the element is a number, it represents
     * the corresponding dimension size. If the element is a string, it represents a symbolic dimension.
     */
    readonly shape: ReadonlyArray<number | string>;
  }

  /**
   * Represents the metadata of a value.
   */
  export type ValueMetadata = NonTensorValueMetadata | TensorValueMetadata;

  // #endregion
}

/**
 * Represent a runtime instance of an ONNX model.
 */
export interface InferenceSession {
  // #region run()

  /**
   * Execute the model asynchronously with the given feeds and options.
   *
   * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param options - Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

  /**
   * Execute the model asynchronously with the given feeds, fetches and options.
   *
   * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param fetches - Representation of the model output. See type description of `InferenceSession.OutputType` for
   * detail.
   * @param options - Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(
    feeds: InferenceSession.FeedsType,
    fetches: InferenceSession.FetchesType,
    options?: InferenceSession.RunOptions,
  ): Promise<InferenceSession.ReturnType>;

  // #endregion

  // #region release()

  /**
   * Release the inference session and the underlying resources.
   */
  release(): Promise<void>;

  // #endregion

  // #region profiling

  /**
   * Start profiling.
   */
  startProfiling(): void;

  /**
   * End profiling.
   */
  endProfiling(): void;

  // #endregion

  // #region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: readonly string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: readonly string[];

  /**
   * Get input metadata of the loaded model.
   */
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];

  /**
   * Get output metadata of the loaded model.
   */
  readonly outputMetadata: readonly InferenceSession.ValueMetadata[];

  // #endregion
}

export interface InferenceSessionFactory {
  // #region create()

  /**
   * Create a new inference session and load model asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(uri: string, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from segment of an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param byteOffset - The beginning of the specified portion of the array buffer.
   * @param byteLength - The length in bytes of the array buffer.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
    buffer: ArrayBufferLike,
    byteOffset: number,
    byteLength?: number,
    options?: InferenceSession.SessionOptions,
  ): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from a Uint8Array.
   *
   * @param buffer - A Uint8Array representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const InferenceSession: InferenceSessionFactory = InferenceSessionImpl;
