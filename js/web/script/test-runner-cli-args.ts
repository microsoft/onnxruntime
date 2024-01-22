// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import minimist from 'minimist';
import npmlog from 'npmlog';
import {Env, InferenceSession} from 'onnxruntime-common';

import {Logger} from '../lib/onnxjs/instrument';
import {Test} from '../test/test-types';

/* eslint-disable max-len */
const HELP_MESSAGE = `
test-runner-cli

Run ONNX Runtime Web tests, models, benchmarks in different environments.

Usage:
 test-runner-cli <mode> ... [options]

Modes:
 suite0                        Run all unittests, all operator tests and node model tests that described in suite test list
 suite1                        Run all operator tests and node model tests that described in suite test list
 model                         Run a single model test
 unittest                      Run all unittests
 op                            Run a single operator test

Options:

*** General Options ***

 -h, --help                    Print this message.
 -d, --debug                   Specify to run test runner in debug mode.
                                 Debug mode outputs verbose log for test runner, sets up environment debug flag, and keeps karma not to exit after tests completed.
 -b=<...>, --backend=<...>     Specify one or more backend(s) to run the test upon.
                                 Backends can be one or more of the following, splitted by comma:
                                   webgl
                                   webgpu
                                   wasm
                                   webnn
 -e=<...>, --env=<...>         Specify the environment to run the test. Should be one of the following:
                                 chrome     (default)
                                 edge       (Windows only)
                                 firefox
                                 electron
                                 safari     (MacOS only)
                                 node
                                 bs         (for BrowserStack tests)
 -p, --profile                 Enable profiler.
                                 Profiler will generate extra logs which include the information of events time consumption
 -P[=<...>], --perf[=<...>]    Generate performance number. Cannot be used with flag --debug.
                                 This flag can be used with a number as value, specifying the total count of test cases to run. The test cases may be used multiple times. Default value is 10.
 -c, --file-cache              Enable file cache.
 -i=<...>, --io-binding=<...>  Specify the IO binding testing type. Should be one of the following:
                                 none          (default)
                                 gpu-tensor      use pre-allocated GPU tensors for inputs and outputs
                                 gpu-location    use pre-allocated GPU tensors for inputs and set preferredOutputLocation to 'gpu-buffer'

*** Session Options ***
 -u=<...>, --optimized-model-file-path=<...>        Specify whether to dump the optimized model.
 -o=<...>, --graph-optimization-level=<...>    Specify graph optimization level.
                                                 Default is 'all'. Valid values are 'disabled', 'basic', 'extended', 'all'.
*** Logging Options ***

 --log-verbose=<...>           Set log level to verbose
 --log-info=<...>              Set log level to info
 --log-warning=<...>           Set log level to warning
 --log-error=<...>             Set log level to error
                                 The 4 flags above specify the logging configuration. Each flag allows to specify one or more category(s), splitted by comma. If use the flags without value, the log level will be applied to all category.

*** Backend Options ***

 -x, --wasm-number-threads     Set the WebAssembly number of threads
 --wasm-init-timeout           Set the timeout for WebAssembly backend initialization, in milliseconds
 --wasm-enable-simd            Set whether to enable SIMD
 --wasm-enable-proxy           Set whether to enable proxy worker
 --webgl-context-id            Set the WebGL context ID (webgl/webgl2)
 --webgl-matmul-max-batch-size Set the WebGL matmulMaxBatchSize
 --webgl-texture-cache-mode    Set the WebGL texture cache mode (initializerOnly/full)
 --webgl-texture-pack-mode     Set the WebGL texture pack mode (true/false)
 --webgpu-profiling-mode       Set the WebGPU profiling mode (off/default)
 --webnn-device-type           Set the WebNN device type (cpu/gpu)

*** Browser Options ***

 --no-sandbox                  This flag will be passed to Chrome.
                                 Sometimes Chrome need this flag to work together with Karma.
 --chromium-flags=<...>        This flag will be passed to Chrome and Edge browsers. Can be used multiple times.

Examples:

 Run all suite0 tests:
 > test-runner-cli suite0

 Run single model test (test_relu) on WebAssembly backend
 > test-runner-cli model test_relu --backend=wasm

 Debug unittest
 > test-runner-cli unittest --debug

 Debug operator matmul, highlight verbose log from BaseGlContext and WebGLBackend
 > test-runner-cli op matmul --backend=webgl --debug --log-verbose=BaseGlContext,WebGLBackend

 Profile an ONNX model on WebGL backend
 > test-runner-cli model <model_folder> --profile --backend=webgl

 Run perf testing of an ONNX model on WebGL backend
 > test-runner-cli model <model_folder> -b=webgl -P
 `;
/* eslint-enable max-len */

export declare namespace TestRunnerCliArgs {
  type Mode = 'suite0'|'suite1'|'model'|'unittest'|'op';
  type Backend = 'cpu'|'webgl'|'webgpu'|'wasm'|'onnxruntime'|'webnn';
  type Environment = 'chrome'|'edge'|'firefox'|'electron'|'safari'|'node'|'bs';
  type BundleMode = 'dev'|'perf';
  type IOBindingMode = 'none'|'gpu-tensor'|'gpu-location';
}

export interface TestRunnerCliArgs {
  debug: boolean;
  mode: TestRunnerCliArgs.Mode;
  /**
   * The parameter that used when in mode 'model' or 'op', specifying the search string for the model or op test
   */
  param?: string;
  backends: [TestRunnerCliArgs.Backend];
  env: TestRunnerCliArgs.Environment;

  /**
   * Bundle Mode
   *
   * this field affects the behavior of Karma and build script.
   *
   * Mode "perf":
   *   - use "dist/ort.all.min.js" as main file
   *   - use "test/ort.test.min.js" as test file
   * Mode "dev":
   *   - use "dist/ort.all.js" as main file
   *   - use "test/ort.test.js" as test file
   */
  bundleMode: TestRunnerCliArgs.BundleMode;

  ioBindingMode: TestRunnerCliArgs.IOBindingMode;

  logConfig: Test.Config['log'];

  /**
   * Whether to enable InferenceSession's profiler
   */
  profile: boolean;

  /**
   * Whether to enable file cache
   */
  fileCache: boolean;

  /**
   * Specify the times that test cases to run
   */
  times?: number;

  /**
   * whether to dump the optimized model
   */
  optimizedModelFilePath?: string;

  /**
   * Specify graph optimization level
   */
  graphOptimizationLevel: 'disabled'|'basic'|'extended'|'all';

  cpuOptions?: InferenceSession.CpuExecutionProviderOption;
  cudaOptions?: InferenceSession.CudaExecutionProviderOption;
  cudaFlags?: Record<string, unknown>;
  wasmOptions?: InferenceSession.WebAssemblyExecutionProviderOption;
  webglOptions?: InferenceSession.WebGLExecutionProviderOption;
  webnnOptions?: InferenceSession.WebNNExecutionProviderOption;
  globalEnvFlags?: Test.Options['globalEnvFlags'];
  noSandbox?: boolean;
  chromiumFlags: string[];
}


function parseBooleanArg(arg: unknown, defaultValue: boolean): boolean;
function parseBooleanArg(arg: unknown): boolean|undefined;
function parseBooleanArg(arg: unknown, defaultValue?: boolean): boolean|undefined {
  if (typeof arg === 'undefined') {
    return defaultValue;
  }

  if (typeof arg === 'boolean') {
    return arg;
  }

  if (typeof arg === 'number') {
    return arg !== 0;
  }

  if (typeof arg === 'string') {
    if (arg.toLowerCase() === 'true') {
      return true;
    }
    if (arg.toLowerCase() === 'false') {
      return false;
    }
  }

  throw new TypeError(`invalid boolean arg: ${arg}`);
}

function parseLogLevel<T>(arg: T) {
  let v: string[]|boolean;
  if (typeof arg === 'string') {
    v = arg.split(',');
  } else if (Array.isArray(arg)) {
    v = [];
    for (const e of arg) {
      v.push(...e.split(','));
    }
  } else {
    v = arg ? true : false;
  }
  return v;
}

function parseLogConfig(args: minimist.ParsedArgs) {
  const config: Array<{category: string; config: Logger.Config}> = [];
  const verbose = parseLogLevel(args['log-verbose']);
  const info = parseLogLevel(args['log-info']);
  const warning = parseLogLevel(args['log-warning']);
  const error = parseLogLevel(args['log-error']);

  if (typeof error === 'boolean' && error) {
    config.push({category: '*', config: {minimalSeverity: 'error'}});
  } else if (typeof warning === 'boolean' && warning) {
    config.push({category: '*', config: {minimalSeverity: 'warning'}});
  } else if (typeof info === 'boolean' && info) {
    config.push({category: '*', config: {minimalSeverity: 'info'}});
  } else if (typeof verbose === 'boolean' && verbose) {
    config.push({category: '*', config: {minimalSeverity: 'verbose'}});
  }

  if (Array.isArray(error)) {
    config.push(...error.map(i => ({category: i, config: {minimalSeverity: 'error' as Logger.Severity}})));
  }
  if (Array.isArray(warning)) {
    config.push(...warning.map(i => ({category: i, config: {minimalSeverity: 'warning' as Logger.Severity}})));
  }
  if (Array.isArray(info)) {
    config.push(...info.map(i => ({category: i, config: {minimalSeverity: 'info' as Logger.Severity}})));
  }
  if (Array.isArray(verbose)) {
    config.push(...verbose.map(i => ({category: i, config: {minimalSeverity: 'verbose' as Logger.Severity}})));
  }

  return config;
}

function parseCpuOptions(_args: minimist.ParsedArgs): InferenceSession.CpuExecutionProviderOption {
  return {name: 'cpu'};
}

function parseCpuFlags(_args: minimist.ParsedArgs): Record<string, unknown> {
  return {};
}

function parseWasmOptions(_args: minimist.ParsedArgs): InferenceSession.WebAssemblyExecutionProviderOption {
  return {name: 'wasm'};
}

function parseWasmFlags(args: minimist.ParsedArgs): Env.WebAssemblyFlags {
  const numThreads = args.x || args['wasm-number-threads'];
  if (typeof numThreads !== 'undefined' && typeof numThreads !== 'number') {
    throw new Error('Flag "x"/"wasm-number-threads" must be a number value');
  }
  const initTimeout = args['wasm-init-timeout'];
  if (typeof initTimeout !== 'undefined' && typeof initTimeout !== 'number') {
    throw new Error('Flag "wasm-init-timeout" must be a number value');
  }
  let simd = args['wasm-enable-simd'];
  if (simd === 'true') {
    simd = true;
  } else if (simd === 'false') {
    simd = false;
  } else if (typeof simd !== 'undefined' && typeof simd !== 'boolean') {
    throw new Error('Flag "wasm-enable-simd" must be a boolean value');
  }
  let proxy = args['wasm-enable-proxy'];
  if (proxy === 'true') {
    proxy = true;
  } else if (proxy === 'false') {
    proxy = false;
  } else if (typeof proxy !== 'undefined' && typeof proxy !== 'boolean') {
    throw new Error('Flag "wasm-enable-proxy" must be a boolean value');
  }
  return {numThreads, initTimeout, simd, proxy};
}

function parseWebglOptions(_args: minimist.ParsedArgs): InferenceSession.WebGLExecutionProviderOption {
  return {name: 'webgl'};
}

function parseWebglFlags(args: minimist.ParsedArgs): Partial<Env.WebGLFlags> {
  const contextId = args['webgl-context-id'];
  if (contextId !== undefined && contextId !== 'webgl' && contextId !== 'webgl2') {
    throw new Error('Flag "webgl-context-id" is invalid');
  }
  const matmulMaxBatchSize = args['webgl-matmul-max-batch-size'];
  if (matmulMaxBatchSize !== undefined && typeof matmulMaxBatchSize !== 'number') {
    throw new Error('Flag "webgl-matmul-max-batch-size" must be a number value');
  }
  const textureCacheMode = args['webgl-texture-cache-mode'];
  if (textureCacheMode !== undefined && textureCacheMode !== 'initializerOnly' && textureCacheMode !== 'full') {
    throw new Error('Flag "webgl-texture-cache-mode" is invalid');
  }
  const pack = args['webgl-texture-pack-mode'];
  if (pack !== undefined && typeof pack !== 'boolean') {
    throw new Error('Flag "webgl-texture-pack-mode" is invalid');
  }
  const async = args['webgl-async'];
  if (async !== undefined && typeof async !== 'boolean') {
    throw new Error('Flag "webgl-async" is invalid');
  }
  return {contextId, matmulMaxBatchSize, textureCacheMode, pack};
}

function parseWebgpuFlags(args: minimist.ParsedArgs): Partial<Env.WebGpuFlags> {
  const profilingMode = args['webgpu-profiling-mode'];
  if (profilingMode !== undefined && profilingMode !== 'off' && profilingMode !== 'default') {
    throw new Error('Flag "webgpu-profiling-mode" is invalid');
  }
  const validateInputContent = args['webgpu-validate-input-content'];
  if (validateInputContent !== undefined && typeof validateInputContent !== 'boolean') {
    throw new Error('Flag "webgpu-validate-input-content" is invalid');
  }
  return {profilingMode, validateInputContent};
}

function parseWebNNOptions(args: minimist.ParsedArgs): InferenceSession.WebNNExecutionProviderOption {
  const deviceType = args['webnn-device-type'];
  if (deviceType !== undefined && deviceType !== 'cpu' && deviceType !== 'gpu') {
    throw new Error('Flag "webnn-device-type" is invalid');
  }
  return {name: 'webnn', deviceType};
}

function parseGlobalEnvFlags(args: minimist.ParsedArgs): NonNullable<TestRunnerCliArgs['globalEnvFlags']> {
  const wasm = parseWasmFlags(args);
  const webgl = parseWebglFlags(args);
  const webgpu = parseWebgpuFlags(args);
  const cpuFlags = parseCpuFlags(args);
  return {webgl, wasm, webgpu, ...cpuFlags};
}

export function parseTestRunnerCliArgs(cmdlineArgs: string[]): TestRunnerCliArgs {
  const args = minimist(cmdlineArgs);

  if (args.help || args.h) {
    console.log(HELP_MESSAGE);
    process.exit();
  }

  // Option: -d, --debug
  const debug = parseBooleanArg(args.debug || args.d, false);
  if (debug) {
    npmlog.level = 'verbose';
  }
  npmlog.verbose('TestRunnerCli.Init', 'Parsing commandline arguments...');

  const mode = args._.length === 0 ? 'suite0' : args._[0];

  // Option: -e=<...>, --env=<...>
  const envArg = args.env || args.e;
  const env = (typeof envArg !== 'string') ? 'chrome' : envArg;
  if (['chrome', 'edge', 'firefox', 'electron', 'safari', 'node', 'bs'].indexOf(env) === -1) {
    throw new Error(`not supported env ${env}`);
  }

  // Option: -b=<...>, --backend=<...>
  const browserBackends = ['webgl', 'webgpu', 'wasm', 'webnn'];

  // TODO: remove this when Chrome support WebNN.
  //       we need this for now because Chrome does not support webnn yet,
  //       and ChromeCanary is not in CI.

  const defaultBrowserBackends = ['webgl', 'webgpu', 'wasm' /*, 'webnn'*/];
  const nodejsBackends = ['cpu', 'wasm'];
  const backendArgs = args.backend || args.b;
  const backend = (typeof backendArgs !== 'string') ? (env === 'node' ? nodejsBackends : defaultBrowserBackends) :
                                                      backendArgs.split(',');
  for (const b of backend) {
    if ((env !== 'node' && browserBackends.indexOf(b) === -1) || (env === 'node' && nodejsBackends.indexOf(b) === -1)) {
      throw new Error(`backend ${b} is not supported in env ${env}`);
    }
  }

  const globalEnvFlags = parseGlobalEnvFlags(args);

  if (backend.includes('webnn') && !globalEnvFlags.wasm!.proxy) {
    throw new Error('Backend webnn requires flag "wasm-enable-proxy" to be set to true.');
  }

  // Options:
  // --log-verbose=<...>
  // --log-info=<...>
  // --log-warning=<...>
  // --log-error=<...>
  const logConfig = parseLogConfig(args);
  globalEnvFlags.logLevel = logConfig[0]?.config.minimalSeverity;
  // Option: -p, --profile
  const profile = (args.profile || args.p) ? true : false;
  if (profile) {
    logConfig.push({category: 'Profiler.session', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.node', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.op', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.backend', config: {minimalSeverity: 'verbose'}});
    globalEnvFlags.logLevel = 'verbose';
  }

  // Option: -P[=<...>], --perf[=<...>]
  const perfArg = (args.perf || args.P);
  const perf = perfArg ? true : false;
  const times = (typeof perfArg === 'number') ? perfArg : 10;
  if (debug && perf) {
    throw new Error('Flag "perf" cannot be used together with flag "debug".');
  }
  if (perf && (mode !== 'model')) {
    throw new Error('Flag "perf" can only be used in mode "model".');
  }
  if (perf) {
    logConfig.push({category: 'TestRunner.Perf', config: {minimalSeverity: 'verbose'}});
  }

  // Option: -i=<...>, --io-binding=<...>
  const ioBindingArg = args['io-binding'] || args.i;
  const ioBindingMode = (typeof ioBindingArg !== 'string') ? 'none' : ioBindingArg;
  if (['none', 'gpu-tensor', 'gpu-location'].indexOf(ioBindingMode) === -1) {
    throw new Error(`not supported io binding mode ${ioBindingMode}`);
  }

  // Option: -u, --optimized-model-file-path
  const optimizedModelFilePath = args['optimized-model-file-path'] || args.u || undefined;
  if (typeof optimizedModelFilePath !== 'undefined' && typeof optimizedModelFilePath !== 'string') {
    throw new Error('Flag "optimized-model-file-path" need to be either empty or a valid file path.');
  }

  // Option: -o, --graph-optimization-level
  const graphOptimizationLevel = args['graph-optimization-level'] || args.o || 'all';
  if (typeof graphOptimizationLevel !== 'string' ||
      ['disabled', 'basic', 'extended', 'all'].indexOf(graphOptimizationLevel) === -1) {
    throw new Error(`graph optimization level is invalid: ${graphOptimizationLevel}`);
  }

  // Option: -c, --file-cache
  const fileCache = parseBooleanArg(args['file-cache'] || args.c, false);

  const cpuOptions = parseCpuOptions(args);
  const wasmOptions = parseWasmOptions(args);

  const webglOptions = parseWebglOptions(args);
  const webnnOptions = parseWebNNOptions(args);

  // Option: --no-sandbox
  const noSandbox = !!args['no-sandbox'];

  // parse chromium flags
  let chromiumFlags = args['chromium-flags'];
  if (!chromiumFlags) {
    chromiumFlags = [];
  } else if (typeof chromiumFlags === 'string') {
    chromiumFlags = [chromiumFlags];
  } else if (!Array.isArray(chromiumFlags)) {
    throw new Error(`Invalid command line arg: --chromium-flags: ${chromiumFlags}`);
  }


  npmlog.verbose('TestRunnerCli.Init', ` Mode:              ${mode}`);
  npmlog.verbose('TestRunnerCli.Init', ` Env:               ${env}`);
  npmlog.verbose('TestRunnerCli.Init', ` Debug:             ${debug}`);
  npmlog.verbose('TestRunnerCli.Init', ` Backend:           ${backend}`);
  npmlog.verbose('TestRunnerCli.Init', ` IO Binding Mode:   ${ioBindingMode}`);
  npmlog.verbose('TestRunnerCli.Init', 'Parsing commandline arguments... DONE');

  return {
    debug,
    mode: mode as TestRunnerCliArgs['mode'],
    param: args._.length > 1 ? args._[1] : undefined,
    backends: backend as TestRunnerCliArgs['backends'],
    bundleMode: perf ? 'perf' : 'dev',
    env: env as TestRunnerCliArgs['env'],
    logConfig,
    profile,
    times: perf ? times : undefined,
    ioBindingMode: ioBindingMode as TestRunnerCliArgs['ioBindingMode'],
    optimizedModelFilePath,
    graphOptimizationLevel: graphOptimizationLevel as TestRunnerCliArgs['graphOptimizationLevel'],
    fileCache,
    cpuOptions,
    webglOptions,
    webnnOptions,
    wasmOptions,
    globalEnvFlags,
    noSandbox,
    chromiumFlags
  };
}
