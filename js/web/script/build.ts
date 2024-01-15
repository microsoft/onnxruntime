// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as esbuild from 'esbuild';
import minimist from 'minimist';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';

/**
 * @summary Build script for ort-web using esbuild.
 */

const args = minimist(process.argv.slice(2));
/**
 * --bundle-mode=prod (default)
 *   Build multiple ort-web bundles for production.
 *
 * --bundle-mode=dev
 *   Build a single ort-web bundle for development, and a test bundle.
 *
 * --bundle-mode=perf
 *   Build a single ort-web bundle for performance test, and a test bundle.
 *
 * --bundle-mode=node
 *   Build a single ort-web bundle for nodejs.
 */
const BUNDLE_MODE: 'prod'|'dev'|'perf'|'node' = args['bundle-mode'] || 'prod';

/**
 * --debug
 *   Enable debug mode. In this mode, esbuild metafile feature will be enabled. Simple bundle analysis will be printed.
 *
 * --debug=verbose
 *   Enable debug mode. In this mode, esbuild metafile feature will be enabled. Detailed bundle analysis will be
 * printed.
 *
 * --debug=save
 *  Enable debug mode. In this mode, esbuild metafile feature will be enabled. Full bundle analysis will be saved to a
 * file as JSON.
 */
const DEBUG = args.debug;  // boolean|'verbose'|'save'

const SOURCE_ROOT_FOLDER = path.join(__dirname, '../..');  // <ORT_ROOT>/js/
const DEFAULT_DEFINE = {
  'BUILD_DEFS.DISABLE_WEBGL': 'false',
  'BUILD_DEFS.DISABLE_WEBGPU': 'false',
  'BUILD_DEFS.DISABLE_WEBNN': 'false',
  'BUILD_DEFS.DISABLE_WASM': 'false',
  'BUILD_DEFS.DISABLE_WASM_PROXY': 'false',
  'BUILD_DEFS.DISABLE_WASM_THREAD': 'false',
  'BUILD_DEFS.DISABLE_TRAINING': 'true',
};

const COPYRIGHT_HEADER = `/*!
 * ONNX Runtime Web v${require('../package.json').version}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

interface OrtBuildOptions {
  isProduction?: boolean;
  isNode?: boolean;
  format: 'iife'|'cjs'|'esm';
  outputBundleName: string;
  define?: Record<string, string>;
}

async function buildBundle(options: esbuild.BuildOptions) {
  const result = await esbuild.build({
    logLevel: DEBUG ? (DEBUG === 'verbose' || DEBUG === 'save' ? 'verbose' : 'debug') : 'info',
    metafile: !!DEBUG,
    absWorkingDir: SOURCE_ROOT_FOLDER,
    bundle: true,
    banner: {js: COPYRIGHT_HEADER},
    ...options
  });
  if (DEBUG) {
    if (DEBUG === 'save') {
      await fs.writeFile(
          `${path.basename(options.outfile!)}.esbuild.metafile.json`, JSON.stringify(result.metafile!, null, 2));
    } else {
      console.log(await esbuild.analyzeMetafile(result.metafile!, {verbose: DEBUG === 'verbose'}));
    }
  }
  return result;
}

async function minifyCode(sourceCode: string): Promise<string> {
  const result = await esbuild.transform(sourceCode, {
    minify: true,
    legalComments: 'none',
  });
  return result.code;
}

async function buildOrt({
  isProduction = false,
  isNode = false,
  format,
  outputBundleName,
  define = DEFAULT_DEFINE,
}: OrtBuildOptions) {
  // #region Plugin: resolve ignore imports

  /**
   * This plugin is used to ignore a few nodejs imports that are not used in the browser. Those imported functions are
   * not really used in the browser because they are usually put behind a feature check. However, esbuild does not know
   * that. It will complain about those imports are not available in the browser.
   *
   * This plugin will ignore those imports and replace them with empty exports.
   */
  const excludeNodejsImports = {
    name: 'exclude-nodejs-imports',
    setup(build: esbuild.PluginBuild) {
      build.onResolve({filter: /(^node:|^worker_threads$|^fs$|^path$|^perf_hooks$|^os$)/}, args => ({
                                                                                             namespace: 'nodejs-ignore',
                                                                                             path: args.path,
                                                                                             sideEffects: false,
                                                                                           }));
      build.onLoad({filter: /.*/, namespace: 'nodejs-ignore'}, args => {
        switch (args.path) {
          case 'node:fs/promises':
          case 'node:fs':
          case 'fs':
            return {
              contents: 'export const readFile = undefined;' +
                  'export const readFileSync = undefined;' +
                  'export const createReadStream = undefined;'
            };
          case 'node:os':
          case 'os':
            return {contents: 'export const cpus = undefined;'};
          case 'node:path':
          case 'path':
            return {contents: 'export const join = undefined;'};
          default:
            return {contents: ''};
        }
      });
    },
  };
  // #endregion

  // #region Plugin: web assembly multi-thread worker loader

  /**
   * This plugin is used to load web assembly multi-thread worker code as string.
   *
   * This allows to create the worker from a Blob, so we don't need to create a separate file for the worker.
   */
  const wasmThreadedHandler = {
    name: 'wasm-threaded-handler',
    setup(build: esbuild.PluginBuild) {
      build.onLoad({filter: /[\\/]ort-wasm-threaded\.worker\.js$/}, async args => {
        let contents = await fs.readFile(args.path, {encoding: 'utf-8'});
        if (isProduction) {
          contents = await minifyCode(contents);
        }
        return {loader: 'text', contents};
      });
    },
  };
  // #endregion

  // #region Plugin: generated emscripten .js loader

  /**
   * This plugin is used to patch the generated emscripten .js file for multi-thread build.
   *
   * Since we use inline worker for multi-thread, we make an optimization to use function.toString() to get the
   * implementation of the exported `ortWasmThreaded` function to reduce the size of the bundle. However, the generated
   * function uses a variable `_scriptDir` which is defined inside an IIFE closure. When we use function.toString(), the
   * worker code will throw "_scriptDir is not defined" error.
   *
   * To fix this error, we need to patch the generated code to replace access to `_scriptDir` with `typeof _scriptDir
   * !== "undefined" && _scriptDir`.
   */
  const emscriptenThreadedJsHandler = {
    name: 'emscripten-threaded-js-handler',
    setup(build: esbuild.PluginBuild) {
      build.onLoad({filter: /ort-wasm.*-threaded.*\.js$/}, async args => {
        let contents = await fs.readFile(args.path, {encoding: 'utf-8'});
        // For debug build, Emscripten generates the following code:
        //
        // if (_scriptDir) {
        //   scriptDirectory = _scriptDir;
        // }
        //
        // We replace it with:
        //
        // if (typeof _scriptDir !== "undefined" && _scriptDir) {
        //   scriptDirectory = _scriptDir;
        // }
        contents = contents.replace('if (_scriptDir) {', 'if (typeof _scriptDir !== "undefined" && _scriptDir) {');

        // For release build, Emscripten generates the following code:
        //
        // ...,_scriptDir&&(H=_scriptDir),...
        // We replace it with:
        // ...,(typeof _scriptDir !== "undefined" && _scriptDir)&&(H=_scriptDir),...
        contents =
            contents.replace(/_scriptDir(&&\(.+=_scriptDir\))/, '(typeof _scriptDir !== "undefined" && _scriptDir)$1');

        return {contents};
      });
    }
  };
  // #endregion

  // #region Plugin: proxy worker loader

  /**
   * This plugin is used to load proxy worker code as string.
   */
  const proxyWorkerHandler = {
    name: 'proxy-worker-handler',
    setup(build: esbuild.PluginBuild) {
      build.onResolve(
          {filter: /proxy-worker\/main$/},
          async args => ({path: args.path, namespace: 'proxy-worker', pluginData: args.resolveDir}));

      build.onLoad({filter: /.*/, namespace: 'proxy-worker'}, async args => {
        const result = await buildBundle({
          entryPoints: [path.resolve(args.pluginData, args.path)],
          outfile: `web/dist/${outputBundleName}.proxy.js`,
          platform: 'browser',
          plugins: [excludeNodejsImports, wasmThreadedHandler, emscriptenThreadedJsHandler],
          define: {
            ...build.initialOptions.define,
            'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
          },
          sourcemap: isProduction ? false : 'inline',
          minify: isProduction,
          write: false,
        });

        return {loader: 'text', contents: result.outputFiles![0].text};
      });
    },
  };
  // #endregion

  await buildBundle({
    entryPoints: ['web/lib/index.ts'],
    outfile: `web/dist/${outputBundleName}.js`,
    platform: isNode ? 'node' : 'browser',
    format,
    globalName: 'ort',
    plugins: isNode ? undefined :
                      [excludeNodejsImports, proxyWorkerHandler, wasmThreadedHandler, emscriptenThreadedJsHandler],
    external: isNode ? ['onnxruntime-common'] : undefined,
    define,
    sourcemap: isProduction ? 'linked' : 'inline',
    minify: isProduction,
  });
}

async function buildTest() {
  const isProduction = BUNDLE_MODE === 'perf';

  await buildBundle({
    absWorkingDir: path.join(SOURCE_ROOT_FOLDER, 'web/test'),

    entryPoints: ['test-main.ts'],
    outfile: isProduction ? 'ort.test.min.js' : 'ort.test.js',
    platform: 'browser',
    format: 'iife',
    define: DEFAULT_DEFINE,
    sourcemap: isProduction ? false : 'inline',
    sourceRoot: path.join(SOURCE_ROOT_FOLDER, 'web/test'),
    external: ['../../node'],
    plugins: [
      // polyfill nodejs modules
      require('esbuild-plugin-polyfill-node').polyfillNode({globals: false}),
      // make "ort" external
      {
        name: 'make-ort-external',
        setup(build: esbuild.PluginBuild) {
          build.onResolve(
              {filter: /^onnxruntime-common$/},
              _args => ({path: 'onnxruntime-common', namespace: 'make-ort-external'}));
          build.onLoad(
              {filter: /.*/, namespace: 'make-ort-external'},
              _args => ({contents: 'module.exports = globalThis.ort;'}));
        }
      }
    ],
    minify: isProduction,
  });
}

async function main() {
  // tasks for each esbuild bundle
  const buildTasks: Array<Promise<void>> = [];
  /**
   * add one build task
   */
  const addBuildTask = async (task: Promise<void>) => {
    if (DEBUG) {
      // in DEBUG mode, build sequentially
      await task;
    } else {
      buildTasks.push(task);
    }
  };
  /**
   * add all 6 build tasks for web bundles. Includes:
   * - IIFE, debug:                [name].js
   * - IIFE, production:           [name].min.js
   * - CJS, debug:                 cjs/[name].js
   * - CJS, production:            cjs/[name].min.js
   * - ESM, debug:                 esm/[name].js
   * - ESM, production:            esm/[name].min.js
   */
  const addAllWebBuildTasks = async (options: Omit<OrtBuildOptions, 'format'>) => {
    // [name].js
    await addBuildTask(buildOrt({
      ...options,
      format: 'iife',
    }));
    // [name].min.js
    await addBuildTask(buildOrt({
      ...options,
      outputBundleName: options.outputBundleName + '.min',
      format: 'iife',
      isProduction: true,
    }));
    // cjs/[name].js
    await addBuildTask(buildOrt({
      ...options,
      outputBundleName: 'cjs/' + options.outputBundleName,
      format: 'cjs',
    }));
    // cjs/[name].min.js
    await addBuildTask(buildOrt({
      ...options,
      outputBundleName: 'cjs/' + options.outputBundleName + '.min',
      format: 'cjs',
      isProduction: true,
    }));
    // esm/[name].js
    await addBuildTask(buildOrt({
      ...options,
      outputBundleName: 'esm/' + options.outputBundleName,
      format: 'esm',
    }));
    // esm/[name].min.js
    await addBuildTask(buildOrt({
      ...options,
      outputBundleName: 'esm/' + options.outputBundleName + '.min',
      format: 'esm',
      isProduction: true,
    }));
  };

  if (BUNDLE_MODE === 'node' || BUNDLE_MODE === 'prod') {
    // ort.node.min.js
    await addBuildTask(buildOrt({
      isProduction: true,
      isNode: true,
      format: 'cjs',
      outputBundleName: 'ort.node.min',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_WEBGPU': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WEBNN': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
        'BUILD_DEFS.DISABLE_WASM_THREAD': 'true',
      },
    }));
  }

  if (BUNDLE_MODE === 'dev') {
    // ort.all.js
    await addBuildTask(buildOrt({outputBundleName: 'ort.all', format: 'iife', define: {...DEFAULT_DEFINE}}));
  }

  if (BUNDLE_MODE === 'perf') {
    // ort.all.min.js
    await addBuildTask(buildOrt({
      isProduction: true,
      outputBundleName: 'ort.all.min',
      format: 'iife',
    }));
  }

  if (BUNDLE_MODE === 'prod') {
    // ort.all[.min].js
    await addAllWebBuildTasks({outputBundleName: 'ort.all'});

    // ort[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_WEBGPU': 'true'},
    });
    // ort.webgpu[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgpu',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_WEBGL': 'true', 'BUILD_DEFS.DISABLE_WEBNN': 'true'},
    });
    // ort.wasm[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_WEBGPU': 'true', 'BUILD_DEFS.DISABLE_WEBGL': 'true'},
    });
    // ort.webgl[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgl',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_WEBGPU': 'true',
        'BUILD_DEFS.DISABLE_WASM': 'true',
        'BUILD_DEFS.DISABLE_WEBNN': 'true',
      },
    });
    // ort.wasm-core[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm-core',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_WEBGPU': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WEBNN': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
        'BUILD_DEFS.DISABLE_WASM_THREAD': 'true',
      },
    });
    // ort.training.wasm[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.training.wasm',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_TRAINING': 'false',
        'BUILD_DEFS.DISABLE_WEBGPU': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WEBNN': 'true',
      },
    });
  }

  if (BUNDLE_MODE === 'dev' || BUNDLE_MODE === 'perf') {
    await addBuildTask(buildTest());
  }

  await Promise.all(buildTasks);

  if (BUNDLE_MODE === 'prod') {
    // generate package.json files under each of the dist folders for commonJS and ESModule
    // this trick allows typescript to import this package as different module type
    // see also: https://evertpot.com/universal-commonjs-esm-typescript-packages/
    await fs.writeFile(path.resolve(__dirname, '../dist/cjs', 'package.json'), '{"type": "commonjs"}');
    await fs.writeFile(path.resolve(__dirname, '../dist/esm', 'package.json'), '{"type": "module"}');
  }
}

void main();
