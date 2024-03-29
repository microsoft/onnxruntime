// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as esbuild from 'esbuild';
import minimist from 'minimist';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import {SourceMapConsumer, SourceMapGenerator} from 'source-map';

console.time('BUILD');

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

/**
 * Root folder of the source code: `<ORT_ROOT>/js/`
 */
const SOURCE_ROOT_FOLDER = path.join(__dirname, '../..');

/**
 * Default define values for the build.
 */
const DEFAULT_DEFINE = {
  'BUILD_DEFS.DISABLE_WEBGL': 'false',
  'BUILD_DEFS.DISABLE_JSEP': 'false',
  'BUILD_DEFS.DISABLE_WASM': 'false',
  'BUILD_DEFS.DISABLE_WASM_PROXY': 'false',
  'BUILD_DEFS.DISABLE_WASM_THREAD': 'false',
  'BUILD_DEFS.DISABLE_TRAINING': 'true',

  // 'process': 'undefined',
  // 'typeof navigator': '"object"',
};

// const WEB_DEFINE = {
//   'process': 'undefined',
// };

const COPYRIGHT_HEADER = `/*!
 * ONNX Runtime Web v${require('../package.json').version}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

/**
 * A custom footer used to append to the end of IIFE bundle to make it compatible with CommonJS module system.
 *
 * ESBuild does not support UMD format (which is a combination of IIFE and CommonJS). We don't want to generate 2 build
 * targets (IIFE and CommonJS) because it will increase the package size. Instead, we generate IIFE and append this
 * footer to make it compatible with CommonJS module system.
 *
 * see also: https://github.com/evanw/esbuild/issues/507
 */
const COMMONJS_FOOTER = `
if (typeof exports === "object" && typeof module === "object") {
  module.exports = ort;
}
`;
const COMMONJS_FOOTER_MIN = 'typeof exports=="object"&&typeof module=="object"&&(module.exports=ort);';

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
    footer: options.format === 'iife' ? {js: options.minify ? COMMONJS_FOOTER_MIN : COMMONJS_FOOTER} : undefined,
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

// async function minifyCode(sourceCode: string): Promise<string> {
//   const result = await esbuild.transform(sourceCode, {
//     minify: true,
//     legalComments: 'none',
//   });
//   return result.code;
// }

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
          plugins: [excludeNodejsImports],
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

  // distribution code is split into multiple files:
  // - [bundle-name][.min].[m]js
  // - [bundle-name].proxy[.min].[m]js
  // - ort[-training]-wasm[-simd][-threaded][.jsep].js
  // - ort-wasm[-simd]-threaded[.jsep].worker.js
  const external = isNode ? ['onnxruntime-common'] : [
    'node:fs/promises',
    'node:fs',
    'node:os',
  ];
  const plugins: esbuild.Plugin[] = isNode ? [] : [/*excludeNodejsImports, */];
  //
  // TODO:
  if (process.env.YouDontKnow) {
    plugins.push(proxyWorkerHandler);
  }

  await buildBundle({
    entryPoints: ['web/lib/index.ts'],
    outfile: `web/dist/${outputBundleName}.${format === 'esm' ? 'mjs' : 'js'}`,
    platform: isNode ? 'node' : 'browser',
    format,
    globalName: 'ort',
    plugins,
    external,
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


/**
 * Perform the post-process step after ESBuild finishes the build.
 *
 * This is a custom post process step to insert magic comments to a specific import call:
 * ```
 * ... await import(` ...
 * ```
 * to:
 * ```
 * ... await import(/* webpackIgnore: true *\/` ...
 * ```
 *
 * Why we need this?
 *
 * If a project uses Webpack to bundle the code, Webpack will try to resolve the dynamic import calls. However, we don't
 * want Webpack to resolve the dynamic import calls inside the ort-web bundle because:
 *
 * - We want to keep the ort-*.js and ort-*.wasm as-is. This makes it able to replace the ort-*.js and ort-*.wasm with a
 * custom build if needed.
 * - The Emscripten generated code uses `require()` to load Node.js modules. Those code is guarded by a feature check to
 * make sure only run in Node.js. Webpack does not recognize the feature check and will try to resolve the `require()`
 * in browser environment. This will cause the Webpack build to fail.
 * - There are multiple entry points that use dynamic import to load the ort-*.js and ort-*.wasm. If the content of the
 * dynamic import is resolved by Webpack, it will be duplicated in the final bundle. This will increase the bundle size.
 *
 * What about other bundlers?
 *
 * TBD
 *
 */
async function postProcess() {
  const IMPORT_MAGIC_COMMENT = '/* webpackIgnore: true */';
  const IMPORT_ORIGINAL = 'await import(`';
  const IMPORT_NEW = `await import(${IMPORT_MAGIC_COMMENT}\``;

  const files = await fs.readdir(path.join(SOURCE_ROOT_FOLDER, 'web/dist'));
  for (const file of files) {
    // only process on "ort.*.min.js" and "ort.*.min.mjs" files.
    if ((file.endsWith('.min.js') || file.endsWith('.min.mjs')) && file.startsWith('ort.')) {
      const jsFilePath = path.join(SOURCE_ROOT_FOLDER, 'web/dist', file);
      const sourcemapFilePath = jsFilePath + '.map';

      const originalJsFileSize = (await fs.stat(jsFilePath)).size;

      if (!(await fs.stat(sourcemapFilePath)).isFile()) {
        continue;
      }

      const jsFileLines = (await fs.readFile(jsFilePath, 'utf-8')).split('\n');

      let line = -1, column = -1, found = false;
      for (let i = 0; i < jsFileLines.length; i++) {
        column = jsFileLines[i].indexOf(IMPORT_ORIGINAL);
        if (column !== -1) {
          if (found || column !== jsFileLines[i].lastIndexOf(IMPORT_ORIGINAL)) {
            throw new Error('Multiple dynamic import calls found. Should not happen.');
          }
          line = i + 1;
          jsFileLines[i] = jsFileLines[i].replace(IMPORT_ORIGINAL, IMPORT_NEW);
          found = true;
        }
      }
      if (!found) {
        if (file.includes('webgl')) {
          // skip webgl
          continue;
        }
        throw new Error('Dynamic import call not found. Should not happen.');
      }

      await SourceMapConsumer.with(await fs.readFile(sourcemapFilePath, 'utf-8'), null, async (consumer) => {
        consumer.eachMapping((mapping) => {
          if (mapping.generatedLine === line && mapping.generatedColumn >= column) {
            mapping.generatedColumn += IMPORT_MAGIC_COMMENT.length;
          }
        });

        await fs.writeFile(sourcemapFilePath, SourceMapGenerator.fromSourceMap(consumer).toString());
      });

      await fs.writeFile(jsFilePath, jsFileLines.join('\n'));
      const newJsFileSize = (await fs.stat(jsFilePath)).size;
      if (newJsFileSize - originalJsFileSize !== IMPORT_MAGIC_COMMENT.length) {
        console.log(`File: ${file}, Original size: ${originalJsFileSize}, New size: ${newJsFileSize}`);
        throw new Error(`Failed to insert magic comment to file "${file}".`);
      }
    }
  }
}

async function validate() {
  const files = await fs.readdir(path.join(SOURCE_ROOT_FOLDER, 'web/dist'));
  for (const file of files) {
    // validate on all "ort.*.min.js" and "ort.*.min.mjs" files.
    if ((file.endsWith('.min.js') || file.endsWith('.min.mjs')) && file.startsWith('ort.')) {
      const content = await fs.readFile(path.join(SOURCE_ROOT_FOLDER, 'web/dist', file), 'utf-8');

      // all files should not contain BUILD_DEFS definition. BUILD_DEFS should be defined in the build script only.
      //
      // If the final bundle contains BUILD_DEFS definition, it means the build script is not working correctly. In this
      // case, we should fix the build script (this file).
      //
      if (content.includes('BUILD_DEFS')) {
        throw new Error(`Validation failed: "${file}" contains BUILD_DEFS definition.`);
      }
    }
  }
}

async function main() {
  console.timeLog('BUILD', 'Start building ort-web bundles...');

  /**
   * add all 4 build tasks for web bundles. Includes:
   * - IIFE/CJS, debug:                [name].js
   * - IIFE/CJS, production:           [name].min.js
   * - ESM, debug:                     [name].mjs
   * - ESM, production:                [name].min.mjs
   */
  const addAllWebBuildTasks = async (options: Omit<OrtBuildOptions, 'format'>) => {
    // [name].js
    await buildOrt({
      ...options,
      format: 'iife',
    });
    // [name].min.js
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName + '.min',
      format: 'iife',
      isProduction: true,
    });
    // [name].mjs
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName,
      format: 'esm',
    });
    // [name].min.mjs
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName + '.min',
      format: 'esm',
      isProduction: true,
    });
  };

  if (BUNDLE_MODE === 'node' || BUNDLE_MODE === 'prod') {
    // ort.node.min.js
    await buildOrt({
      isProduction: true,
      isNode: true,
      format: 'cjs',
      outputBundleName: 'ort.node.min',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
        'BUILD_DEFS.DISABLE_WASM_THREAD': 'true',
      },
    });
  }

  if (BUNDLE_MODE === 'dev') {
    // ort.all.js
    await buildOrt({outputBundleName: 'ort.all', format: 'iife', define: {...DEFAULT_DEFINE}});
  }

  if (BUNDLE_MODE === 'perf') {
    // ort.all.min.js
    await buildOrt({
      isProduction: true,
      outputBundleName: 'ort.all.min',
      format: 'iife',
    });
  }

  if (BUNDLE_MODE === 'prod') {
    // ort.all[.min].[m]js
    await addAllWebBuildTasks({outputBundleName: 'ort.all'});

    // ort[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_JSEP': 'true'},
    });
    // ort.webgpu[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgpu',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_WEBGL': 'true'},
    });
    // ort.wasm[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_JSEP': 'true', 'BUILD_DEFS.DISABLE_WEBGL': 'true'},
    });
    // ort.webgl[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgl',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WASM': 'true',
      },
    });
    // ort.wasm-core[.min].js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm-core',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
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
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
      },
    });
  }

  if (BUNDLE_MODE === 'dev' || BUNDLE_MODE === 'perf') {
    await buildTest();
  }

  if (BUNDLE_MODE === 'prod') {
    console.timeLog('BUILD', 'Start post-processing...');
    await postProcess();

    console.timeLog('BUILD', 'Start validating...');
    await validate();
  }

  console.timeEnd('BUILD');
}

void main();
