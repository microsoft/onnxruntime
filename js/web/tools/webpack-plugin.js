// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const fs = require('node:fs');
const path = require('node:path');

/**
 * Webpack plugin for ONNX Runtime Web
 *
 * This plugin is used for consuming the OnnxRuntime Web library in a webpack project.
 *
 * The plugin will:
 * - Add WebAssembly files (`node_modules/onnxruntime-web/dist/ort-*.{js|wasm}`) into the webpack externals list to
 * prevent webpack from bundling them
 * - Enable dynamic import in the output environment
 * - Copy the WebAssembly files to the output folder
 *
 *
 * @example
 * ```js
 * // webpack.config.js
 * const OnnxRuntimeWebWebpackPlugin = require('onnxruntime-web/tools/webpack-plugin');
 *
 * module.exports = () => {
 *  return {
 *    target: ["web"],
 *    // ...
 *    plugins: [
 *      new OnnxRuntimeWebWebpackPlugin(),
 *    ]
 *  };
 * ```
 */
class OnnxRuntimeWebWebpackPlugin {
  /**
   * @typedef OnnxRuntimeWebWebpackPluginOptions
   * @type {object}
   * @property {string} [wasmPathOverride] - an optional path to override the default path of the WebAssembly files in
   * the output folder. Must be a relative path.
   */

  /**
   * @param {OnnxRuntimeWebWebpackPluginOptions} options - options for customizing the plugin
   */
  constructor(options) {
    this.options = options || {};
    if (this.options.wasmPathOverride) {
      // check if the wasmPathOverride is a valid relative path
      if (typeof this.options.wasmPathOverride !== 'string') {
        throw new Error('wasmPathOverride must be a string');
      }
      if (path.isAbsolute(this.options.wasmPathOverride)) {
        throw new Error('wasmPathOverride must be a relative path');
      }
    }
  }

  apply(compiler) {
    const TAP_TAG = 'OnnxRuntimeWebWebpackPlugin';
    const DIST_FOLDER = path.resolve(__dirname, '../dist');
    const WASM_FILES = /^(\.\/)?ort-.+\.(js|wasm)$/;

    // Let webpack ignore all "import()" calls. We want to preserve them as-is.
    compiler.hooks.normalModuleFactory.tap(TAP_TAG, (factory) => {
      const tapImportCall = (parser) => {
        parser.hooks.importCall.tap(TAP_TAG, () => {
          if (parser.state.current.context === DIST_FOLDER) {
            // if the context is "<...>/node_modules/onnxruntime-web/dist", ignore the import call.
            return false;
          }
        });
      };
      factory.hooks.parser.for('javascript/auto').tap(TAP_TAG, tapImportCall);
      factory.hooks.parser.for('javascript/dynamic').tap(TAP_TAG, tapImportCall);
      factory.hooks.parser.for('javascript/esm').tap(TAP_TAG, tapImportCall);
    });

    // Let webpack copy the WebAssembly/JS files to the output folder
    compiler.hooks.thisCompilation.tap(TAP_TAG, (compilation) => {
      compilation.hooks.processAssets.tapAsync(
          {
            name: TAP_TAG,
            stage: compiler.webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONAL,
          },
          async (_, callback) => {
            try {
              fs.readdirSync(DIST_FOLDER).forEach((file) => {
                if (WASM_FILES.test(file)) {
                  const sourceFilename = path.resolve(DIST_FOLDER, file);
                  const info = {
                    copied: true,
                    sourceFilename,
                  };
                  const source = new compiler.webpack.sources.RawSource(fs.readFileSync(sourceFilename));
                  const targetFile = this.options.wasmPathOverride ?
                      path.normalize(path.join(this.options.wasmPathOverride, file)).replace(/\\/g, '/') :
                      file;
                  const existingAsset = compilation.getAsset(targetFile);
                  if (existingAsset) {
                    compilation.updateAsset(targetFile, source, info);
                  } else {
                    compilation.emitAsset(targetFile, source, info);
                  }
                }
              });
              callback();
            } catch (error) {
              callback(error);
            }
          });
    });
  }
}

module.exports = OnnxRuntimeWebWebpackPlugin;
