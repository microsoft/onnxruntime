const webpack = require("webpack");
const path = require('path');
const fs = require('fs');
const APP_DIR = path.resolve(__dirname, "./src/");
const DIST_DIR = path.resolve(__dirname, "./dist/");

if (!fs.existsSync('dist')){
  fs.mkdirSync('dist');
}
fs.createReadStream('../../dist/onnx-wasm.wasm').pipe(fs.createWriteStream('dist/onnx-wasm.wasm'));
fs.createReadStream('../../dist/onnx-worker.js').pipe(fs.createWriteStream('dist/onnx-worker.js'));

module.exports = (env, argv) => {
  const config = {
    entry: APP_DIR + "/index.js",
    output : {
      path : DIST_DIR,
      filename: "main.js"
    },
    node: {fs: 'empty'},
    resolve: {
      extensions: ['.js']
    }
  };
  if (argv.mode === 'production') {
    config.mode = 'production';
    config.devtool = 'source-map';
  } else {
    config.mode = 'development';
    config.devtool = 'eval-source-map';
  }
  return config;
};