const path = require('path');
const webpack = require('webpack');

function buildAllConfig({
  suffix = '',
  format = 'umd',
  target = 'ES2017',
  mode = 'production',
  devtool = 'source-map'
}) {
  return {
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: `ort${suffix}.js`,
      library: {
        name: 'ort',
        type: format
      }
    },
    externals: {
      'fs': 'fs',
      'path': 'path',
    },
    resolve: { extensions: ['.ts', '.js'] },
    plugins: [new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] })],
    module: {
      rules: [{
        test: /\.tsx?$/,
        use: [
          {
            loader: 'ts-loader',
            options: {
              compilerOptions: { target: target }
            }
          }
        ]
      }]
    },
    mode: mode,
    devtool: devtool,
  };
}

function buildConfig({
  suffix = '',
  format = 'umd',
  target = 'ES2017',
  mode = 'production',
  devtool = 'source-map'
}) {
  return {
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: `ort-web${suffix}.js`,
      library: {
        type: format
      }
    },
    externals: {
      'onnxruntime-common': 'ort',
      'fs': 'fs',
      'path': 'path',
    },
    resolve: { extensions: ['.ts', '.js'] },
    plugins: [new webpack.WatchIgnorePlugin({ paths: [/\.js$/, /\.d\.ts$/] })],
    module: {
      rules: [{
        test: /\.tsx?$/,
        use: [
          {
            loader: 'ts-loader',
            options: {
              compilerOptions: { target: target }
            }
          }
        ]
      }]
    },
    mode: mode,
    devtool: devtool,
  };
}

module.exports = (env, argv) => {
  return [
    buildAllConfig({ format: 'umd', mode: 'development', devtool: 'inline-source-map', target: 'es5' }),
    buildAllConfig({ format: 'umd', suffix: '.min', target: 'es5' }),
    buildConfig({ format: 'umd', mode: 'development', devtool: 'inline-source-map', target: 'es5' }),
    buildConfig({ format: 'umd', suffix: '.min', target: 'es5' }),
    //buildConfig({ format: 'commonjs', suffix: '.node', target: 'es5' }),
  ];
};
