const path = require('path');
const webpack = require('webpack');
const minimist = require('minimist');

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

function buildWebConfig({
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

function buildTestRunnerConfig({
  suffix = '',
  format = 'umd',
  target = 'es5',
  mode = 'production',
  devtool = 'source-map'
}) {
  return {
    entry: path.resolve(__dirname, 'test/test-runner2.ts'),
    output: {
      path: path.resolve(__dirname, 'test'),
      filename: `ort-test-runner${suffix}.js`,
      library: {
        type: format
      }
    },
    externals: {
      '..': 'ort',
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

module.exports = () => {
  const args = minimist(process.argv);
  const bundleMode = args['bundle-mode'] || 'prod';  // 'prod'|'dev'|'perf'|undefined;
  const builds = [];

  if (bundleMode === 'prod' || bundleMode === 'perf') {
    builds.push(
      buildAllConfig({ suffix: '.min', target: 'es5' }),
      buildWebConfig({ suffix: '.min', target: 'es5' })
    );
  } else { // dev
    builds.push(
      buildAllConfig({ mode: 'development', devtool: 'inline-source-map', target: 'es5' }),
      buildWebConfig({ mode: 'development', devtool: 'inline-source-map', target: 'es5' }),
    );
  }

  if (bundleMode === 'dev') {
    builds.push(buildTestRunnerConfig({ suffix: '.dev', mode: 'development', devtool: 'inline-source-map' }));
  } else if (bundleMode === 'perf') {
    builds.push(buildTestRunnerConfig({ suffix: '.perf' }));
  }

  return builds;
};
