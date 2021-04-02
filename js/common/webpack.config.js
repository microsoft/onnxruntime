const path = require('path');
const webpack = require('webpack');

function buildConfig({
  suffix = '',
  format = 'umd',
  target = 'ES2017',
  devtool = 'source-map'
}) {
  return {
    entry: path.resolve(__dirname, 'lib/index.ts'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: `ort-common${suffix}.js`,
      libraryTarget: format,
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
    mode: 'production',
    devtool: devtool,
  };
}

module.exports = (env, argv) => {
  return [
    buildConfig({ format: 'umd', suffix: '.min', target: 'es5' }),
    buildConfig({ format: 'commonjs', suffix: '.node', target: 'es5' }),
  ];
};
