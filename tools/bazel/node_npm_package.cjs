// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');

const [destination, addon, runtime, bridge, platform, arch] = process.argv.slice(2);
const output = path.resolve(destination, 'package');
const version = fs.readFileSync('VERSION_NUMBER', 'utf8').trim();
const upstream = JSON.parse(fs.readFileSync('js/node/package.json', 'utf8'));
if (upstream.version !== version) {
  throw new Error(`Node package version ${upstream.version} must match VERSION_NUMBER (${version}).`);
}
const commonOutput = path.join(output, 'node_modules/onnxruntime-common');
fs.mkdirSync(commonOutput, {recursive: true});
const converted = ts.convertCompilerOptionsFromJson(
    JSON.parse(fs.readFileSync('js/tsconfig.json', 'utf8')).compilerOptions, process.cwd());
const formatHost = {
  getCurrentDirectory: () => process.cwd(),
  getCanonicalFileName: (name) => name,
  getNewLine: () => '\n',
};
function check(diagnostics) {
  const errors = diagnostics.filter((item) => item.category === ts.DiagnosticCategory.Error);
  if (errors.length) {
    throw new Error(ts.formatDiagnosticsWithColorAndContext(errors, formatHost));
  }
}
check(converted.errors);
function compile(source, target, extra = {}) {
  const options = {
    ...converted.options,
    incremental: false,
    declaration: true,
    declarationMap: false,
    sourceMap: false,
    noEmitOnError: true,
    rootDir: path.resolve(source),
    outDir: target,
    types: ['node'],
    typeRoots: [path.resolve('bazel/node/node_modules/@types')],
    ...extra,
  };
  const sources = fs.readdirSync(source).filter((name) => name.endsWith('.ts'))
      .map((name) => path.resolve(source, name));
  const program = ts.createProgram(sources, options);
  check(ts.getPreEmitDiagnostics(program));
  const result = program.emit();
  check(result.diagnostics);
  if (result.emitSkipped) {
    throw new Error(`TypeScript skipped production output for ${source}; inspect compiler diagnostics.`);
  }
}
compile('js/common/lib', commonOutput);
fs.writeFileSync(path.join(commonOutput, 'package.json'), JSON.stringify({
  name: 'onnxruntime-common', version, main: 'index.js', types: 'index.d.ts', license: 'MIT',
}));
compile('js/node/lib', path.join(output, 'dist'), {
  baseUrl: process.cwd(),
  paths: {'onnxruntime-common': [path.join(commonOutput, 'index.d.ts')]},
});
const nativeDirectory = path.join(output, 'bin/napi-v6', platform, arch);
fs.mkdirSync(nativeDirectory, {recursive: true});
for (const file of [addon, runtime, bridge]) {
  fs.copyFileSync(file, path.join(nativeDirectory, path.basename(file)));
}
fs.writeFileSync(path.join(output, 'package.json'), JSON.stringify({
  name: 'onnxruntime-node',
  version,
  description: 'Experimental source-built single-platform CPU ONNX Runtime Node binding.',
  license: 'MIT',
  main: 'dist/index.js',
  types: 'dist/index.d.ts',
  os: [platform],
  cpu: [arch],
  engines: {node: '>=20'},
  dependencies: {'onnxruntime-common': version},
  bundledDependencies: ['onnxruntime-common'],
}));
for (const name of ['LICENSE', 'ThirdPartyNotices.txt']) {
  fs.copyFileSync(name, path.join(output, name));
}
fs.copyFileSync('LICENSE', path.join(commonOutput, 'LICENSE'));
