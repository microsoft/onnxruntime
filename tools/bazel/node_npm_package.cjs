// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');

const [kind, destination, addon, runtime, bridge, platform, arch, flavor, ...providerLibraries] =
    process.argv.slice(2);
if (!['common', 'node'].includes(kind)) {
  throw new Error(`Unknown npm package kind ${kind}; select common or node.`);
}
const output = path.resolve(destination, 'package');
const version = fs.readFileSync('VERSION_NUMBER', 'utf8').trim();
const upstreamPath = kind === 'common' ? 'js/common/package.json' : 'js/node/package.json';
const upstream = JSON.parse(fs.readFileSync(upstreamPath, 'utf8'));
const commonUpstream = JSON.parse(fs.readFileSync('js/common/package.json', 'utf8'));
if (upstream.version !== version) {
  throw new Error(`${upstream.name} version ${upstream.version} must match VERSION_NUMBER (${version}).`);
}
if (commonUpstream.version !== version) {
  throw new Error(`onnxruntime-common version ${commonUpstream.version} must match VERSION_NUMBER (${version}).`);
}
fs.mkdirSync(output, {recursive: true});
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
function copyNotices(target) {
  for (const name of ['LICENSE', 'ThirdPartyNotices.txt']) {
    fs.copyFileSync(name, path.join(target, name));
  }
}
function writeCommonPackage(target) {
  const cjs = path.join(target, 'dist/cjs');
  const esm = path.join(target, 'dist/esm');
  compile('js/common/lib', cjs, {module: ts.ModuleKind.CommonJS, moduleResolution: ts.ModuleResolutionKind.Node10});
  compile('js/common/lib', esm, {module: ts.ModuleKind.Node16, moduleResolution: ts.ModuleResolutionKind.Node16});
  fs.writeFileSync(path.join(cjs, 'package.json'), '{"type":"commonjs"}\n');
  fs.writeFileSync(path.join(target, 'package.json'), JSON.stringify({
    name: commonUpstream.name,
    version,
    description: commonUpstream.description,
    license: commonUpstream.license,
    author: commonUpstream.author,
    repository: commonUpstream.repository,
    keywords: commonUpstream.keywords,
    type: 'module',
    main: 'dist/cjs/index.js',
    types: 'dist/cjs/index.d.ts',
    exports: {
      '.': {
        types: './dist/cjs/index.d.ts',
        require: './dist/cjs/index.js',
        import: './dist/esm/index.js',
      },
    },
    engines: {node: '>=18'},
  }, null, 2) + '\n');
  fs.copyFileSync('js/common/README.md', path.join(target, 'README.md'));
  copyNotices(target);
}

if (kind === 'common') {
  writeCommonPackage(output);
  process.exit(0);
}
for (const value of [addon, runtime, bridge, platform, arch, flavor]) {
  if (!value) {
    throw new Error(
        'Node packaging requires addon, runtime, provider bridge, platform, architecture, and flavor arguments.');
  }
}
if (!['cpu', 'cuda'].includes(flavor)) {
  throw new Error(`Unknown Node package flavor ${flavor}; select cpu or cuda.`);
}
if ((flavor === 'cuda') !== (providerLibraries.length === 1)) {
  throw new Error('CUDA Node packaging requires exactly one explicit CUDA provider library; CPU packaging requires none.');
}
const commonOutput = path.join(output, 'node_modules/onnxruntime-common');
writeCommonPackage(commonOutput);
compile('js/node/lib', path.join(output, 'dist'), {
  baseUrl: process.cwd(),
  paths: {'onnxruntime-common': [path.join(commonOutput, 'dist/cjs/index.d.ts')]},
});
const nativeDirectory = path.join(output, 'bin/napi-v6', platform, arch);
fs.mkdirSync(nativeDirectory, {recursive: true});
for (const file of [addon, runtime, bridge, ...providerLibraries]) {
  fs.copyFileSync(file, path.join(nativeDirectory, path.basename(file)));
}
fs.writeFileSync(path.join(output, 'package.json'), JSON.stringify({
  name: upstream.name,
  version,
  description: `Experimental source-built single-platform ${flavor.toUpperCase()} ONNX Runtime Node binding.`,
  license: upstream.license,
  author: upstream.author,
  repository: upstream.repository,
  keywords: upstream.keywords,
  binary: upstream.binary,
  main: 'dist/index.js',
  types: 'dist/index.d.ts',
  os: [platform],
  cpu: [arch],
  engines: {node: '>=20'},
  dependencies: {'onnxruntime-common': version},
  bundledDependencies: ['onnxruntime-common'],
  onnxruntimeBazelExecutionProviders: flavor === 'cuda' ? ['cpu', 'cuda'] : ['cpu'],
}, null, 2) + '\n');
fs.writeFileSync(path.join(output, 'README.md'), [
  '# onnxruntime-node native Bazel package',
  '',
  `This archive contains the source-built ${platform}-${arch} ${flavor.toUpperCase()} runtime and Node N-API v6 binding.`,
  'It intentionally contains no downloader, postinstall script, training API, or other platform payload.',
  '',
].join('\n'));
copyNotices(output);
