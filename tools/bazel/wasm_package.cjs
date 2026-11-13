// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');

const [destination, loader, binary, variant] = process.argv.slice(2);
if (!['scalar', 'simd'].includes(variant)) {
  throw new Error(`Unknown WASM variant ${variant}; select scalar or simd.`);
}
const version = fs.readFileSync('VERSION_NUMBER', 'utf8').trim();
const upstream = JSON.parse(fs.readFileSync('js/web/package.json', 'utf8'));
if (upstream.version !== version) {
  throw new Error(`onnxruntime-web version ${upstream.version} must match VERSION_NUMBER (${version}).`);
}
const output = path.resolve(destination, 'package');
const dist = path.join(output, 'dist');
fs.mkdirSync(dist, {recursive: true});
fs.copyFileSync(loader, path.join(dist, 'ort-wasm.mjs'));
fs.copyFileSync(binary, path.join(dist, 'ort-wasm.wasm'));

const defines = {
  DISABLE_WEBGL: true,
  DISABLE_JSEP: true,
  DISABLE_WEBGPU: true,
  DISABLE_WEBNN: true,
  DISABLE_WASM: false,
  DISABLE_WASM_PROXY: true,
  ENABLE_JSPI: false,
  ENABLE_BUNDLE_WASM_JS: false,
  IS_ESM: false,
  ESM_IMPORT_META_URL: undefined,
  BUNDLE_FILENAME: 'ort.js',
};
const modules = [];
for (const directory of ['js/common/lib', 'js/web/lib/wasm', 'js/web/lib']) {
  for (const name of fs.readdirSync(directory)) {
    if (!name.endsWith('.ts') || name.endsWith('.d.ts')) continue;
    if (directory === 'js/web/lib' &&
        !['backend-wasm.ts', 'validate-session-options.ts', 'version.ts'].includes(name)) continue;
    const source = `${directory}/${name}`;
    const content = fs.readFileSync(source, 'utf8').replace(/\bBUILD_DEFS\.([A-Z_]+)/g, (_, key) => {
      if (!Object.hasOwn(defines, key)) {
        throw new Error(`Unconfigured frontend build flag ${key} in ${source}; update the native Bazel web configuration.`);
      }
      return JSON.stringify(defines[key]) ?? 'undefined';
    });
    const result = ts.transpileModule(content, {
      fileName: source.replace(/\.ts$/, '.cts'),
      reportDiagnostics: true,
      compilerOptions: {module: ts.ModuleKind.Node16, target: ts.ScriptTarget.ES2020, esModuleInterop: true},
    });
    const errors = result.diagnostics.filter((d) => d.category === ts.DiagnosticCategory.Error);
    if (errors.length) {
      throw new Error(ts.formatDiagnosticsWithColorAndContext(errors, {
        getCurrentDirectory: () => process.cwd(),
        getCanonicalFileName: (name) => name,
        getNewLine: () => '\n',
      }));
    }
    modules.push(`${JSON.stringify(source.slice(0, -3))}:function(require,module,exports){\n${result.outputText}\n}`);
  }
}
const entry = `
const common = load('js/common/lib/index');
const backend = load('js/web/lib/backend-wasm').wasmBackend;
const init = backend.init.bind(backend);
backend.init = async function(name) {
  if (common.env.wasm.numThreads !== 1 || common.env.wasm.proxy ||
      common.env.wasm.simd === 'relaxed') {
    throw new Error('This native Bazel WASM package supports numThreads=1, proxy=false and scalar/fixed SIMD only.');
  }
  await init(name);
};
common.env.wasm.numThreads = 1;
common.env.wasm.simd = ${variant === 'simd' ? "'fixed'" : 'false'};
common.env.wasm.proxy = false;
common.env.wasm.wasmPaths = {mjs: new URL('ort-wasm.mjs', baseUrl).href,
                            wasm: new URL('ort-wasm.wasm', baseUrl).href};
common.registerBackend('cpu', backend, 10);
common.registerBackend('wasm', backend, 10);
Object.defineProperty(common.env.versions, 'web', {
  value: load('js/web/lib/version').version, enumerable: true
});
return common;
`;
fs.writeFileSync(path.join(dist, 'ort.js'), `// Source-assembled by native Bazel; no prebuilt runtime.
(function(factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require, require('node:url').pathToFileURL(__filename).href);
  } else {
    globalThis.ort = factory(undefined, document.currentScript.src);
  }
})(function(nativeRequire, baseUrl) {
  const modules = {${modules.join(',\n')}};
  const cache = {};
  function resolve(request, parent) {
    if (request === 'onnxruntime-common') return 'js/common/lib/index';
    if (!request.startsWith('.')) return request;
    const parts = parent.split('/').slice(0, -1);
    for (const part of request.split('/')) {
      if (part === '..') parts.pop();
      else if (part !== '.') parts.push(part);
    }
    return parts.join('/').replace(/\\.js$/, '');
  }
  function load(id) {
    if (!modules[id]) {
      if (nativeRequire) return nativeRequire(id);
      throw new Error('Module ' + id + ' is unavailable in this CPU-only native Bazel web package.');
    }
    if (!cache[id]) {
      const module = {exports: {}};
      cache[id] = module;
      modules[id]((request) => load(resolve(request, id)), module, module.exports);
    }
    return cache[id].exports;
  }
  ${entry}
});
`);

const converted = ts.convertCompilerOptionsFromJson(
    JSON.parse(fs.readFileSync('js/tsconfig.json', 'utf8')).compilerOptions, process.cwd());
const typeRoot = path.join(output, 'types');
const typeSources = fs.readdirSync('js/common/lib').filter((name) => name.endsWith('.ts'))
    .map((name) => path.resolve('js/common/lib', name));
const typeProgram = ts.createProgram(typeSources, {
  ...converted.options,
  incremental: false,
  declaration: true,
  emitDeclarationOnly: true,
  declarationMap: false,
  noEmitOnError: true,
  rootDir: path.resolve('js/common/lib'),
  outDir: typeRoot,
  types: ['node'],
  typeRoots: [path.resolve('bazel/node/node_modules/@types')],
});
const typeResult = typeProgram.emit();
const typeErrors = [...converted.errors, ...ts.getPreEmitDiagnostics(typeProgram), ...typeResult.diagnostics]
    .filter((item) => item.category === ts.DiagnosticCategory.Error);
if (typeErrors.length || typeResult.emitSkipped) {
  throw new Error(ts.formatDiagnosticsWithColorAndContext(typeErrors, {
    getCurrentDirectory: () => process.cwd(),
    getCanonicalFileName: (name) => name,
    getNewLine: () => '\n',
  }) || 'Web API declaration emission was skipped; inspect the TypeScript compiler configuration.');
}
fs.writeFileSync(path.join(output, 'package.json'), JSON.stringify({
  name: upstream.name,
  version,
  description: `Source-built single-threaded CPU ONNX Runtime Web package (${variant}).`,
  license: upstream.license,
  author: upstream.author,
  repository: upstream.repository,
  keywords: upstream.keywords,
  main: './dist/ort.js',
  browser: './dist/ort.js',
  types: './types/index.d.ts',
  exports: {
    '.': {types: './types/index.d.ts', require: './dist/ort.js', default: './dist/ort.js'},
    './ort-wasm.mjs': './dist/ort-wasm.mjs',
    './ort-wasm.wasm': './dist/ort-wasm.wasm',
  },
  engines: {node: '>=18'},
  onnxruntimeBazelVariant: variant,
  onnxruntimeBazelFeatures: {threads: false, proxy: false, webgpu: false, webnn: false, jsep: false},
}, null, 2) + '\n');
fs.writeFileSync(path.join(output, 'README.md'), [
  '# onnxruntime-web native Bazel package',
  '',
  `This archive contains the source-built single-threaded CPU ${variant} WASM runtime.`,
  'It does not include threaded, proxy, WebGPU, WebNN, JSEP, or WebGL implementations.',
  '',
].join('\n'));
for (const name of ['LICENSE', 'ThirdPartyNotices.txt']) {
  fs.copyFileSync(name, path.join(output, name));
}
