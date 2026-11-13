// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');

const output = path.resolve(process.argv[2]);
const files = [
  'unittests/lib/model-metadata',
  'unittests/lib/inference-session',
  'unittests/lib/tensor',
  'api/simple-api-tests',
];

function emit(source, destination) {
  const contents = fs.readFileSync(source, 'utf8');
  const result = ts.transpileModule(contents, {
    fileName: source,
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
    },
  });
  const errors = result.diagnostics.filter((d) => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) {
    throw new Error(ts.formatDiagnosticsWithColorAndContext(errors, {
      getCurrentDirectory: () => process.cwd(),
      getCanonicalFileName: (name) => name,
      getNewLine: () => '\n',
    }));
  }
  const target = path.join(output, destination);
  fs.mkdirSync(path.dirname(target), {recursive: true});
  fs.writeFileSync(target, result.outputText);
}

for (const name of fs.readdirSync('js/common/lib')) {
  if (name.endsWith('.ts') && !name.endsWith('.d.ts')) {
    emit(`js/common/lib/${name}`, `node_modules/onnxruntime-common/${name.replace(/\.ts$/, '.js')}`);
  }
}
fs.writeFileSync(path.join(output, 'node_modules/onnxruntime-common/package.json'),
    JSON.stringify({name: 'onnxruntime-common', main: 'index.js'}));
for (const name of fs.readdirSync('js/node/lib')) {
  if (name.endsWith('.ts')) {
    emit(`js/node/lib/${name}`, `js/node/dist/${name.replace(/\.ts$/, '.js')}`);
  }
}
fs.writeFileSync(path.join(output, 'js/node/package.json'),
    JSON.stringify({name: 'onnxruntime-node', main: 'dist/index.js'}));
for (const file of [...files, 'test-utils']) {
  emit(`js/node/test/${file}.ts`, `js/node/test/${file}.js`);
}
for (const source of [
  'js/node/test/testdata',
  'js/node/test/ort-schema/protobuf/onnx.js',
  'onnxruntime/test/testdata/onnx_backend_test_series_filters.jsonc',
  'onnxruntime/test/testdata/onnx_backend_test_series_overrides.jsonc',
]) {
  const target = path.join(output, source);
  fs.mkdirSync(path.dirname(target), {recursive: true});
  fs.cpSync(source, target, {recursive: true});
}
fs.writeFileSync(path.join(output, 'test-files.json'), JSON.stringify(files));
