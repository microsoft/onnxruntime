// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');
const converted = ts.convertCompilerOptionsFromJson(
    JSON.parse(fs.readFileSync('js/tsconfig.json', 'utf8')).compilerOptions, process.cwd());
const options = {
  ...converted.options,
  incremental: false,
  declaration: true,
  declarationMap: false,
  sourceMap: false,
  noEmitOnError: true,
  rootDir: path.resolve('js/common/lib'),
  outDir: path.resolve(process.argv[2]),
  types: ['node'],
  typeRoots: [path.resolve('bazel/node/node_modules/@types')],
};
const sources = fs.readdirSync('js/common/lib').filter((name) => name.endsWith('.ts'))
    .map((name) => path.resolve('js/common/lib', name));
const program = ts.createProgram(sources, options);
const result = program.emit();
const errors = [...converted.errors, ...ts.getPreEmitDiagnostics(program), ...result.diagnostics]
    .filter((item) => item.category === ts.DiagnosticCategory.Error);
if (errors.length || result.emitSkipped) {
  throw new Error(ts.formatDiagnosticsWithColorAndContext(errors, {
    getCurrentDirectory: () => process.cwd(),
    getCanonicalFileName: (name) => name,
    getNewLine: () => '\n',
  }) || 'Common API emission was skipped; inspect the TypeScript compiler configuration.');
}
