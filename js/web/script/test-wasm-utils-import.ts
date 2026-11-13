// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { buildSync } from 'esbuild';
import assert from 'node:assert/strict';
import * as path from 'node:path';
import { runInNewContext } from 'node:vm';

interface Scenario {
  name: string;
  metaUrl?: string;
  origin?: string;
  bundleFilename?: string;
  node?: boolean;
  classicScript?: string;
  expected?: string;
}

const resolveScriptSrc = (scenario: Scenario): string | undefined => {
  const result = buildSync({
    entryPoints: [path.join(__dirname, '../lib/wasm/wasm-utils-import.ts')],
    bundle: true,
    platform: 'browser',
    format: 'cjs',
    write: false,
    define: {
      'BUILD_DEFS.IS_ESM': String(!scenario.classicScript),
      'BUILD_DEFS.ESM_IMPORT_META_URL': JSON.stringify(scenario.metaUrl) ?? 'undefined',
      'BUILD_DEFS.BUNDLE_FILENAME': JSON.stringify(scenario.bundleFilename ?? 'ort.bundle.min.mjs'),
      'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
      'BUILD_DEFS.ENABLE_BUNDLE_WASM_JS': 'false',
      'BUILD_DEFS.DISABLE_JSEP': 'true',
      'BUILD_DEFS.ENABLE_JSPI': 'false',
      'BUILD_DEFS.DISABLE_WEBGPU': 'true',
    },
  });
  const module = { exports: {} as { scriptSrc?: string } };
  runInNewContext(
    result.outputFiles[0].text,
    {
      module,
      URL,
      location: scenario.origin === undefined ? undefined : { origin: scenario.origin },
      process: scenario.node ? { versions: { node: '24' } } : undefined,
      document: scenario.classicScript ? { currentScript: { src: scenario.classicScript } } : undefined,
    },
    { timeout: 1000 },
  );
  return module.exports.scriptSrc;
};

const scenarios: Scenario[] = [
  {
    name: 'local file with an opaque origin',
    metaUrl: 'file:///build/entry.mjs',
    origin: 'null',
    expected: 'file:///build/ort.bundle.min.mjs',
  },
  {
    name: 'relative bundle name resolved against a file URL',
    metaUrl: 'file:///build/entry.mjs',
    bundleFilename: './ort.bundle.min.mjs',
    origin: 'null',
    expected: 'file:///build/ort.bundle.min.mjs',
  },
  {
    name: 'hardcoded file URL in an HTTPS page',
    metaUrl: 'file:///build/entry.mjs',
    origin: 'https://app.example',
    expected: 'file:///build/ort.bundle.min.mjs',
  },
  {
    name: 'file URL without a location global',
    metaUrl: 'file:///build/entry.mjs',
    expected: 'file:///build/ort.bundle.min.mjs',
  },
  {
    name: 'unmodified HTTPS import URL',
    metaUrl: 'https://cdn.example/ort.bundle.min.mjs',
    origin: 'https://app.example',
    expected: 'https://cdn.example/ort.bundle.min.mjs',
  },
  {
    name: 'unmodified blob import URL',
    metaUrl: 'blob:https://app.example/1234',
    origin: 'null',
    expected: 'blob:https://app.example/1234',
  },
  {
    name: 'unmodified data import URL',
    metaUrl: 'data:text/javascript,export default 0',
    origin: 'null',
    expected: 'data:text/javascript,export default 0',
  },
  { name: 'missing inline import URL', origin: 'null' },
  { name: 'Node ignores browser location', metaUrl: 'file:///build/entry.mjs', origin: 'null', node: true },
  {
    name: 'classic script source',
    classicScript: 'https://cdn.example/ort.min.js',
    origin: 'https://app.example',
    expected: 'https://cdn.example/ort.min.js',
  },
];

for (const scenario of scenarios) {
  assert.equal(resolveScriptSrc(scenario), scenario.expected, scenario.name);
}

// An invalid bundle URL must still fail instead of being hidden by the origin fallback.
assert.throws(
  () =>
    resolveScriptSrc({
      name: 'invalid bundle URL',
      metaUrl: 'file:///build/entry.mjs',
      bundleFilename: 'http://[',
      origin: 'null',
    }),
  TypeError,
);
