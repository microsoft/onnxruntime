// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const path = require("node:path");
const { test } = require("node:test");

const script = path.join(__dirname, "validate-shader-key.js");
const configuration =
  "subgroups=1;subgroup_matrix=0;max_storage_buffer_binding_size=134217728";
const shader = "enable subgroups;\n@compute @workgroup_size(1) fn main() {}\n";

function record(code, config = configuration, key = "test-key") {
  const configField = config === null ? "" : `, Configuration="${config}"`;
  const keyField = key === null ? "" : `, Key="${key}"`;
  const header = `=== WebGPU Shader code [Test${configField}${keyField}]`;
  return `${header} Start ===\n${code}${header} End ===\n`;
}

function validate(log) {
  const result = spawnSync(process.execPath, [script], {
    input: log,
    encoding: "utf8",
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null);
  return result;
}

test("equivalent configurations compare independently compiled shaders", () => {
  const result = validate(record(shader) + record(shader));
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Total 1 shader keys found/);
});

test("different device configurations may generate different source", () => {
  const configurations = [
    configuration.replace("subgroups=1", "subgroups=0"),
    configuration.replace("subgroup_matrix=0", "subgroup_matrix=1"),
    configuration.replace("134217728", "65536"),
  ];
  for (const other of configurations) {
    const result = validate(record(shader) + record("different source\n", other));
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /Total 2 shader keys found/);
  }
});

test("a conflicting key in an equivalent configuration still fails", () => {
  const result = validate(
    record(shader) +
      record("unrelated source\n", "other-configuration") +
      record("conflicting source\n")
  );
  assert.equal(result.status, 1, result.stdout);
  for (const text of [
    "Found inconsistent shader code",
    "test-key",
    configuration,
    ...shader.trim().split("\n"),
    "conflicting source",
  ]) {
    assert.ok(result.stderr.includes(text), result.stderr);
  }
});

test("shader headers require configuration and key metadata", () => {
  const verbose =
    'onnxruntime::webgpu::WebGpuContext::Run Starting program "test-key"\n';
  const original = record(shader);
  const start = original.slice(0, original.lastIndexOf("=== WebGPU Shader code"));
  for (const malformed of [
    record(shader, null),
    record(shader, configuration, null),
    record(shader, null, null),
    record(shader, ""),
    record(shader, configuration, ""),
  ]) {
    const end = malformed.slice(malformed.lastIndexOf("=== WebGPU Shader code"));
    for (const log of [original + verbose + malformed, start + end]) {
      const result = validate(log);
      assert.equal(result.status, 1, result.stdout);
      assert.match(
        result.stderr,
        /Shader header must include nonempty Configuration and Key fields/
      );
    }
  }
});

test("shader end must match the starting key and configuration", () => {
  const original = record(shader);
  const end = original.lastIndexOf("=== WebGPU Shader code");
  const start = original.slice(0, end);
  for (const ending of [
    original.slice(end).replace('Key="test-key"', 'Key="other-key"'),
    original.slice(end).replace(configuration, "other-configuration"),
  ]) {
    const result = validate(start + ending);
    assert.equal(result.status, 1, result.stdout);
    assert.match(result.stderr, /Shader end key or configuration does not match/);
  }
});
