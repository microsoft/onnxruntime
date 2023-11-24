// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as base64 from 'base64-js';
import * as fs from 'node:fs/promises';

import {Attribute} from '../lib/onnxjs/attribute';
import {Graph} from '../lib/onnxjs/graph';

export function base64toBuffer(data: string): Uint8Array {
  return base64.toByteArray(data);
}

export function bufferToBase64(buffer: Uint8Array): string {
  return base64.fromByteArray(buffer);
}

export async function readFile(file: string) {
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    // node
    return fs.readFile(file);
  } else {
    // browser
    const response = await fetch(file);
    return new Uint8Array(await response.arrayBuffer());
  }
}

export async function readJsonFile(file: string): Promise<any> {
  const content = await readFile(file);
  return JSON.parse(new TextDecoder().decode(content));
}

/**
 * create a single-node graph for unit test purpose
 */
export function createMockGraph(opType: string, attributes: Attribute): Graph {
  const node: Graph.Node = {name: '', opType, inputs: [], outputs: [], attributes};
  return {
    getInputIndices: () => [],
    getInputNames: () => [],
    getOutputIndices: () => [],
    getOutputNames: () => [],
    getNodes: () => [node],
    getValues: () => []
  };
}
