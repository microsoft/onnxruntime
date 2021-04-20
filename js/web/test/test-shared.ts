// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import * as fs from 'fs';
import {promisify} from 'util';

import {Attribute} from '../lib/onnxjs/attribute';
import {Graph} from '../lib/onnxjs/graph';

export function base64toBuffer(data: string): Uint8Array {
  return Buffer.from(data, 'base64');
}

export function bufferToBase64(buffer: Uint8Array): string {
  return Buffer.from(buffer).toString('base64');
}

async function readFile(file: string) {
  if (typeof fetch === 'undefined') {
    // node
    return promisify(fs.readFile)(file);
  } else {
    // browser
    const response = await fetch(file);
    const buffer = await response.arrayBuffer();
    return Buffer.from(buffer);
  }
}

export async function readJsonFile(file: string): Promise<any> {
  const content = await readFile(file);
  return JSON.parse(content.toString());
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
