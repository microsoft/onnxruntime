// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Graph} from './graph';
import {Operator} from './operators';

export interface OpSet {
  domain: string;
  version: number;
}

export declare namespace OpSet {
  interface OperatorConstructor {
    (node: Graph.Node): Operator;
  }

  /**
   * Domain of an opset, it can be an empty string(default value, represent for ai.onnx), or 'ai.onnx.ml'
   */
  type Domain = ''|'ai.onnx.ml';

  /**
   * A resolve rule consists of 4 items: opType, opSetDomain, versionSelector and operatorConstructor
   */
  type ResolveRule = [string, Domain, string, OperatorConstructor];
}

export function resolveOperator(node: Graph.Node, opsets: readonly OpSet[], rules: readonly OpSet.ResolveRule[]) {
  for (const rule of rules) {
    const opType = rule[0];
    const domain = rule[1];
    const versionSelector = rule[2];
    const opConstructor = rule[3];

    if (node.opType === opType) {  // operator type matches
      for (const opset of opsets) {
        // opset '' and 'ai.onnx' are considered the same.
        if (opset.domain === domain || (opset.domain === 'ai.onnx' && domain === '')) {  // opset domain found
          if (matchSelector(opset.version, versionSelector)) {
            return opConstructor(node);
          }
        }
      }
    }
  }

  throw new TypeError(`cannot resolve operator '${node.opType}' with opsets: ${
      opsets.map(set => `${set.domain || 'ai.onnx'} v${set.version}`).join(', ')}`);
}

function matchSelector(version: number, selector: string): boolean {
  if (selector.endsWith('+')) {
    // minimum version match ('7+' expects version>=7)
    const rangeStart = Number.parseInt(selector.substring(0, selector.length - 1), 10);
    return !isNaN(rangeStart) && rangeStart <= version;
  } else if (selector.split('-').length === 2) {
    // range match ('6-8' expects 6<=version<=8)
    const pair = selector.split('-');
    const rangeStart = Number.parseInt(pair[0], 10);
    const rangeEnd = Number.parseInt(pair[1], 10);
    return !isNaN(rangeStart) && !isNaN(rangeEnd) && rangeStart <= version && version <= rangeEnd;
  } else {
    // exact match ('7' expects version===7)
    return Number.parseInt(selector, 10) === version;
  }
}
