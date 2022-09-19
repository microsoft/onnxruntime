// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WORKGROUP_SIZE} from './common';

export const declareWorkgroupSize = (size?: number): string =>
    `const WORKGROUP_SIZE: u32 = ${size ?? WORKGROUP_SIZE}u;\n`;

export const declareDataSize = (size: number): string => `const DATA_SIZE = u32(${size});`;

// export const declareInputsOutputs = (inputs: readonly string[], outputs: readonly string[]): string => {
// };

const globalIndexExpression = (dispatchGroup: [number, number, number]): string => {
  const [, y, z] = dispatchGroup;

  if (y === 1 && z === 1) {
    return 'global_id.x';
  }

  if (z === 1) {
    return `global_id.x + global_id.y * ${y}u * WORKGROUP_SIZE`;
  }

  return `global_id.x + (global_id.y * ${y}u + global_id.z * ${y * z}u) * WORKGROUP_SIZE`;
};

export const mainBegin = (dispatchGroup: [number, number, number]): string => `@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let global_index = ${globalIndexExpression(dispatchGroup)};
  // Guard against out-of-bounds work group sizes
    if (global_index >= DATA_SIZE) {
      return;
    }
`;
