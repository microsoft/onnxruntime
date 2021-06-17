// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {Tensor} from '../tensor';

export interface ProgramVariable {
  type: 'float'|'int';
  name: string;
  arrayLength?: number;
  data: number|number[];
}

export interface ProgramInfo {
  samplers: string[];
  variables?: ProgramVariable[];
  shaderSource: string;
  hasMain?: boolean;

  expectPackedInputs?: boolean;
  expectPackedOutputs?: boolean;
}


export class WebGLInferenceHandler /* implements InferenceHandler */ {
  runProgram(programInfo: ProgramInfo, inputs: Tensor[]): Tensor {
    const programKey = generateKey(programInfo, inputs);
    let artifact = this.cache.get(programKey);
    if (!cachedArtifact) {
      artifact = this.compile(programInfo);
      this.cache.set(programKey, artifact);
    }

    return this.run(artifact, ...);
  }
}

//
// Operator resolve
//
// ...
// ['Conv', '7+'] --> [myConv]
//

function myConv(inferenceHandler: WebGLInferenceHandler, attribute: Attribute, inputs: Tensor[]): Tensor[] {
  const packed = env.webgl.pack;

  const group = ...;
  if (group > 1) {
    inferenceHandler.runProgram(packedGroupedConv(attribute, inputs), inputs);
  } else {
    ...
  }
}

function packedGroupedConv(attribute: Attribute, inputs: Tensor[]): ProgramInfo {
  let shader = '..';
  // shader += '...';

  return {
    samplers: ['X', 'W'],
    shaderSource: shader,
    expectPackedInputs: true,
    expectPackedOutputs: true,
  };
}
