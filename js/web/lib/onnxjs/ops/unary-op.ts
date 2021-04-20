// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class UnaryOp implements Operator {
  constructor(protected typeConstraint: readonly Tensor.DataType[], protected resultType?: Tensor.DataType) {}

  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(_attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (this.typeConstraint.indexOf(inputs[0].type) === -1) {
      return false;
    }

    return true;
  }
}
