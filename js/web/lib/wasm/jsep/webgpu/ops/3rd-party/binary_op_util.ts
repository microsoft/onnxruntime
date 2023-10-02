/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/* eslint-disable @typescript-eslint/naming-convention */

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/binary_op_util.ts

export enum BinaryOpType {
  ADD,
  ATAN2,
  COMPLEX_MULTIPLY_IMAG,
  COMPLEX_MULTIPLY_REAL,
  DIV,
  ELU_DER,
  EQUAL,
  FLOOR_DIV,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  LOGICAL_AND,
  LOGICAL_OR,
  MAX,
  MIN,
  MOD,
  MUL,
  NOT_EQUAL,
  POW,
  PRELU,
  SQUARED_DIFFERENCE,
  SUB
}

const ADD = 'let resultTemp = a + b;';
const ATAN2 = 'let resultTemp = atan2(a, b);';
// (Ar + Ai)(Br + Bi) =
// ArBr + ArBi + AiBr + AiBi = ArBr - AB + ArBi + AiBr
// Yr = ArBr - AB
// Yi = ArBi + AiBr
const COMPLEX_MULTIPLY_REAL = 'let resultTemp = areal * breal - aimag * bimag;';
const COMPLEX_MULTIPLY_IMAG = 'let resultTemp = areal * bimag + aimag * breal;';
const DIV = 'let resultTemp = a / b;';
const ELU_DER = 'let resultTemp = select(a * (b + 1.0), a, b >= b - b);';
const EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a == b);
`;
const FLOOR_DIV = `
  let remainder =
      select(a % b, round(a % b), (round(a) == a) & (round(b) == b));
  let quotient = (a - remainder) / b;
  let resultTemp =
      round(select(quotient, quotient - 1, sign(remainder) == -sign(b)));
`;
const GREATER = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a > b);
`;
const GREATER_EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a >= b);
`;
const LESS = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a < b);
`;
const LESS_EQUAL = `
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a <= b);
`;
const LOGICAL_AND = 'return f32(a >= 1.0 && b >= 1.0);';
const LOGICAL_AND_VEC4 = `return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`;
const LOGICAL_OR = 'return f32(a >= 1.0 || b >= 1.0);';
const LOGICAL_OR_VEC4 = `return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`;
const MAX = 'let resultTemp = max(a, b);';
const MIN = 'let resultTemp = min(a, b);';
const MOD = `
  let isNaN = b == 0.;
  var resultTemp = a % b;
  resultTemp = select((resultTemp + b) % b, resultTemp,
      (a < 0. && b < 0.) || (a >= 0. && b > 0.));
`;
const MOD_VEC4 = `
  let isNaN = !vec4<bool>(b);
  var resultTemp = vec4<f32>(a % b);
  if (!((a[0] < 0. && b[0] < 0.) || (a[0] >= 0. && b[0] > 0.))) {
    resultTemp[0] = (resultTemp[0] + b[0]) % b[0];
  }
  if (!((a[1] < 0. && b[1] < 0.) || (a[1] >= 0. && b[1] > 0.))) {
    resultTemp[1] = (resultTemp[1] + b[1]) % b[1];
  }
  if (!((a[2] < 0. && b[2] < 0.) || (a[2] >= 0. && b[2] > 0.))) {
    resultTemp[2] = (resultTemp[2] + b[2]) % b[2];
  }
  if (!((a[3] < 0. && b[3] < 0.) || (a[3] >= 0. && b[3] > 0.))) {
    resultTemp[3] = (resultTemp[3] + b[3]) % b[3];
  }
`;
const MUL = 'let resultTemp = a * b;';
const NOT_EQUAL = `
  var resultTemp = f32(a != b);
  let valueForNaN = 1.0;
`;
const NOT_EQUAL_VEC4 = `
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
`;

const POW = `
  let isNaN = a < 0.0 && floor(b) < b;
  if (b == 0.0) {
    return 1.0;
  }
  var resultTemp = select(sign(a) * pow(abs(a), b), pow(abs(a), b),
      round(abs(b) % 2.0) != 1.0);
`;
const POW_VEC4 = `
  let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = (a < vec4<f32>(0.0)) & (floor(b) < b);
`;

// eslint-disable-next-line @typescript-eslint/quotes
const PRELU = `if (a < 0.0) { return b * a; }  return a;`;
const PRELU_VEC4 = `
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`;
const SQUARED_DIFFERENCE = 'let resultTemp = (a - b) * (a - b);';
const SUB = 'let resultTemp = a - b;';

export const getBinaryOpString = (type: BinaryOpType, useVec4?: boolean) => {
  let doOpSnippet = '';

  // Ops with NaN check
  do {
    switch (type) {
      case BinaryOpType.ATAN2:
        doOpSnippet = ATAN2;
        break;
      case BinaryOpType.MAX:
        doOpSnippet = MAX;
        break;
      case BinaryOpType.MIN:
        doOpSnippet = MIN;
        break;
      case BinaryOpType.MOD:
        doOpSnippet = useVec4 ? MOD_VEC4 : MOD;
        break;
      case BinaryOpType.NOT_EQUAL:
        doOpSnippet = useVec4 ? NOT_EQUAL_VEC4 : NOT_EQUAL;
        break;
      case BinaryOpType.POW:
        doOpSnippet = useVec4 ? POW_VEC4 : POW;
        break;
      default:
        continue;
    }

    let isNaN: string;
    let dTypeN: string;
    let boolN: string;
    if (useVec4) {
      isNaN = 'isnanVec4';
      dTypeN = 'vec4<f32>';
      boolN = 'vec4<bool>';
    } else {
      isNaN = 'isnan';
      dTypeN = 'f32';
      boolN = 'bool';
    }

    return `
      let aIsNaN = ${isNaN}(a);
      let aPostLegalization = select(a, ${dTypeN}(42), aIsNaN);
      let bIsNaN = ${isNaN}(b);
      let bPostLegalization = select(b, ${dTypeN}(42), bIsNaN);
      let isNaN = false;
      let valueForNaN = uniforms.NAN;
      {
        let a = aPostLegalization;
        let b = bPostLegalization;
        ${doOpSnippet}
        return select(
            resultTemp, ${dTypeN}(valueForNaN),
            ${boolN}(isNaN) | aIsNaN | bIsNaN);
      }
    `;
    // eslint-disable-next-line no-constant-condition
  } while (false);

  // Ops without NaN check
  switch (type) {
    case BinaryOpType.ADD:
      doOpSnippet = ADD;
      break;
    case BinaryOpType.COMPLEX_MULTIPLY_IMAG:
      doOpSnippet = COMPLEX_MULTIPLY_IMAG;
      break;
    case BinaryOpType.COMPLEX_MULTIPLY_REAL:
      doOpSnippet = COMPLEX_MULTIPLY_REAL;
      break;
    case BinaryOpType.DIV:
      doOpSnippet = DIV;
      break;
    case BinaryOpType.ELU_DER:
      doOpSnippet = ELU_DER;
      break;
    case BinaryOpType.EQUAL:
      doOpSnippet = EQUAL;
      break;
    case BinaryOpType.FLOOR_DIV:
      doOpSnippet = FLOOR_DIV;
      break;
    case BinaryOpType.GREATER:
      doOpSnippet = GREATER;
      break;
    case BinaryOpType.GREATER_EQUAL:
      doOpSnippet = GREATER_EQUAL;
      break;
    case BinaryOpType.LESS:
      doOpSnippet = LESS;
      break;
    case BinaryOpType.LESS_EQUAL:
      doOpSnippet = LESS_EQUAL;
      break;
    case BinaryOpType.LOGICAL_AND:
      return useVec4 ? LOGICAL_AND_VEC4 : LOGICAL_AND;
    case BinaryOpType.LOGICAL_OR:
      return useVec4 ? LOGICAL_OR_VEC4 : LOGICAL_OR;
    case BinaryOpType.MUL:
      doOpSnippet = MUL;
      break;
    case BinaryOpType.PRELU:
      return useVec4 ? PRELU_VEC4 : PRELU;
    case BinaryOpType.SQUARED_DIFFERENCE:
      doOpSnippet = SQUARED_DIFFERENCE;
      break;
    case BinaryOpType.SUB:
      doOpSnippet = SUB;
      break;
    default:
      // throw new Error(`BinaryType ${type} is not implemented!`);
  }
  return `
    ${doOpSnippet}
    return resultTemp;
  `;
};
