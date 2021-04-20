// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';
import {replaceInlines} from '../../../../lib/onnxjs/backends/webgl/glsl-function-inliner';
import {Logger} from '../../../../lib/onnxjs/instrument';

function removeWhiteSpace(s: string): string {
  return s.replace(/\s+/gm, ' ');
}

describe('#UnitTest# - FunctionInliner', () => {
  it('replaces inline and removes original declaration', () => {
    const script = `
      precision mediump float;
      varying vec4 vColor;

      @inline
        float
          _and(
              float a,
              float b
            ) {
        bool aBool = bool(a);
        bool bBool = bool(b);
        return float(aBool ^^ bBool);
      }
      @inline float _less(float a, float b) { return float(a < b); }
      void main(void) {
        float m = 2.3;
        float l;
        l = _less(2.0, 3.4);
        float k = _and(1.2, -2.1);
        gl_FragColor = vColor;
      }
      `;
    const result = replaceInlines(script);
    const expected = `
      precision mediump float;
      varying vec4 vColor;

      void main(void) {
        float m = 2.3;
        float l;
        {
          float a = 2.0;
          float b = 3.4;
          l = float(a < b);
        }
        float k;
        {
          float a = 1.2;
          float b = -2.1;
          bool aBool = bool(a);
          bool bBool = bool(b);
          k = float(aBool ^^ bBool);
        }
        gl_FragColor = vColor;
      }
      `;
    const result2 = removeWhiteSpace(result);
    const expected2 = removeWhiteSpace(expected);
    Logger.verbose(`Result after cleanup:\n${result2}`);
    Logger.verbose(`Expected after cleanup:\n${expected2}`);
    expect(result2).to.equal(expected2);
  });
});
