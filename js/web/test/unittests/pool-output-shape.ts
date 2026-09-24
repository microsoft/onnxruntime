// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// These tests target PoolConvUtil.computePoolOutputShape directly (the pure shape function),
// not the end-to-end pooling op. That is intentional: while the shape math now honors
// ceil_mode, execution is still gated by the ceil_mode throw in parseAveragePool/MaxPoolAttributes
// (js/web/lib/wasm/jsep/webgpu/ops/pool.ts), because the WebGPU kernel does not yet implement
// ceil_mode trailing-padding handling. Removing that throw + adding kernel support is a tracked
// follow-up; until then the shape path is only reachable via this unit test.

import { expect } from 'chai';

import { PoolConvUtil as JsepPoolConvUtil } from '../../lib/wasm/jsep/util';
import { PoolConvUtil as OnnxjsPoolConvUtil } from '../../lib/onnxjs/util';

// Ground-truth output shapes are taken from the C++ CPU reference
// (onnxruntime/core/providers/cpu/nn/pool_attributes.h ComputeOutputSize) and the
// matching CPU pool_op_test.cc cases, so all pooling targets share one oracle and
// cannot silently re-diverge on ceil_mode.
interface PoolShapeCase {
  readonly name: string;
  readonly isGlobalOperator: boolean;
  readonly inputDims: number[];
  readonly strides: number[];
  readonly dilations: number[];
  readonly kernelShape: number[];
  readonly pads: number[];
  readonly autoPad?: string;
  readonly ceilMode: number;
  readonly expectedShape: number[];
  // When set, asserts the auto_pad pad distribution written back to the pads out-param.
  readonly expectedPads?: number[];
}

const poolShapeCases: PoolShapeCase[] = [
  // floor mode must stay identical to the previous behavior (no regression).
  {
    name: 'NOTSET floor mode is unchanged (test_maxpool_2d_default analogue)',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [2, 2],
    dilations: [1, 1],
    kernelShape: [3, 3],
    pads: [0, 0, 0, 0],
    ceilMode: 0,
    expectedShape: [1, 1, 1, 1],
  },
  // test_maxpool_2d_ceil: floor would give [1,1] but ceil_mode gives [2,2].
  {
    name: 'test_maxpool_2d_ceil -> [1,1,2,2]',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [2, 2],
    dilations: [1, 1],
    kernelShape: [3, 3],
    pads: [0, 0, 0, 0],
    ceilMode: 1,
    expectedShape: [1, 1, 2, 2],
  },
  // AveragePool_10_ceil1_2d: asymmetric strides [3,1] -> [2,3].
  {
    name: 'AveragePool_10_ceil1_2d -> [1,1,2,3]',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [3, 1],
    dilations: [1, 1],
    kernelShape: [2, 2],
    pads: [0, 0, 0, 0],
    ceilMode: 1,
    expectedShape: [1, 1, 2, 3],
  },
  // AveragePool_18/19_ceil_count_include_pad_1d: PyTorch #183528 repro shape.
  {
    name: 'AveragePool ceil 1d (pads 3,3 kernel 7 stride 3) -> [1,2,4]',
    isGlobalOperator: false,
    inputDims: [1, 2, 9],
    strides: [3],
    dilations: [1],
    kernelShape: [7],
    pads: [3, 3],
    ceilMode: 1,
    expectedShape: [1, 2, 4],
  },
  // AveragePool_18_ceil_count_include_pad_2d.
  {
    name: 'AveragePool ceil 2d (pads 1 kernel 3 stride 2) -> [1,1,3,3]',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [2, 2],
    dilations: [1, 1],
    kernelShape: [3, 3],
    pads: [1, 1, 1, 1],
    ceilMode: 1,
    expectedShape: [1, 1, 3, 3],
  },
  // AveragePool_18_ceil_count_include_pad_3d.
  {
    name: 'AveragePool ceil 3d (pads 1 kernel 3 stride 2) -> [1,1,2,2,2]',
    isGlobalOperator: false,
    inputDims: [1, 1, 3, 3, 3],
    strides: [2, 2, 2],
    dilations: [1, 1, 1],
    kernelShape: [3, 3, 3],
    pads: [1, 1, 1, 1, 1, 1],
    ceilMode: 1,
    expectedShape: [1, 1, 2, 2, 2],
  },
  // Shrink-rule discriminator: naive ceil() would give 3, but the last window would start
  // entirely in the trailing padding, so C++ ComputeOutputSize shrinks it back to 2.
  {
    name: 'shrink rule: last window starting in padding is dropped -> [1,1,2]',
    isGlobalOperator: false,
    inputDims: [1, 1, 3],
    strides: [2],
    dilations: [1],
    kernelShape: [2],
    pads: [1, 1],
    ceilMode: 1,
    expectedShape: [1, 1, 2],
  },
  // ceil_mode with dilation (AveragePool_19_dilation_2d shape).
  {
    name: 'ceil 2d with dilation 2 -> [1,1,2,2]',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [1, 1],
    dilations: [2, 2],
    kernelShape: [2, 2],
    pads: [0, 0, 0, 0],
    ceilMode: 1,
    expectedShape: [1, 1, 2, 2],
  },
  // VALID auto_pad honors ceil_mode.
  {
    name: 'VALID auto_pad + ceil_mode -> [1,1,2,2]',
    isGlobalOperator: false,
    inputDims: [1, 1, 4, 4],
    strides: [2, 2],
    dilations: [1, 1],
    kernelShape: [3, 3],
    pads: [0, 0, 0, 0],
    autoPad: 'VALID',
    ceilMode: 1,
    expectedShape: [1, 1, 2, 2],
  },
  // SAME_UPPER auto_pad honors ceil_mode (exercises the recomputed-pad branch).
  // Reviewer-verified: in=5, stride=2, kernel=3, SAME_UPPER, ceil -> 3.
  {
    name: 'SAME_UPPER auto_pad + ceil_mode -> [1,1,3]',
    isGlobalOperator: false,
    inputDims: [1, 1, 5],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_UPPER',
    ceilMode: 1,
    expectedShape: [1, 1, 3],
  },
  // SAME_LOWER auto_pad honors ceil_mode (same numeric case, mirrored padding split).
  {
    name: 'SAME_LOWER auto_pad + ceil_mode -> [1,1,3]',
    isGlobalOperator: false,
    inputDims: [1, 1, 5],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_LOWER',
    ceilMode: 1,
    expectedShape: [1, 1, 3],
  },
  // Non-divisible SAME_UPPER + ceil_mode regression guard: legacyTargetSize must use C++
  // integer division. With float division this yielded [1,1,2] instead of the correct [1,1,1].
  // (in=2, stride=2, kernel=3, SAME_UPPER, ceil -> 1, matching C++ ComputeSizePadDilations.)
  {
    name: 'SAME_UPPER auto_pad + ceil_mode non-divisible -> [1,1,1]',
    isGlobalOperator: false,
    inputDims: [1, 1, 2],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_UPPER',
    ceilMode: 1,
    expectedShape: [1, 1, 1],
  },
  // SAME_LOWER symmetric counterpart of the non-divisible regression guard.
  {
    name: 'SAME_LOWER auto_pad + ceil_mode non-divisible -> [1,1,1]',
    isGlobalOperator: false,
    inputDims: [1, 1, 2],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_LOWER',
    ceilMode: 1,
    expectedShape: [1, 1, 1],
  },
  // ceilMode=0 SAME_* regression guards: the Math.floor(legacyTargetSize) fix also corrects the
  // auto_pad pad DISTRIBUTION on the floor path (output shape is unchanged, so only asserting the
  // shape would miss it). Old float division gave pads (1,1); C++ integer division gives (0,1) for
  // SAME_UPPER and (1,0) for SAME_LOWER at in=2, stride=2, kernel=3.
  {
    name: 'SAME_UPPER auto_pad ceilMode=0 non-divisible corrects pad split -> pads [0,1]',
    isGlobalOperator: false,
    inputDims: [1, 1, 2],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_UPPER',
    ceilMode: 0,
    expectedShape: [1, 1, 1],
    expectedPads: [0, 1],
  },
  {
    name: 'SAME_LOWER auto_pad ceilMode=0 non-divisible corrects pad split -> pads [1,0]',
    isGlobalOperator: false,
    inputDims: [1, 1, 2],
    strides: [2],
    dilations: [1],
    kernelShape: [3],
    pads: [0, 0],
    autoPad: 'SAME_LOWER',
    ceilMode: 0,
    expectedShape: [1, 1, 1],
    expectedPads: [1, 0],
  },
];

function runPoolConvUtil(
  computePoolOutputShape: typeof JsepPoolConvUtil.computePoolOutputShape,
  testCase: PoolShapeCase,
): { shape: number[]; pads: number[] } {
  // strides/dilations/kernelShape/pads are copied because computePoolOutputShape mutates pads
  // in the auto_pad branches. The mutated pads copy is returned so tests can assert the
  // computed auto_pad pad distribution, not just the output shape.
  const pads = testCase.pads.slice();
  const shape = computePoolOutputShape(
    testCase.isGlobalOperator,
    testCase.inputDims,
    testCase.strides.slice(),
    testCase.dilations.slice(),
    testCase.kernelShape.slice(),
    pads,
    testCase.autoPad,
    testCase.ceilMode,
  );
  return { shape, pads };
}

describe('PoolConvUtil.computePoolOutputShape ceil_mode', () => {
  for (const testCase of poolShapeCases) {
    it(`[jsep] ${testCase.name}`, () => {
      const { shape, pads } = runPoolConvUtil(JsepPoolConvUtil.computePoolOutputShape, testCase);
      expect(shape).to.deep.equal(testCase.expectedShape);
      if (testCase.expectedPads) {
        expect(pads).to.deep.equal(testCase.expectedPads);
      }
    });
    it(`[onnxjs] ${testCase.name}`, () => {
      const { shape, pads } = runPoolConvUtil(OnnxjsPoolConvUtil.computePoolOutputShape, testCase);
      expect(shape).to.deep.equal(testCase.expectedShape);
      if (testCase.expectedPads) {
        expect(pads).to.deep.equal(testCase.expectedPads);
      }
    });
  }

  it('defaults to floor when ceilMode is omitted', () => {
    const jsep = JsepPoolConvUtil.computePoolOutputShape(false, [1, 1, 4, 4], [2, 2], [1, 1], [3, 3], [0, 0, 0, 0]);
    const onnxjs = OnnxjsPoolConvUtil.computePoolOutputShape(false, [1, 1, 4, 4], [2, 2], [1, 1], [3, 3], [0, 0, 0, 0]);
    expect(jsep).to.deep.equal([1, 1, 1, 1]);
    expect(onnxjs).to.deep.equal([1, 1, 1, 1]);
  });
});
