// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { env, InferenceSession, LoraAdapter, Tensor } from 'onnxruntime-common';

// onnxruntime/test/testdata/lora/two_params_lora_model.onnx
const ONNX_MODEL_TWO_PARAMS_LORA = Uint8Array.from([
  8, 10, 58, 197, 3, 10, 40, 10, 7, 105, 110, 112, 117, 116, 95, 120, 10, 8, 119, 101, 105, 103, 104, 116, 95, 120, 18,
  11, 109, 109, 95, 111, 117, 116, 112, 117, 116, 95, 120, 34, 6, 77, 97, 116, 77, 117, 108, 10, 44, 10, 7, 105, 110,
  112, 117, 116, 95, 120, 10, 12, 108, 111, 114, 97, 95, 112, 97, 114, 97, 109, 95, 97, 18, 11, 109, 109, 95, 111, 117,
  116, 112, 117, 116, 95, 97, 34, 6, 77, 97, 116, 77, 117, 108, 10, 48, 10, 11, 109, 109, 95, 111, 117, 116, 112, 117,
  116, 95, 97, 10, 12, 108, 111, 114, 97, 95, 112, 97, 114, 97, 109, 95, 98, 18, 11, 109, 109, 95, 111, 117, 116, 112,
  117, 116, 95, 98, 34, 6, 77, 97, 116, 77, 117, 108, 10, 39, 10, 11, 109, 109, 95, 111, 117, 116, 112, 117, 116, 95,
  120, 10, 11, 109, 109, 95, 111, 117, 116, 112, 117, 116, 95, 98, 18, 6, 111, 117, 116, 112, 117, 116, 34, 3, 65, 100,
  100, 18, 21, 116, 119, 111, 95, 112, 97, 114, 97, 109, 115, 95, 108, 111, 114, 97, 95, 109, 111, 100, 101, 108, 42,
  82, 8, 4, 8, 4, 16, 1, 34, 64, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
  0, 0, 224, 64, 0, 0, 0, 65, 0, 0, 16, 65, 0, 0, 32, 65, 0, 0, 48, 65, 0, 0, 64, 65, 0, 0, 80, 65, 0, 0, 96, 65, 0, 0,
  112, 65, 0, 0, 128, 65, 66, 8, 119, 101, 105, 103, 104, 116, 95, 120, 42, 20, 8, 4, 8, 0, 16, 1, 66, 12, 108, 111,
  114, 97, 95, 112, 97, 114, 97, 109, 95, 97, 42, 20, 8, 0, 8, 4, 16, 1, 66, 12, 108, 111, 114, 97, 95, 112, 97, 114,
  97, 109, 95, 98, 90, 25, 10, 7, 105, 110, 112, 117, 116, 95, 120, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 4, 10, 2, 8,
  4, 90, 33, 10, 12, 108, 111, 114, 97, 95, 112, 97, 114, 97, 109, 95, 97, 18, 17, 10, 15, 8, 1, 18, 11, 10, 2, 8, 4,
  10, 5, 18, 3, 100, 105, 109, 90, 33, 10, 12, 108, 111, 114, 97, 95, 112, 97, 114, 97, 109, 95, 98, 18, 17, 10, 15, 8,
  1, 18, 11, 10, 5, 18, 3, 100, 105, 109, 10, 2, 8, 4, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10, 12, 8,
  1, 18, 8, 10, 2, 8, 4, 10, 2, 8, 4, 66, 2, 16, 21,
]);

// onnxruntime/test/testdata/lora/two_params_lora_model.onnx_adapter
const ONNX_ADAPTER_TWO_PARAMS_LORA = Uint8Array.from([
  12, 0, 0, 0, 84, 79, 82, 84, 0, 0, 0, 0, 140, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0,
  0, 104, 0, 0, 0, 4, 0, 0, 0, 172, 255, 255, 255, 60, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 16, 0, 0, 0, 0, 0,
  224, 64, 0, 0, 0, 65, 0, 0, 16, 65, 0, 0, 32, 65, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 12, 0, 0, 0, 108, 111, 114, 97, 95, 112, 97, 114, 97, 109, 95, 98, 0, 0, 0, 0, 12, 0, 20, 0, 4, 0, 8, 0, 12, 0, 16,
  0, 12, 0, 0, 0, 60, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 16, 0, 0, 0, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160,
  64, 0, 0, 192, 64, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 108, 111, 114,
  97, 95, 112, 97, 114, 97, 109, 95, 97, 0, 0, 0, 0,
]);

// Expected value of each output row when "input_x" is all ones. See two_params_lora_model.py.
const EXPECTED_OUTPUT_ROW_BASE = [28, 32, 36, 40];
const EXPECTED_OUTPUT_ROW_WITH_ADAPTER = [154, 176, 198, 220];

const createFeeds = (): InferenceSession.FeedsType => ({
  input_x: new Tensor('float32', new Float32Array(16).fill(1), [4, 4]),
});

const expectOutput = (results: InferenceSession.ReturnType, expectedRow: number[]): void => {
  const output = results.output as Tensor;
  expect(output.dims).to.deep.equal([4, 4]);
  expect(Array.from(output.data as Float32Array)).to.deep.equal([
    ...expectedRow,
    ...expectedRow,
    ...expectedRow,
    ...expectedRow,
  ]);
};

const expectRejected = async (promise: Promise<unknown>, expectedMessage: string): Promise<void> => {
  let error: unknown;
  try {
    await promise;
  } catch (e) {
    error = e;
  }
  expect(error).to.be.an('error');
  expect((error as Error).message).to.include(expectedMessage);
};

describe('#UnitTest# - wasm - LoRA adapter', () => {
  let session: InferenceSession;

  before(async () => {
    session = await InferenceSession.create(ONNX_MODEL_TWO_PARAMS_LORA, { executionProviders: ['wasm'] });
  });

  after(async () => {
    await session.release();
  });

  it('run without LoRA adapter', async () => {
    expect(session.inputNames).to.deep.equal(['input_x']);
    expectOutput(await session.run(createFeeds()), EXPECTED_OUTPUT_ROW_BASE);
    expectOutput(await session.run(createFeeds(), { activeLoraAdapters: [] }), EXPECTED_OUTPUT_ROW_BASE);
  });

  it('create LoRA adapter with invalid argument', async () => {
    for (const arg of [42, null, new Float32Array(4)]) {
      await expectRejected(LoraAdapter.create(arg as unknown as Uint8Array), "must be 'path' or 'buffer'");
    }
  });

  it('run with invalid activeLoraAdapters', async () => {
    await expectRejected(
      session.run(createFeeds(), { activeLoraAdapters: {} as LoraAdapter[] }),
      "'activeLoraAdapters' must be an array.",
    );
    await expectRejected(
      session.run(createFeeds(), { activeLoraAdapters: [{ release: async () => {} }] as LoraAdapter[] }),
      'must be an array of LoraAdapter objects',
    );
  });

  if (env.wasm.proxy && typeof document !== 'undefined') {
    it('create LoRA adapter in proxy mode', async () => {
      await expectRejected(LoraAdapter.create(ONNX_ADAPTER_TWO_PARAMS_LORA), 'not supported for proxy');
    });
  } else {
    it('run with LoRA adapter', async () => {
      const adapter = await LoraAdapter.create(ONNX_ADAPTER_TWO_PARAMS_LORA);
      try {
        expectOutput(
          await session.run(createFeeds(), { activeLoraAdapters: [adapter] }),
          EXPECTED_OUTPUT_ROW_WITH_ADAPTER,
        );
        // the adapter is active only for the run that it is passed to.
        expectOutput(await session.run(createFeeds()), EXPECTED_OUTPUT_ROW_BASE);
      } finally {
        await adapter.release();
      }
    });

    it('run with LoRA adapter created from a subarray', async () => {
      const buffer = new Uint8Array(ONNX_ADAPTER_TWO_PARAMS_LORA.byteLength + 16);
      buffer.set(ONNX_ADAPTER_TWO_PARAMS_LORA, 8);
      const adapter = await LoraAdapter.create(buffer.subarray(8, 8 + ONNX_ADAPTER_TWO_PARAMS_LORA.byteLength));
      try {
        expectOutput(
          await session.run(createFeeds(), { activeLoraAdapters: [adapter] }),
          EXPECTED_OUTPUT_ROW_WITH_ADAPTER,
        );
      } finally {
        await adapter.release();
      }
    });

    it('create LoRA adapter from invalid data', async () => {
      await expectRejected(LoraAdapter.create(new Uint8Array([1, 2, 3, 4])), "Can't create a LoRA adapter.");
    });

    it('run with released LoRA adapter', async () => {
      const adapter = await LoraAdapter.create(ONNX_ADAPTER_TWO_PARAMS_LORA);
      await adapter.release();
      await expectRejected(session.run(createFeeds(), { activeLoraAdapters: [adapter] }), 'invalid LoRA adapter id');
      await expectRejected(adapter.release(), 'invalid adapter id');
    });

    if (typeof window !== 'undefined') {
      it('run with LoRA adapter on a session using IO binding', async () => {
        // IO binding is used when an output is preferred to be on GPU. The error is thrown before any GPU work.
        const ioBindingSession = await InferenceSession.create(ONNX_MODEL_TWO_PARAMS_LORA, {
          executionProviders: ['wasm'],
          preferredOutputLocation: 'gpu-buffer',
        });
        const adapter = await LoraAdapter.create(ONNX_ADAPTER_TWO_PARAMS_LORA);
        try {
          await expectRejected(
            ioBindingSession.run(createFeeds(), { activeLoraAdapters: [adapter] }),
            'not supported for a session that uses IO binding',
          );
        } finally {
          await adapter.release();
          await ioBindingSession.release();
        }
      });
    }
  }
});
