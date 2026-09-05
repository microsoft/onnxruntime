// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

import { listSupportedBackends } from '../../lib/backend';
import { assertTensorEqual, SQUEEZENET_INPUT0_DATA, SQUEEZENET_OUTPUT0_DATA, TEST_DATA_ROOT } from '../test-utils';

const skipWithoutWebGpu = (context: Mocha.Context) => {
  if (!listSupportedBackends().some((backend) => backend.name === 'webgpu')) {
    context.skip();
  }
};

describe('API Tests - InferenceSession.run()', async () => {
  let session: InferenceSession;
  const input0 = new Tensor('float32', SQUEEZENET_INPUT0_DATA, [1, 3, 224, 224]);
  const expectedOutput0 = new Tensor('float32', SQUEEZENET_OUTPUT0_DATA, [1, 1000, 1, 1]);

  before(async () => {
    session = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'squeezenet.onnx'));
  });

  it('multiple run() calls', async () => {
    for (let i = 0; i < 1000; i++) {
      const result = await session!.run({ data_0: input0 }, ['softmaxout_1']);
      assertTensorEqual(result.softmaxout_1, expectedOutput0);
    }
  }).timeout(process.arch === 'x64' ? '120s' : 0);

  it('keeps native input resources alive when Tensor disposal bypasses instance methods', async () => {
    const disposeInput = [
      (input: Tensor) => input.dispose(),
      (input: Tensor) => input.dispose.bind(input)(),
      (input: Tensor) => (Object.getPrototypeOf(input) as { dispose: () => void }).dispose.call(input),
    ];

    for (const dispose of disposeInput) {
      const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
      try {
        const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
        const run = localSession.run({ input });
        dispose(input);
        // Releasing while the run is in flight defers the native teardown; it must neither throw
        // nor disturb the queued work.
        await localSession.release();
        assertTensorEqual((await run).output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
      } finally {
        await localSession.release().catch(() => {});
      }
    }
  });

  it('copies CPU input data before starting an asynchronous run', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const inputData = input.data;
    const run = localSession.run({ input });

    inputData.fill(0);
    const result = await run;
    assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    await localSession.release();
  });

  it('copies CPU input data with the arena disabled', async () => {
    // Input copies come from the session's CPU allocator. With enableCpuMemArena off that is the
    // plain allocator rather than the arena, a path the default options never exercise.
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'), {
      enableCpuMemArena: false,
    });
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const inputData = input.data;
      const run = localSession.run({ input });

      inputData.fill(0);
      const result = await run;
      assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    } finally {
      await localSession.release().catch(() => {});
    }
  });

  it('keeps resources alive across overlapping runs after Tensor disposal', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    for (let i = 0; i < 10; ++i) {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const firstRun = localSession.run({ input });
      const secondRun = localSession.run({ input });

      input.dispose();
      const results = await Promise.all([firstRun, secondRun]);
      for (const result of results) {
        assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
      }
    }
    await localSession.release();
  });

  it('survives disposal re-entered from a Tensor property getter', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const inputData = input.data;
    let reads = 0;

    // The binding reads Tensor.data while preparing the run. Releasing the session from that read
    // must leave the run being prepared with a live session; teardown waits for it to finish.
    Object.defineProperty(input, 'data', {
      configurable: true,
      get: () => {
        reads++;
        void localSession.release().catch(() => {});
        return inputData;
      },
    });

    try {
      const run = localSession.run({ input });
      assert.ok(reads > 0, 'expected the binding to read Tensor.data');
      assertTensorEqual((await run).output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    } finally {
      await localSession.release().catch(() => {});
    }
  });

  it('defers session teardown until in-flight runs finish', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const runs = [localSession.run({ input }), localSession.run({ input }), localSession.run({ input })];

      await localSession.release();
      for (const result of await Promise.all(runs)) {
        assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
      }

      await assert.rejects(localSession.run({ input }), /Session already disposed/);
      await assert.rejects(localSession.release(), /Session already disposed/);
    } finally {
      await localSession.release().catch(() => {});
    }
  });

  it('rejects a preallocated CPU output whose type differs from the model output', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      // Same byte length as the float32 output, so only a type check can catch this.
      const output = new Tensor('int32', new Int32Array(5), [1, 5]);

      await assert.rejects(localSession.run({ input }, { output }), /Preallocated output tensor has type int32/);
      assert.deepStrictEqual(Array.from(output.data as Int32Array), [0, 0, 0, 0, 0]);
    } finally {
      await localSession.release();
    }
  });

  it('rejects a preallocated CPU output whose shape differs from the model output', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      // Same element count and byte length as the [1,5] output, so only a shape check can catch this.
      const output = new Tensor('float32', new Float32Array(5), [5, 1]);

      await assert.rejects(localSession.run({ input }, { output }), /Preallocated output tensor has shape \[5,1\]/);
      assert.deepStrictEqual(Array.from(output.data as Float32Array), [0, 0, 0, 0, 0]);
    } finally {
      await localSession.release();
    }
  });

  it('reads Tensor.data once when preparing a preallocated output', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const output = new Tensor('float32', new Float32Array(5), [1, 5]);
      const firstRead = new Float32Array(5);
      const laterReads = new Float32Array(5);
      let reads = 0;

      // An accessor may hand back a different object on every read, so the buffer that gets
      // validated has to be the same one that gets leased and written into.
      Object.defineProperty(output, 'data', {
        configurable: true,
        get: () => (reads++ === 0 ? firstRead : laterReads),
      });

      await localSession.run({ input }, { output });
      assert.strictEqual(reads, 1);
      assert.deepStrictEqual(Array.from(firstRead), [1, 2, 3, 4, 5]);
      assert.deepStrictEqual(Array.from(laterReads), [0, 0, 0, 0, 0]);
    } finally {
      await localSession.release();
    }
  });

  it('allows disjoint views of one ArrayBuffer as concurrent preallocated outputs', async () => {
    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const first = await InferenceSession.create(modelPath);
    const second = await InferenceSession.create(modelPath);
    try {
      const buffer = new ArrayBuffer(64);
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const results = await Promise.all([
        first.run({ input }, { output: new Tensor('float32', new Float32Array(buffer, 0, 5), [1, 5]) }),
        second.run({ input }, { output: new Tensor('float32', new Float32Array(buffer, 32, 5), [1, 5]) }),
      ]);
      for (const result of results) {
        assertTensorEqual(result.output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
      }
    } finally {
      await first.release();
      await second.release();
    }
  });

  it('still rejects overlapping views of one ArrayBuffer', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const buffer = new ArrayBuffer(64);
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const running = localSession.run(
        { input },
        { output: new Tensor('float32', new Float32Array(buffer, 0, 5), [1, 5]) },
      );
      await assert.rejects(
        localSession.run({ input }, { output: new Tensor('float32', new Float32Array(buffer, 4, 5), [1, 5]) }),
        /Preallocated output buffer is already in use/,
      );
      await running;
    } finally {
      await localSession.release();
    }
  });

  it('does not assign result properties through an inherited setter', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    const outputData = new Float32Array(5);
    let stored: unknown;
    let sets = 0;

    // The Javascript layer assigns 'output' once while building `fetches`. If native result
    // construction assigned too, this setter would run between the validation of the preallocated
    // buffer and the copy into it, and the detach below would leave the copy writing through a
    // dead backing store. Reaching the end of this test is the assertion.
    Object.defineProperty(Object.prototype, 'output', {
      configurable: true,
      get: () => stored,
      set: (value) => {
        stored = value;
        if (++sets === 2) {
          structuredClone(outputData.buffer, { transfer: [outputData.buffer] });
        }
      },
    });

    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const output = new Tensor('float32', outputData, [1, 5]);
      const result = await localSession.run({ input }, { output });
      assert.strictEqual(result.output, output);
    } finally {
      delete (Object.prototype as { output?: unknown }).output;
      await localSession.release().catch(() => {});
    }
  });

  it('is unaffected by a poisoned Array prototype', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    // Far too small to hold the output: if an inherited accessor could intercept the binding's
    // internal pinning, this is what validation and the copy would disagree about.
    const decoy = new Float32Array(1);
    const indices = [0, 1, 2, 3, 4];
    for (const index of indices) {
      Object.defineProperty(Array.prototype, index, {
        configurable: true,
        get: () => decoy,
        set: () => {},
      });
    }

    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const output = new Tensor('float32', new Float32Array(5), [1, 5]);
      const result = await localSession.run({ input }, { output });
      assert.strictEqual(result.output, output);
      assertTensorEqual(output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));

      // A fresh output builds its dims array natively; an inherited index setter must not be able to
      // swallow those elements either.
      const allocated = (await localSession.run({ input })).output;
      assert.deepStrictEqual(allocated.dims, [1, 5]);
      assertTensorEqual(allocated, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    } finally {
      for (const index of indices) {
        delete (Array.prototype as unknown as Record<number, unknown>)[index];
      }
      await localSession.release().catch(() => {});
    }
  });

  it('keeps output data valid after the run and the session are gone', async () => {
    // Outputs are handed to Javascript as external ArrayBuffers over ORT's own memory, so the
    // OrtValue's ownership has to move to the ArrayBuffer rather than stay with the vector the run
    // held it in. If it does not, the data dies with the worker and this reads freed memory.
    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const first = await InferenceSession.create(modelPath);
    const retained: Tensor[] = [];
    // Also keep storage whose owning Tensor is dropped immediately: ORT offers no way to detach a
    // buffer from its OrtValue, so the whole native value has to stay alive for as long as any
    // JavaScript view of it does, not merely for as long as the Tensor does.
    let orphanedData: Float32Array;
    let orphanedBuffer: ArrayBufferLike;
    try {
      for (let i = 0; i < 8; i++) {
        retained.push((await first.run({ input: new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]) })).output);
      }
      orphanedData = (await first.run({ input: new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]) })).output
        .data as Float32Array;
      orphanedBuffer = (
        (await first.run({ input: new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]) })).output.data as Float32Array
      ).buffer;
    } finally {
      await first.release();
    }

    const gc = (global as unknown as { gc?: () => void }).gc;
    gc?.();

    // Churn allocations through an unrelated session, then check the retained outputs again.
    const second = await InferenceSession.create(modelPath);
    try {
      for (let i = 0; i < 50; i++) {
        await second.run({ input: new Tensor('float32', [5, 4, 3, 2, 1], [1, 5]) });
      }
    } finally {
      await second.release();
    }

    gc?.();
    for (const output of retained) {
      assertTensorEqual(output, new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]));
    }
    assert.deepStrictEqual(Array.from(orphanedData), [1, 2, 3, 4, 5]);
    assert.deepStrictEqual(Array.from(new Float32Array(orphanedBuffer)), [1, 2, 3, 4, 5]);
  });

  it('rejects preallocated string outputs', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_string.onnx'));
    const input = new Tensor('string', ['a', 'b', 'c', 'd', 'e'], [1, 5]);
    const output = new Tensor('string', ['', '', '', '', ''], [1, 5]);

    await assert.rejects(
      localSession.run({ input }, { output }),
      /Preallocated string output tensors are not supported/,
    );
    await localSession.release();
  });

  it('reuses a preallocated CPU output', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const output = new Tensor('float32', [0, 0, 0, 0, 0], [1, 5]);

      const result = await localSession.run({ input }, { output });
      assert.strictEqual(result.output, output);
      assertTensorEqual(output, input);
    } finally {
      await localSession.release().catch(() => {});
    }
  });

  it('rejects overlapping preallocated CPU outputs', async () => {
    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const localSession = await InferenceSession.create(modelPath);
    const secondSession = await InferenceSession.create(modelPath);
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const output = new Tensor('float32', [0, 0, 0, 0, 0], [1, 5]);

      const firstRun = localSession.run({ input }, { output });
      await assert.rejects(secondSession.run({ input }, { output }), /Preallocated output buffer is already in use/);
      await firstRun;
    } finally {
      await localSession.release().catch(() => {});
      await secondSession.release().catch(() => {});
    }
  });

  it('rejects a detached preallocated CPU output buffer', async () => {
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'));
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const outputData = new Float32Array(5);
      const output = new Tensor('float32', outputData, [1, 5]);

      const run = localSession.run({ input }, { output });
      structuredClone(outputData.buffer, { transfer: [outputData.buffer] });
      await assert.rejects(run, /Preallocated output tensor buffer was detached/);
    } finally {
      await localSession.release().catch(() => {});
    }
  });

  it('reuses a preallocated GPU output', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const localSession = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const secondSession = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const readbackSession = await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] });
    try {
      const preallocatedOutput = (await localSession.run({ input: new Tensor('float32', [0, 0, 0, 0, 0], [1, 5]) }))
        .output;
      const expectedOutput = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);

      const firstRun = localSession.run({ input: expectedOutput }, { output: preallocatedOutput });
      await assert.rejects(
        secondSession.run({ input: expectedOutput }, { output: preallocatedOutput }),
        /Preallocated output buffer is already in use/,
      );
      await firstRun;

      const result = await localSession.run({ input: expectedOutput }, { output: preallocatedOutput });
      assert.strictEqual(result.output, preallocatedOutput);
      assertTensorEqual((await readbackSession.run({ input: preallocatedOutput })).output, expectedOutput);

      preallocatedOutput.dispose();
    } finally {
      await localSession.release().catch(() => {});
      await secondSession.release().catch(() => {});
      await readbackSession.release().catch(() => {});
    }
  });

  it('publishes a GPU output without assigning through an inherited setter', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const localSession = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const readbackSession = await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] });

    // A gpu-buffer output reaches Tensor.fromGpuBuffer() through an options object built natively.
    // Assigning its properties would run this setter, and unwinding from it would release the
    // device value on this thread rather than through the device lock.
    Object.defineProperty(Object.prototype, 'download', {
      configurable: true,
      set: () => {
        throw new Error('inherited setter reached');
      },
    });
    try {
      assert.throws(() => {
        (({}) as { download?: unknown }).download = 1;
      }, /inherited setter reached/);

      const expectedOutput = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const result = await localSession.run({ input: expectedOutput });
      assert.strictEqual(result.output.location, 'gpu-buffer');
      assertTensorEqual((await readbackSession.run({ input: result.output })).output, expectedOutput);
      result.output.dispose();
    } finally {
      delete (Object.prototype as { download?: unknown }).download;
      await localSession.release().catch(() => {});
      await readbackSession.release().catch(() => {});
    }
  });

  it('runs concurrent inferences across device sessions safely', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    // No preferredOutputLocation, so these take the plain Run() path. Sessions on the same device
    // still share the provider's global state, so the runs have to serialize against each other.
    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const sessions = [
      await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] }),
      await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] }),
    ];
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const expected = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      for (let round = 0; round < 10; round++) {
        const results = await Promise.all(
          sessions.flatMap((session) => [session.run({ input }), session.run({ input })]),
        );
        for (const result of results) {
          assertTensorEqual(result.output, expected);
        }
      }
    } finally {
      for (const session of sessions) {
        await session.release().catch(() => {});
      }
    }
  });

  it('runs concurrent IO-binding inferences across sessions safely', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    // preferredOutputLocation selects the IO-binding path, where binding does device work outside
    // the guard ORT applies to graph execution.
    const localSession = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const readbackSession = await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] });
    const sessions = [localSession, readbackSession];
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const expected = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      // Two sessions on the default device share one WebGpuContext, and therefore one command
      // encoder, so they have to serialize against each other and not merely within a session.
      const secondSession = await InferenceSession.create(modelPath, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
      });
      sessions.push(secondSession);
      for (let round = 0; round < 5; round++) {
        const results = await Promise.all([
          ...Array.from({ length: 2 }, async () => localSession.run({ input })),
          ...Array.from({ length: 2 }, async () => secondSession.run({ input })),
        ]);
        for (const result of results) {
          assert.strictEqual(result.output.location, 'gpu-buffer');
          assertTensorEqual((await readbackSession.run({ input: result.output })).output, expected);
          result.output.dispose();
        }
      }
    } finally {
      for (const session of sessions) {
        await session.release().catch(() => {});
      }
    }
  });

  it('lets a GPU output outlive the session that produced it', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    // The buffer belongs to an allocator owned by the session's execution provider, so releasing it
    // after the session is gone would call into a destroyed provider. The value holds the session
    // alive instead, whether it is disposed late or left to be collected.
    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'), {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
    const disposedLate = (await localSession.run({ input })).output;
    // Deliberately never disposed: its finalizer has to be safe too.
    void (await localSession.run({ input })).output;

    await localSession.release();
    assert.strictEqual(disposedLate.location, 'gpu-buffer');
    disposedLate.dispose();
    (global as unknown as { gc?: () => void }).gc?.();
  });

  it('rejects a device output on a session without preferredOutputLocation', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    const modelPath = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    const deviceSession = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const plainSession = await InferenceSession.create(modelPath, { executionProviders: ['webgpu'] });
    try {
      const input = new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]);
      const deviceTensor = (await deviceSession.run({ input })).output;

      // Without IO binding this would reach ORT as a preallocated fetch it may replace instead of
      // fill, leaving the caller's buffer stale and the promise resolved.
      await assert.rejects(
        plainSession.run({ input }, { output: deviceTensor }),
        /requires the session to be created with 'preferredOutputLocation'/,
      );
      deviceTensor.dispose();
    } finally {
      await plainSession.release().catch(() => {});
      await deviceSession.release().catch(() => {});
    }
  });

  it('keeps a GPU buffer alive when Tensor disposal bypasses instance methods', async function () {
    // eslint-disable-next-line no-invalid-this
    skipWithoutWebGpu(this);

    const localSession = await InferenceSession.create(path.join(TEST_DATA_ROOT, 'test_types_float.onnx'), {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
    });
    const disposeInput = [
      (input: Tensor) => input.dispose.bind(input)(),
      (input: Tensor) => (Object.getPrototypeOf(input) as { dispose: () => void }).dispose.call(input),
    ];

    for (const dispose of disposeInput) {
      const gpuInput = (await localSession.run({ input: new Tensor('float32', [1, 2, 3, 4, 5], [1, 5]) })).output;
      assert.strictEqual(gpuInput.location, 'gpu-buffer');

      const run = localSession.run({ input: gpuInput });
      dispose(gpuInput);
      const result = await run;
      assert.strictEqual(result.output.location, 'gpu-buffer');
      result.output.dispose();
    }
    await localSession.release();
  });
});
