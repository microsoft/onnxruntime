// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import * as path from 'path';
import { InferenceSession, Tensor } from 'onnxruntime-common';

import { binding, initOrt } from '../../../lib/binding';
import { onnx } from '../../ort-schema/protobuf/onnx';

const pluginPath = process.env.ORT_ENCRYPTION_PLUGIN_LIBRARY;

describe('Encrypted external EPContext inference', () => {
  it('compiles, encrypts, persists, decrypts and runs the restored context', async function () {
    if (!pluginPath) {
      // eslint-disable-next-line no-invalid-this
      this.skip();
    }
    // eslint-disable-next-line no-invalid-this
    this.timeout(60000);
    // eslint-disable-next-line no-underscore-dangle
    const compile = binding.__testCompileEpContextModel;
    // eslint-disable-next-line no-underscore-dangle
    const unregister = binding.__testUnregisterEpContextPlugin;
    assert(compile && unregister, 'Rebuild with ORT_NODEJS_TEST_EP_CONTEXT=ON to run this test.');
    initOrt();

    const shape = { elemType: 1, shape: { dim: [{ dimValue: 3 }, { dimValue: 2 }] } };
    const source = Buffer.from(
      onnx.ModelProto.encode({
        irVersion: 8,
        opsetImport: [{ domain: '', version: 13 }],
        graph: {
          name: 'encrypted-mul',
          node: [{ name: 'mul', opType: 'Mul', input: ['x', 'y'], output: ['z'] }],
          input: ['x', 'y'].map((name) => ({ name, type: { tensorType: shape } })),
          output: [{ name: 'z', type: { tensorType: shape } }],
        },
      }).finish(),
    );
    const registration = 'node_encryption_test';
    const directory = mkdtempSync(path.join(tmpdir(), 'ort-encrypted-context-'));
    let registered = false;
    try {
      const compiled = compile(pluginPath, registration, source);
      registered = true;
      const graph = onnx.ModelProto.decode(compiled.model).graph;
      assert(graph);
      assert(graph.node);
      assert.strictEqual(graph.node.length, 1);
      assert.strictEqual(graph.node[0].opType, 'EPContext');
      assert.strictEqual(graph.initializer?.length ?? 0, 0);
      assert.strictEqual(compiled.context.toString(), 'ort-test-mul-float32-v1');
      const modelName = 'compiled.onnx';
      const key = randomBytes(32);
      const encrypt = (plaintext: Buffer, name: string): Buffer => {
        const nonce = randomBytes(12);
        const cipher = createCipheriv('aes-256-gcm', key, nonce);
        cipher.setAAD(Buffer.from(name));
        const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
        return Buffer.concat([nonce, cipher.getAuthTag(), ciphertext]);
      };
      const decrypt = (record: Buffer, name: string, decryptionKey: Buffer = key): Buffer => {
        const decipher = createDecipheriv('aes-256-gcm', decryptionKey, record.subarray(0, 12));
        decipher.setAuthTag(record.subarray(12, 28));
        decipher.setAAD(Buffer.from(name));
        return Buffer.concat([decipher.update(record.subarray(28)), decipher.final()]);
      };
      const encryptedModel = encrypt(compiled.model, modelName);
      const encryptedContext = encrypt(compiled.context, compiled.contextName);
      const contextName = compiled.contextName;
      writeFileSync(path.join(directory, 'model.enc'), encryptedModel);
      writeFileSync(path.join(directory, 'context.enc'), encryptedContext);
      compiled.model.fill(0);
      compiled.context.fill(0);
      assert.deepStrictEqual(readdirSync(directory).sort(), ['context.enc', 'model.enc']);

      const modelRecord = readFileSync(path.join(directory, 'model.enc'));
      const contextRecord = readFileSync(path.join(directory, 'context.enc'));
      const wrongKey = Buffer.from(key);
      wrongKey[0] = (wrongKey[0] + 1) % 256;
      assert.throws(() => decrypt(modelRecord, modelName, wrongKey), /authenticate/);
      assert.throws(() => decrypt(contextRecord, contextName, wrongKey), /authenticate/);
      assert.throws(() => decrypt(contextRecord, `${contextName}.wrong`), /authenticate/);
      const tamperedModel = Buffer.from(modelRecord);
      tamperedModel[tamperedModel.length - 1] = (tamperedModel[tamperedModel.length - 1] + 1) % 256;
      assert.throws(() => decrypt(tamperedModel, modelName), /authenticate/);
      const tamperedContext = Buffer.from(contextRecord);
      tamperedContext[tamperedContext.length - 1] = (tamperedContext[tamperedContext.length - 1] + 1) % 256;
      assert.throws(() => decrypt(tamperedContext, contextName), /authenticate/);

      const model = decrypt(modelRecord, modelName);
      const options = {
        __testEpContextProvider: registration,
        epContextDataRead: {
          maxDataSize: 1024,
          callback: (name: string) => {
            assert.strictEqual(name, contextName);
            return decrypt(contextRecord, name);
          },
        },
      };
      let callbackCount = 0;
      const session = await InferenceSession.create(model, {
        ...options,
        epContextDataRead: {
          ...options.epContextDataRead,
          callback: (name: string) => {
            callbackCount++;
            return options.epContextDataRead.callback(name);
          },
        },
      });
      try {
        assert.strictEqual(callbackCount, 1);
        for (const [x, y] of [
          [
            [1, 2, 3, 4, 5, 6],
            [2, -1, 0, 3, 2, 1],
          ],
          [
            [-1, 0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0, -1],
          ],
        ]) {
          const result = await session.run({
            x: new Tensor('float32', Float32Array.from(x), [3, 2]),
            y: new Tensor('float32', Float32Array.from(y), [3, 2]),
          });
          assert(result.z.data instanceof Float32Array);
          assert.deepStrictEqual(
            Array.from(result.z.data),
            x.map((value, index) => value * y[index]),
          );
        }
      } finally {
        await session.release();
      }
      await assert.rejects(
        InferenceSession.create(model, { ...options, epContextDataRead: undefined }),
        /requires a model path/,
      );
      for (const record of [tamperedContext, contextRecord]) {
        await assert.rejects(
          InferenceSession.create(model, {
            ...options,
            epContextDataRead: {
              maxDataSize: 1024,
              callback: (name: string) => decrypt(record, name, record === contextRecord ? wrongKey : key),
            },
          }),
          /authenticate/,
        );
      }
      await assert.rejects(
        InferenceSession.create(model, {
          ...options,
          epContextDataRead: { maxDataSize: 1024, callback: () => Buffer.from('invalid compiled payload') },
        }),
        /payload/,
      );
    } finally {
      rmSync(directory, { recursive: true, force: true });
      if (registered) {
        unregister(registration);
      }
    }
  });
});
