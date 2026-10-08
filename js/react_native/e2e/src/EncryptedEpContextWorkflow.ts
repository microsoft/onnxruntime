// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line import/no-internal-modules -- AES is the package's documented public subpath.
import { gcm } from '@noble/ciphers/aes';
import { Buffer } from 'buffer';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

interface TestApi {
  // eslint-disable-next-line @typescript-eslint/naming-convention
  __testCompileEpContextModel(
    plugin: string,
    registration: string,
    sourceModel: string,
    outputDirectory: string,
  ): { model: Uint8Array; context: Uint8Array; contextName: string; key: Uint8Array };
  // eslint-disable-next-line @typescript-eslint/naming-convention
  __testUnregisterEpContextPlugin(registration: string): void;
}

export const runEncryptedEpContextWorkflow = async (): Promise<string> => {
  // This explicit configuration is provisioned by the device test runner, not by production applications.
  const documentConfig = `${RNFS.DocumentDirectoryPath}/ort-encryption-config.json`;
  const bundled = !(await RNFS.exists(documentConfig));
  const bundleDirectory = `${RNFS.MainBundlePath}/ort-encryption`;
  const config: unknown = JSON.parse(
    await RNFS.readFile(bundled ? `${bundleDirectory}/config.json` : documentConfig, 'utf8'),
  );
  if (
    !config ||
    typeof config !== 'object' ||
    !('plugin' in config) ||
    typeof config.plugin !== 'string' ||
    !('sourceModel' in config) ||
    typeof config.sourceModel !== 'string'
  ) {
    throw new Error('Encryption test configuration requires plugin and sourceModel paths');
  }
  if (
    bundled &&
    [config.plugin, config.sourceModel].some((path) => path === '.' || path === '..' || !/^[a-zA-Z0-9_.-]+$/.test(path))
  ) {
    throw new Error('Bundled encryption fixture paths must be filenames');
  }
  const plugin = bundled ? `${bundleDirectory}/${config.plugin}` : config.plugin;
  const sourceModel = bundled ? `${bundleDirectory}/${config.sourceModel}` : config.sourceModel;
  const api = globalThis.OrtApi as typeof globalThis.OrtApi & Partial<TestApi>;
  // eslint-disable-next-line no-underscore-dangle
  const compile = api?.__testCompileEpContextModel;
  // eslint-disable-next-line no-underscore-dangle
  const unregister = api?.__testUnregisterEpContextPlugin;
  if (!compile || !unregister) {
    throw new Error('Rebuild the mobile binding with ORT_RN_TEST_EP_CONTEXT=1');
  }
  // Initialize the real language binding before using its native fixture environment.
  const warmup = await InferenceSession.create(sourceModel);
  await warmup.release();
  const directory = `${RNFS.DocumentDirectoryPath}/ort-encryption-${Date.now()}`;
  await RNFS.mkdir(directory);
  const registration = 'rn_encryption_test';
  let registered = false;
  try {
    const compiled = compile(plugin, registration, sourceModel, directory);
    registered = true;
    // A new native-generated key per compilation permits distinct per-asset counter nonces.
    const key = compiled.key;
    const assetName = (name: string) => Uint8Array.from(Buffer.from(name, 'utf8'));
    const encrypt = (bytes: Uint8Array, name: string, nonceId: number) => {
      const nonce = new Uint8Array(12);
      nonce[11] = nonceId;
      return Buffer.concat([Buffer.from(nonce), Buffer.from(gcm(key, nonce, assetName(name)).encrypt(bytes))]);
    };
    const decrypt = (bytes: Uint8Array, name: string, decryptionKey = key) =>
      gcm(decryptionKey, bytes.subarray(0, 12), assetName(name)).decrypt(bytes.subarray(12));
    const modelName = 'compiled.onnx';
    const contextName = compiled.contextName;
    const modelPath = `${directory}/model.enc`;
    const contextPath = `${directory}/context.enc`;
    await RNFS.writeFile(modelPath, encrypt(compiled.model, modelName, 1).toString('base64'), 'base64');
    await RNFS.writeFile(contextPath, encrypt(compiled.context, contextName, 2).toString('base64'), 'base64');
    compiled.model.fill(0);
    compiled.context.fill(0);
    const files = await RNFS.readDir(directory);
    if (files.length !== 2 || files.some((file) => !file.name.endsWith('.enc'))) {
      throw new Error('Compilation/persistence unexpectedly wrote plaintext files');
    }
    const modelRecord = Uint8Array.from(Buffer.from(await RNFS.readFile(modelPath, 'base64'), 'base64'));
    const contextRecord = Uint8Array.from(Buffer.from(await RNFS.readFile(contextPath, 'base64'), 'base64'));
    const expectAuthenticationFailure = (record: Uint8Array, name: string, readKey = key) => {
      try {
        decrypt(record, name, readKey);
      } catch (error) {
        if (/tag|auth/i.test(String(error))) {
          return;
        }
        throw error;
      }
      throw new Error('Unauthenticated ciphertext was accepted');
    };
    const wrongKey = key.slice();
    wrongKey[0] = (wrongKey[0] + 1) % 256;
    expectAuthenticationFailure(modelRecord, modelName, wrongKey);
    expectAuthenticationFailure(contextRecord, contextName, wrongKey);
    expectAuthenticationFailure(contextRecord, `${contextName}.wrong`);
    const tamper = (record: Uint8Array) => {
      const changed = record.slice();
      changed[changed.length - 1] = (changed[changed.length - 1] + 1) % 256;
      return changed;
    };
    expectAuthenticationFailure(tamper(modelRecord), modelName);
    expectAuthenticationFailure(tamper(contextRecord), contextName);
    const model = decrypt(modelRecord, modelName);
    let calls = 0;
    const options = {
      // eslint-disable-next-line @typescript-eslint/naming-convention -- Matches the gated native fixture option.
      __testEpContextProvider: registration,
      epContextDataRead: {
        maxDataSize: 1024,
        callback: (name: string) => {
          if (name !== contextName) {
            throw new Error(`Unexpected context name: ${name}`);
          }
          calls++;
          return decrypt(contextRecord, name);
        },
      },
    };
    const session = await InferenceSession.create(model, options);
    try {
      if (calls !== 1) {
        throw new Error(`Expected one real context read, received ${calls}`);
      }
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
        if (!(result.z.data instanceof Float32Array) || result.z.data.length !== 6) {
          throw new Error('Restored EP returned an invalid tensor');
        }
        for (let index = 0; index < 6; index++) {
          if (result.z.data[index] !== x[index] * y[index]) {
            throw new Error(`Restored context numerical mismatch at ${index}`);
          }
        }
      }
    } finally {
      await session.release();
    }
    const rejectLoad = async (callback: ((name: string) => Uint8Array) | undefined, message: RegExp) => {
      let loaded: InferenceSession | undefined;
      try {
        loaded = await InferenceSession.create(model, {
          ...options,
          epContextDataRead: callback ? { maxDataSize: 1024, callback } : undefined,
        });
      } catch (error) {
        if (message.test(String(error))) {
          return;
        }
        throw error;
      } finally {
        await loaded?.release();
      }
      throw new Error('Invalid context unexpectedly loaded');
    };
    await rejectLoad(undefined, /requires a model path/);
    await rejectLoad((name) => decrypt(tamper(contextRecord), name), /tag|auth/i);
    await rejectLoad((name) => decrypt(contextRecord, name, wrongKey), /tag|auth/i);
    await rejectLoad(() => Uint8Array.from(Buffer.from('invalid compiled payload')), /payload/);
    return 'Encrypted compiled model/context: two inference cases and all negative controls passed';
  } finally {
    try {
      if (registered) {
        unregister(registration);
      }
    } finally {
      await RNFS.unlink(directory);
    }
  }
};
