import * as ort from 'onnxruntime-web';

// The following line uses Vite's "Explicit URL Imports" feature to load the wasm file as an asset.
//
// see https://vite.dev/guide/assets.html#explicit-url-imports
//
import wasmFileUrl from '/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';

// wasmFileUrl is the URL of the wasm file. Vite will make sure it's available in both development and production.
ort.env.wasm.wasmPaths = { wasm: wasmFileUrl };

// Model data for "test_abs/model.onnx"
const testModelData =
  'CAcSDGJhY2tlbmQtdGVzdDpJCgsKAXgSAXkiA0FicxIIdGVzdF9hYnNaFwoBeBISChAIARIMCgIIAwoCCAQKAggFYhcKAXkSEgoQCAESDAoCCAMKAggECgIIBUIECgAQDQ==';

const base64StringToUint8Array = (base64String) => {
  const charArray = atob(base64String);
  const length = charArray.length;
  const buffer = new Uint8Array(new ArrayBuffer(length));
  for (let i = 0; i < length; i++) {
    buffer[i] = charArray.charCodeAt(i);
  }
  return buffer;
};

let mySession;

const assert = (cond, msg) => {
  if (!cond) throw new Error(msg);
};

export const createTestSession = async (multiThreaded, proxy) => {
  const model = base64StringToUint8Array(testModelData);
  const options = {};

  if (multiThreaded) {
    ort.env.wasm.numThreads = 2;
    assert(typeof SharedArrayBuffer !== 'undefined', 'SharedArrayBuffer is not supported');
  }
  if (proxy) {
    ort.env.wasm.proxy = true;
  }
  mySession = await ort.InferenceSession.create(model, options);
};

export const runTestSessionAndValidate = async () => {
  try {
    // test data: [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, ... 58, -59]
    const inputData = [...Array(60).keys()].map((i) => (i % 2 === 0 ? i : -i));
    const expectedOutputData = inputData.map((i) => Math.abs(i));

    const fetches = await mySession.run({ x: new ort.Tensor('float32', inputData, [3, 4, 5]) });

    const y = fetches.y;

    assert(y instanceof ort.Tensor, 'unexpected result');
    assert(y.dims.length === 3 && y.dims[0] === 3 && y.dims[1] === 4 && y.dims[2] === 5, 'incorrect shape');

    for (let i = 0; i < expectedOutputData.length; i++) {
      assert(y.data[i] === expectedOutputData[i], `output data mismatch at index ${i}`);
    }

    return 'PASS';
  } catch (e) {
    return `FAIL: ${e}`;
  }
};
