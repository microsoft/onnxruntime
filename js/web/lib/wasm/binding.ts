
/* eslint-disable @typescript-eslint/naming-convention */
declare interface OnnxWasmBindingJs {
  (self: OnnxWasmBindingJs): Promise<void>;

  _malloc: (ptr: number) => number;
  _free: (ptr: number) => void;

  buffer: ArrayBuffer;

  HEAP8: Int8Array;
  HEAP16: Int16Array;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  HEAPU16: Uint16Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;

  stackSave(): number;
  stackRestore(stack: number): void;
  stackAlloc(size: number): number;

  UTF8ToString(offset: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, offset: number, maxBytes: number): void;

  _OrtInit(): void;

  _OrtCreateSession(dataOffset: number, dataLength: number): number;
  _OrtReleaseSession(sessionHandle: number): void;
  _OrtGetInputCount(sessionHandle: number): number;
  _OrtGetOutputCount(sessionHandle: number): number;
  _OrtGetInputName(sessionHandle: number, index: number): number;
  _OrtGetOutputName(sessionHandle: number, index: number): number;

  _OrtFree(stringHandle: number): void;

  _OrtCreateTensor(dataType: number, dataOffset: number, dataLength: number, dimsOffset: number, dimsLength: number):
      number;
  _OrtGetTensorData(tensorHandle: number, dataType: number, dataOffset: number, dimsOffset: number, dimsLength: number):
      void;
  _OrtReleaseTensor(tensorHandle: number): void;
  _OrtRun(
      sessionHandle: number, inputNamesOffset: number, inputsOffset: number, inputCount: number,
      outputNamesOffset: number, outputCount: number, outputsOffset: number): number;
}
/* eslint-enable @typescript-eslint/naming-convention */

// some global parameters to deal with wasm binding initialization
let binding: OnnxWasmBindingJs|undefined;
let initialized = false;
let initializing = false;

/**
 * initialize the WASM instance.
 *
 * this function should be called before any other calls to the WASM binding.
 */
export const init = async(): Promise<void> => {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error('multiple calls to \'init()\' detected.');
  }

  initializing = true;

  return new Promise<void>((resolve, reject) => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
    binding = require('../../dist/onnxruntime_wasm') as OnnxWasmBindingJs;
    binding(binding).then(
        () => {
          // resolve init() promise
          resolve();
          initializing = false;
          initialized = true;
        },
        err => {
          initializing = false;
          reject(err);
        });
  });
};

export const getInstance = (): OnnxWasmBindingJs => binding!;
