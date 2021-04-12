import wasmModuleFactory, {BackendWasmModule} from './onnxruntime_wasm';

// some global parameters to deal with wasm binding initialization
let binding: BackendWasmModule|undefined;
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
    wasmModuleFactory().then(
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

export const getInstance = (): BackendWasmModule => binding!;
