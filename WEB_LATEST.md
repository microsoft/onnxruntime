### Pipelines

- CI:
- Packaging: https://aiinfra.visualstudio.com/Lotus/_build?definitionId=1080

### Artifacts

- ort-wasm-simd-threaded.mjs
  ```
  --parallel --config Release --skip_submodule_sync --build_wasm --enable_wasm_simd --enable_wasm_threads --target onnxruntime_webassembly --skip_tests --disable_rtti --build_dir /mnt/vss/_work/1/b/wasm_inferencing --enable_wasm_api_exception_catching --wasm_run_tests_in_browser
  ```

- ort-wasm-simd-threaded.asyncify.mjs
  ```
  --parallel --config Release --skip_submodule_sync --build_wasm --enable_wasm_simd --enable_wasm_threads --target onnxruntime_webassembly --skip_tests --disable_rtti --build_dir /mnt/vss/_work/1/b/wasm_inferencing_webgpu --use_webgpu --use_webnn --target onnxruntime_webassembly --enable_wasm_api_exception_catching --skip_tests
  ```

- ort-wasm-simd-threaded.jspi.mjs
  ```
  --parallel --config Release --skip_submodule_sync --build_wasm --enable_wasm_simd --enable_wasm_threads --target onnxruntime_webassembly --skip_tests --disable_rtti --build_dir /mnt/vss/_work/1/b/wasm_inferencing_webgpu_jspi --use_webgpu --use_webnn --enable_wasm_jspi --target onnxruntime_webassembly --skip_tests
  ```
