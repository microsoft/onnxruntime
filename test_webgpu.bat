rem @echo off
:: if file js\test\data\node\__generated_onnx_node_tests not found, generate it
if not exist "%~dp0js\test\data\node\__generated_onnx_node_tests" (
  pushd "%~dp0js"
  call npm ci
  call npm run prepare-node-tests
  popd
)

for /F "tokens=*" %%A in (%~dp0test_webgpu_cases.txt) do (
  echo %%A
)
