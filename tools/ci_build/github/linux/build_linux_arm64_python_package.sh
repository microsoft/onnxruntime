#/bin/bash
set -x
mkdir /build/dist
PYTHON_EXES=("/opt/python/cp37-cp37m/bin/python3.7" "/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10")
for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  ${PYTHON_EXE} /onnxruntime_src/tools/ci_build/build.py \
                  --build_dir /build \
                  --config Release --update --build \
                  --skip_submodule_sync \
                  --parallel \
                  --enable_lto \
                  --build_wheel \
                  --enable_onnx_tests
  cp /build/Release/dist/*.whl /build/dist
  rm -rf /build/Release
done
