CONFIG=Debug
TARGET_FILE_PREFIX=ort-wasm-simd
ROOT=$(pwd)
BUILD_DIR=$ROOT/build
pushd "$ROOT"/js || exit
npm ci 
popd || exit
pushd "$ROOT"/js/common || exit
npm ci
popd || exit
pushd "$ROOT"/js/web || exit
npm ci
npm run pull:wasm
popd || exit
./build.sh  --skip_submodule_sync --build_wasm --target onnxruntime_webassembly --use_jsep --parallel --skip_tests --path_to_protoc_exe /usr/local/bin/protoc --enable_wasm_simd --config $CONFIG
if [ $? -eq 1 ]
then
  exit
fi
cp "$BUILD_DIR"/MacOS/$CONFIG/$TARGET_FILE_PREFIX.js "$ROOT"/js/web/lib/wasm/binding/$TARGET_FILE_PREFIX.jsep.js
cp "$BUILD_DIR"/MacOS/$CONFIG/$TARGET_FILE_PREFIX.wasm "$ROOT"/js/web/dist/$TARGET_FILE_PREFIX.jsep.wasm
pushd "$ROOT"/js/web || exit
# Test
npm test -- op fused-conv.jsonc  -b=webgpu --wasm-number-threads 1 --debug --log-verbose
#npm test -- op data/ops/add.jsonc
popd || exit