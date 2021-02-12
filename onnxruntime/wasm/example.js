const fs = require('fs');
const process = require('process');

let path = './dist/Release/onnxruntime_wasm.js';
if (process.argv.length == 3) {
  let build_type = process.argv.slice(2)[0];
  if (build_type.toLowerCase() == 'debug') {
    path = './dist/Debug/onnxruntime_wasm.js';
  } else if (build_type.toLowerCase() == 'minsizerel') {
    path = './dist/MinSizeRel/onnxruntime_wasm.js';
  }
}

const onnxjs = require(path);

onnxjs().then((o) => {
  let example = new o.Example();
  model = fs.readFileSync('./model.onnx');
  let loaded = example.Load(model);
  if (loaded) {
    console.log("Model is successfully loaded.");
  }
  let success = example.Run();
  if (success) {
    console.log("Model is successfully executed");
  }
});
