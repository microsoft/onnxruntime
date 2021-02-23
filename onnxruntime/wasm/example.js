const fs = require('fs');
const process = require('process');

if (process.argv.length < 3) {
  console.log("Usage: node example.js model [release|debug|minsizerel]");
  process.exit(1);
}

let path = './dist/Release/onnxruntime_wasm.js';
if (process.argv.length == 4) {
  let build_type = process.argv[3];
  if (build_type.toLowerCase() == 'debug') {
    path = './dist/Debug/onnxruntime_wasm.js';
  } else if (build_type.toLowerCase() == 'minsizerel') {
    path = './dist/MinSizeRel/onnxruntime_wasm.js';
  }
}

const onnxjs = require(path);

onnxjs().then((o) => {
  let example = new o.Example();
  model = fs.readFileSync(process.argv[2]);
  let loaded = example.Load(model);
  if (loaded) {
    console.log("Model is successfully loaded.");
  }
  let success = example.Run();
  if (success) {
    console.log("Model is successfully executed");
  }
});
