const onnxjs = require('./dist/Debug/onnxruntime_wasm.js');
const fs = require('fs');

onnxjs().then((o) => {
  let example = new o.Example();
  model = fs.readFileSync('./model.onnx');
  example.Run(model);
});
