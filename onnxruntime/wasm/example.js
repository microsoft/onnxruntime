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

function generateRandomData(count) {
  const r = new Array(count);
  for (let i = 0; i < count; i++) {
    r[i] = new Float32Array(1 * 4 * 96 * 160);
    for (let j = 0; j < r[i].length; j++) {
      r[i][j] = Math.random();
    }
  }
  return r;
}
//window.r = generateRandomData(50);


onnxjs().then((o) => {
  const model = fs.readFileSync('D:\\eire\\git\\pg\\msra_190729.onnx');

  console.log('init()');
  o._ort_init();

  console.log('allocating...');
  const offset_model = o._malloc(model.byteLength + 4);
  o.HEAP32[offset_model / 4] = model.byteLength; // byte 0 ~ 3 : size of model data in bytes
  o.HEAPU8.set(model, offset_model + 4);         // byte 4 ~ : data
  console.log('create session');
  const sessionHandle = o._ort_create_session(offset_model);
  console.log('session handle=' + sessionHandle);
  o._free(offset_model);

  // Create 1 input == start ==
  const inputs = generateRandomData(10)[0];

  const dims = [1, 4, 96, 160];

  const data_offset = o._malloc(inputs.byteLength);
  o.HEAPU8.set(new Uint8Array(inputs.buffer, inputs.byteOffset, inputs.byteLength), data_offset);

  const tensor_creation_struct_len = (4 + dims.length) * 4;
  const tensor_creation_struct_offset = o._malloc(tensor_creation_struct_len);
  let e = tensor_creation_struct_offset / 4;
  o.HEAP32[e++] = 1;                      // byte 0 ~ 3 : data type
  o.HEAPU32[e++] = inputs.byteLength;     // byte 4 ~ 7 : data size
  o.HEAPU32[e++] = data_offset;           // byte 8 ~ 11 : data (pointer)
  o.HEAPU32[e++] = dims.length;           // byte 12 ~ 15 : dims length
  dims.forEach(d => o.HEAP32[e++] = d);   // byte 16~ : dims

  const input_tensor = o._ort_create_tensor(tensor_creation_struct_offset);
  o._free(tensor_creation_struct_offset);
  // Create 1 input == finish ==

  // run
  const input_count = 1;
  const output_count = 1;
  const run_context_offset = o._malloc(8 + (input_count + output_count) * 8);
  e = run_context_offset / 4;
  o.HEAP32[e++] = input_count;                      // byte 0 ~ 3 : input_count
  o.HEAP32[e++] = output_count;                     // byte 4 ~ 7 : output_count

  const input_name_lengthBytes = o.lengthBytesUTF8("bgrm") + 1;
  const input_name_offset = o._malloc(input_name_lengthBytes);
  o.stringToUTF8("bgrm", input_name_offset, input_name_lengthBytes);
  o.HEAPU32[e++] = input_name_offset;                     // byte 8 ~ 11 : input[0].name
  o.HEAPU32[e++] = input_tensor;                          // byte 12 ~ 15 : input[0].tensor

  const output_name_lengthBytes = o.lengthBytesUTF8("mask") + 1;
  const output_name_offset = o._malloc(output_name_lengthBytes);
  o.stringToUTF8("mask", output_name_offset, output_name_lengthBytes);
  o.HEAPU32[e++] = output_name_offset;                     // byte 16 ~ 19 : output[0].name
  o.HEAPU32[e++] = 0;                                      // byte 20 ~ 23 : output[0].tensor

  o._ort_run(sessionHandle, run_context_offset);

  e = run_context_offset / 4 + 2 + 2 * input_count;
  const output_tensor = o.HEAPU32[e + 1];

  const output_tensor_metadata = o._ort_get_tensor_metadata(output_tensor);

  e = output_tensor_metadata / 4;
  const output_tensor_type = o.HEAPU32[e++];
  const output_tensor_data_size = o.HEAPU32[e++];
  const output_tensor_data_offset = o.HEAPU32[e++];
  const output_tensor_dims_len = o.HEAPU32[e++];
  const output_tensor_dims = [];
  for (let i = 0; i < output_tensor_dims_len; i++) {
    output_tensor_dims.push(o.HEAPU32[e++]);
  }


  o._ort_release_session(sessionHandle);
  console.log('session released');
});
