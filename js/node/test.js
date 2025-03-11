const ort = require('.');

async function main() {
  const session0 = await ort.InferenceSession.create('D:\\pg\\2025-02-18\\mobilenetv2-12.onnx');
  console.log(session0.inputNames);
  console.log(session0.outputNames);
  console.log(session0.inputMetadata);
  console.log(session0.outputMetadata);

  const session1 = await ort.InferenceSession.create(
    'C:\\Users\\fs\\Downloads\\op_test_generated_model_abs_no_shape.onnx',
  );
  console.log(session1.inputNames);
  console.log(session1.outputNames);
  console.log(session1.inputMetadata);
  console.log(session1.outputMetadata);

  const session2 = await ort.InferenceSession.create(
    'C:\\Users\\fs\\Downloads\\op_test_generated_model_abs_symbol.onnx',
  );
  console.log(session2.inputNames);
  console.log(session2.outputNames);
  console.log(session2.inputMetadata);
  console.log(session2.outputMetadata);

  const session3 = await ort.InferenceSession.create(
    'C:\\Users\\fs\\Downloads\\op_test_generated_model_abs_fixed.onnx',
  );
  console.log(session3.inputNames);
  console.log(session3.outputNames);
  console.log(session3.inputMetadata);
  console.log(session3.outputMetadata);
}

main();
