const ort = require('.');

async function main() {
  const session2 = await ort.InferenceSession.create(
    'op_test_generated_model_abs_symbol.onnx', {
      executionProviders: [process.argv[2]],
    }
  );
  console.log(session2.inputNames);
  console.log(session2.outputNames);


  const input_0 = new ort.Tensor('float32', [1, -2, 3, -4, 5, -6, 7, -8], [2, 4]);
  const {output_0} = await session2.run({input_0});

  console.log(output_0.data);
}

main();
