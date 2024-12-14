const ep = process.argv[2] || 'cpu';

(async function () {
    const ort = require('.')
    console.log(ort.listSupportedBackends())
    const options = {executionProviders:[ep]}
    console.log(options)
const t2 = await ort.InferenceSession.create(
    'c:/code/onnxruntime/js/web/test/e2e/model.onnx',
    options)
console.log(t2)
})();
