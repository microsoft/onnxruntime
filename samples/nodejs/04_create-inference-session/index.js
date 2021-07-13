const fs = require('fs');
const util = require('util');
const InferenceSession = require('onnxruntime').InferenceSession;

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // session options: please refer to the other example for details usage for session options
        const options = { intraOpNumThreads: 1 };

        //
        // create inference session from a ONNX model file path
        //
        const session01 = await InferenceSession.create('./model.onnx');
        const session01_B = await InferenceSession.create('./model.onnx', options); // specify options

        //
        // create inference session from an Node.js Buffer (Uint8Array)
        //
        const buffer02 = await util.promisify(fs.readFile)('./model.onnx'); // buffer is Uint8Array
        const session02 = await InferenceSession.create(buffer02);
        const session02_B = await InferenceSession.create(buffer02, options); // specify options

        //
        // create inference session from an ArrayBuffer
        //
        const arrayBuffer03 = buffer02.buffer;
        const offset03 = buffer02.byteOffset;
        const length03 = buffer02.byteLength;
        const session03 = await InferenceSession.create(arrayBuffer03, offset03, length03);
        const session03_B = await InferenceSession.create(arrayBuffer03, offset03, length03); // specify options
    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();
