import * as fs from 'fs-extra';
import * as onnx_proto from 'onnx-proto';
import * as path from 'path';

import {InferenceSession, Tensor} from '../lib';
import {assertTensorEqual} from './test-utils';

export function run(testDataFolder: string) {
  const models = fs.readdirSync(testDataFolder);

  for (const model of models) {
    // read each model folders
    const modelFolder = path.join(testDataFolder, model);
    let modelPath: string;
    const modelTestCases: Array<[(Tensor | undefined)[], (Tensor | undefined)[]]> = [];
    for (const currentFile of fs.readdirSync(modelFolder)) {
      const currentPath = path.join(modelFolder, currentFile);
      const stat = fs.lstatSync(currentPath);
      if (stat.isFile()) {
        const ext = path.extname(currentPath);
        if (ext.toLowerCase() === '.onnx') {
          modelPath = currentPath;
        }
      } else if (stat.isDirectory()) {
        const inputs: (Tensor|undefined)[] = [];
        const outputs: (Tensor|undefined)[] = [];
        for (const dataFile of fs.readdirSync(currentPath)) {
          const dataFileFullPath = path.join(currentPath, dataFile);
          const ext = path.extname(dataFile);

          if (ext.toLowerCase() === '.pb') {
            let tensor: Tensor|undefined = undefined;
            try {
              tensor = loadTensorFromFile(dataFileFullPath);
            } catch (e) {
              console.warn(`[${model}] Failed to load test data: ${e.message}`);
            }

            if (dataFile.indexOf('input') !== -1) {
              inputs.push(tensor);
            } else if (dataFile.indexOf('output') !== -1) {
              outputs.push(tensor);
            }
          }
        }
        modelTestCases.push([inputs, outputs]);
      }
    }

    // add cases
    describe(`${model}`, () => {
      let session: InferenceSession;
      const skipModel = [
        // skip list

        // failed tests
        'test_unique_not_sorted_without_axis',  // bad expected data. enable after
                                                // https://github.com/onnx/onnx/pull/2381 is picked up
        'test_mod_float_mixed_sign_example',  // onnxruntime::Mod::Compute fmod_ was false. fmod attribute must be true
                                              // for float, float16 and double types
        'test_resize_downsample_scales_cubic_align_corners',                // results mismatch with onnx tests
        'test_resize_downsample_scales_linear_align_corners',               // results mismatch with onnx tests
        'test_resize_tf_crop_and_resize',                                   // bad expected data, needs test fix
        'test_resize_upsample_sizes_nearest_ceil_half_pixel',               // bad expected data, needs test fix
        'test_resize_upsample_sizes_nearest_floor_align_corners',           // bad expected data, needs test fix
        'test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric',  // bad expected data, needs test fix
        'test_maxunpool_export_with_output_shape',                          // Invalid output in ONNX test. See
                                                                            // https://github.com/onnx/onnx/issues/2398'

        // pre opset7
        'test_AvgPool1d', 'test_AvgPool1d_stride', 'test_AvgPool2d', 'test_AvgPool2d_stride', 'test_AvgPool3d',
        'test_AvgPool3d_stride1_pad0_gpu_input', 'test_AvgPool3d_stride', 'test_BatchNorm1d_3d_input_eval',
        'test_BatchNorm2d_eval', 'test_BatchNorm2d_momentum_eval', 'test_BatchNorm3d_eval',
        'test_BatchNorm3d_momentum_eval', 'test_GLU', 'test_GLU_dim', 'test_Linear', 'test_PReLU_1d',
        'test_PReLU_1d_multiparam', 'test_PReLU_2d', 'test_PReLU_2d_multiparam', 'test_PReLU_3d',
        'test_PReLU_3d_multiparam', 'test_PoissonNLLLLoss_no_reduce', 'test_Softsign', 'test_operator_add_broadcast',
        'test_operator_add_size1_broadcast', 'test_operator_add_size1_right_broadcast',
        'test_operator_add_size1_singleton_broadcast', 'test_operator_addconstant', 'test_operator_addmm',
        'test_operator_basic', 'test_operator_mm', 'test_operator_non_float_params', 'test_operator_params',
        'test_operator_pow',

        // ConvTransponse supports 4-D only
        'test_convtranspose_1d', 'test_convtranspose_3d',

        // permanently failures
        'test_cast_FLOAT_to_STRING',

        // disabled_due_to_binary_size_concerns
        'test_bitshift_right_uint16', 'test_bitshift_left_uint16'

      ].indexOf(model) !== -1;

      if (!skipModel) {
        before(async () => {
          session = await InferenceSession.create(modelPath);
        });
      }

      // if ('test_strnormalizer_export_monday_casesensintive_lower' !== model) return;

      for (let i = 0; i < modelTestCases.length; i++) {
        const testCase = modelTestCases[i];
        const inputs = testCase[0];
        const expectedOutputs = testCase[1];
        const skip = skipModel || inputs.some(t => t === undefined) || expectedOutputs.some(t => t === undefined);
        (skip ? it.skip : it)(`case${i}`, async () => {
          const feeds = {};
          if (inputs.length !== session.inputNames.length) {
            throw new RangeError('input length does not match name list');
          }
          for (let i = 0; i < inputs.length; i++) {
            feeds[session.inputNames[i]] = inputs[i];
          }
          const outputs = await session.run(feeds);

          let j = 0;
          for (const name of session.outputNames) {
            assertTensorEqual(outputs[name], expectedOutputs[j++]!);
          }
        });
      }
    });

    // break;
  }
}

function loadTensorFromFile(pbFile: string): Tensor {
  const tensorProto = onnx_proto.onnx.TensorProto.decode(fs.readFileSync(pbFile));
  let transferredTypedArray: Tensor.DataType;
  let type: Tensor.Type;
  const dims = tensorProto.dims.map((dim) => typeof dim === 'number' ? dim : dim.toNumber());


  if (tensorProto.dataType === 8) {  // string
    return new Tensor('string', tensorProto.stringData.map(i => i.toString()), dims);
  } else {
    switch (tensorProto.dataType) {
        //     FLOAT = 1,
        //     UINT8 = 2,
        //     INT8 = 3,
        //     UINT16 = 4,
        //     INT16 = 5,
        //     INT32 = 6,
        //     INT64 = 7,
        //     STRING = 8,
        //     BOOL = 9,
        //     FLOAT16 = 10,
        //     DOUBLE = 11,
        //     UINT32 = 12,
        //     UINT64 = 13,
      case onnx_proto.onnx.TensorProto.DataType.FLOAT:
        transferredTypedArray = new Float32Array(tensorProto.rawData.byteLength / 4);
        type = 'float32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT8:
        transferredTypedArray = new Uint8Array(tensorProto.rawData.byteLength);
        type = 'uint8';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT8:
        transferredTypedArray = new Int8Array(tensorProto.rawData.byteLength);
        type = 'int8';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT16:
        transferredTypedArray = new Uint16Array(tensorProto.rawData.byteLength / 2);
        type = 'uint16';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT16:
        transferredTypedArray = new Int16Array(tensorProto.rawData.byteLength / 2);
        type = 'int16';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT32:
        transferredTypedArray = new Int32Array(tensorProto.rawData.byteLength / 4);
        type = 'int32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.INT64:
        transferredTypedArray = new BigInt64Array(tensorProto.rawData.byteLength / 8);
        type = 'int64';
        break;
      case onnx_proto.onnx.TensorProto.DataType.BOOL:
        transferredTypedArray = new Uint8Array(tensorProto.rawData.byteLength);
        type = 'bool';
        break;
      case onnx_proto.onnx.TensorProto.DataType.DOUBLE:
        transferredTypedArray = new Float64Array(tensorProto.rawData.byteLength / 8);
        type = 'float64';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT32:
        transferredTypedArray = new Uint32Array(tensorProto.rawData.byteLength / 4);
        type = 'uint32';
        break;
      case onnx_proto.onnx.TensorProto.DataType.UINT64:
        transferredTypedArray = new BigUint64Array(tensorProto.rawData.byteLength / 8);
        type = 'uint64';
        break;
      default:
        throw new Error(`not supported tensor type: ${tensorProto.dataType}`);
    }
    const transferredTypedArrayRawDataView =
        new Uint8Array(transferredTypedArray.buffer, transferredTypedArray.byteOffset, tensorProto.rawData.byteLength);
    transferredTypedArrayRawDataView.set(tensorProto.rawData);

    return new Tensor(type, transferredTypedArray, dims);
  }
}
