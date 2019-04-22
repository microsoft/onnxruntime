# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import shutil

import onnx
import onnxruntime
import json

from google.protobuf.json_format import MessageToJson

import predict_pb2
import onnx_ml_pb2

# Current models only have one input and one output
def get_io_name(model_file_name):
  sess = onnxruntime.InferenceSession(model_file_name)
  return sess.get_inputs()[0].name, sess.get_outputs()[0].name


def gen_input_pb(pb_full_path, input_name, output_name, request_file_path):
  t = onnx_ml_pb2.TensorProto()
  with open(pb_full_path, 'rb') as fin:
      t.ParseFromString(fin.read())
  predict_request = predict_pb2.PredictRequest()
  predict_request.inputs[input_name].CopyFrom(t)
  predict_request.output_filter.append(output_name)

  with open(request_file_path, "wb") as fout:
      fout.write(predict_request.SerializeToString())


def gen_output_pb(pb_full_path, output_name, response_file_path):
  t = onnx_ml_pb2.TensorProto()
  with open(pb_full_path, 'rb') as fin:
      t.ParseFromString(fin.read())
  predict_response = predict_pb2.PredictResponse()
  predict_response.outputs[output_name].CopyFrom(t)

  with open(response_file_path, "wb") as fout:
      fout.write(predict_response.SerializeToString())


def tensor2dict(full_path):
  t = onnx.TensorProto()
  with open(full_path, 'rb') as f:
    t.ParseFromString(f.read())

  jsonStr = MessageToJson(t, use_integers_for_enums=True)
  data = json.loads(jsonStr)

  return data


def gen_input_json(pb_full_path, input_name, output_name, json_file_path):
  data = tensor2dict(pb_full_path)

  inputs = {}
  inputs[input_name] = data
  output_filters = [ output_name ]

  req = {}
  req["inputs"] = inputs
  req["outputFilter"] = output_filters

  with open(json_file_path, 'w') as outfile:
    json.dump(req, outfile)


def gen_output_json(pb_full_path, output_name, json_file_path):
  data = tensor2dict(pb_full_path)

  output = {}
  output[output_name] = data

  resp = {}
  resp["outputs"] = output

  with open(json_file_path, 'w') as outfile:
    json.dump(resp, outfile)


def gen_req_resp(model_zoo, test_data, copy_model=True):
  opsets = [name for name in os.listdir(model_zoo) if os.path.isdir(os.path.join(model_zoo, name))]
  for opset in opsets:
    os.makedirs(os.path.join(test_data, opset), exist_ok=True)

    current_model_folder = os.path.join(model_zoo, opset)
    current_data_folder = os.path.join(test_data, opset)

    models = [name for name in os.listdir(current_model_folder) if os.path.isdir(os.path.join(current_model_folder, name))]
    for model in models:
      os.makedirs(os.path.join(current_data_folder, model), exist_ok=True)

      src_folder = os.path.join(current_model_folder, model)
      dst_folder = os.path.join(current_data_folder, model)

      if copy_model:
        shutil.copy2(os.path.join(src_folder, 'model.onnx'), dst_folder)

      iname, oname = get_io_name(os.path.join(src_folder, 'model.onnx'))
      model_test_data = [name for name in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, name))]
      for test in model_test_data:
        src = os.path.join(src_folder, test)
        dst = os.path.join(dst_folder, test)
        os.makedirs(dst, exist_ok=True)
        gen_input_json(os.path.join(src, 'input_0.pb'), iname, oname, os.path.join(dst, 'request.json'))
        gen_output_json(os.path.join(src, 'output_0.pb'), oname, os.path.join(dst, 'response.json'))
        gen_input_pb(os.path.join(src, 'input_0.pb'), iname, oname, os.path.join(dst, 'request.pb'))
        gen_output_pb(os.path.join(src, 'output_0.pb'), oname, os.path.join(dst, 'response.pb'))


if __name__ == '__main__':
  model_zoo = os.path.realpath(sys.argv[1])
  test_data = os.path.realpath(sys.argv[2])

  os.makedirs(test_data, exist_ok=True)
  gen_req_resp(model_zoo, test_data)
