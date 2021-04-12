#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
# This tool generates a tiny GPT2 model for testing fusion script.
# You can use benchmark_gpt2.py to get a gpt2 ONNX model as input of this tool.

import onnx
import onnx.utils
import sys
import argparse
import numpy as np
from onnx import ModelProto, TensorProto, numpy_helper
from onnxruntime_tools.transformers.onnx_model import OnnxModel
import os
import onnxruntime
import random
from pathlib import Path
import timeit

DICT_SIZE = 20
SEQ_LEN = 2
""" This class creates a tiny bert model for test purpose. """

# parameters of input base model.
old_parameters = {
    "seq_len": 5,
    "hidden_size": 768,
    "num_heads": 12,
    "size_per_head": 64,
    "word_dict_size": [50257],  # list of supported dictionary size.
    "max_word_position": 1024
}

# parameters of output tiny model.
new_parameters = {
    "seq_len": SEQ_LEN,
    "hidden_size": 4,
    "num_heads": 2,
    "size_per_head": 2,
    "word_dict_size": DICT_SIZE,
    "max_word_position": 8
}


class TinyBertOnnxModel(OnnxModel):
    def __init__(self, model):
        super(TinyBertOnnxModel, self).__init__(model)
        self.resize_model()

    def resize_weight(self, initializer_name, target_shape):
        weight = self.get_initializer(initializer_name)
        w = numpy_helper.to_array(weight)

        target_w = w
        if len(target_shape) == 1:
            target_w = w[:target_shape[0]]
        elif len(target_shape) == 2:
            target_w = w[:target_shape[0], :target_shape[1]]
        elif len(target_shape) == 3:
            target_w = w[:target_shape[0], :target_shape[1], :target_shape[2]]
        elif len(target_shape) == 4:
            target_w = w[:target_shape[0], :target_shape[1], :target_shape[2], :target_shape[3]]
        else:
            print("at most 3 dimensions")

        tensor = onnx.helper.make_tensor(name=initializer_name + '_resize',
                                         data_type=TensorProto.FLOAT,
                                         dims=target_shape,
                                         vals=target_w.flatten().tolist())

        return tensor

    def resize_model(self):
        graph = self.model.graph
        initializers = graph.initializer

        for input in graph.input:
            if (input.type.tensor_type.shape.dim[1].dim_value == old_parameters["seq_len"]):
                print("input", input.name, input.type.tensor_type.shape)
                input.type.tensor_type.shape.dim[1].dim_value = new_parameters["seq_len"]
                print("=>", input.type.tensor_type.shape)

        reshapes = {}
        for initializer in initializers:
            tensor = numpy_helper.to_array(initializer)
            if initializer.data_type == TensorProto.FLOAT:
                dtype = np.float32
            elif initializer.data_type == TensorProto.INT32:
                dtype = np.int32
            elif initializer.data_type == TensorProto.INT64:
                dtype = np.int64
            else:
                print("data type not supported by this tool:", dtype)

            if len(tensor.shape) == 1 and tensor.shape[0] == 1:
                if tensor == old_parameters["num_heads"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["num_heads"], "=>[", new_parameters["num_heads"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([new_parameters["num_heads"]], dtype=dtype),
                                                initializer.name))
                elif tensor == old_parameters["seq_len"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["seq_len"], "=>[", new_parameters["seq_len"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([new_parameters["seq_len"]], dtype=dtype), initializer.name))
                elif tensor == old_parameters["size_per_head"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["size_per_head"], "=>[", new_parameters["size_per_head"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([new_parameters["size_per_head"]], dtype=dtype),
                                                initializer.name))
                elif tensor == old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["hidden_size"], "=>[", new_parameters["hidden_size"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([new_parameters["hidden_size"]], dtype=dtype),
                                                initializer.name))
                elif tensor == 4 * old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          4 * old_parameters["hidden_size"], "=>[", 4 * new_parameters["hidden_size"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([4 * new_parameters["hidden_size"]], dtype=dtype),
                                                initializer.name))
                elif tensor == 3 * old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          3 * old_parameters["hidden_size"], "=>[", 3 * new_parameters["hidden_size"], "]")
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray([3 * new_parameters["hidden_size"]], dtype=dtype),
                                                initializer.name))
            elif len(tensor.shape) == 0:
                if tensor == old_parameters["num_heads"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["num_heads"], "=>", new_parameters["num_heads"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(new_parameters["num_heads"], dtype=dtype), initializer.name))
                elif tensor == old_parameters["seq_len"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["seq_len"], "=>", new_parameters["seq_len"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(new_parameters["seq_len"], dtype=dtype), initializer.name))
                elif tensor == old_parameters["size_per_head"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["size_per_head"], "=>", new_parameters["size_per_head"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(new_parameters["size_per_head"], dtype=dtype),
                                                initializer.name))
                elif tensor == old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          old_parameters["hidden_size"], "=>", new_parameters["hidden_size"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(new_parameters["hidden_size"], dtype=dtype),
                                                initializer.name))
                elif tensor == 4 * old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          4 * old_parameters["hidden_size"], "=>", 4 * new_parameters["hidden_size"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(4 * new_parameters["hidden_size"], dtype=dtype),
                                                initializer.name))
                elif tensor == 3 * old_parameters["hidden_size"]:
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          3 * old_parameters["hidden_size"], "=>", 3 * new_parameters["hidden_size"])
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(3 * new_parameters["hidden_size"], dtype=dtype),
                                                initializer.name))
                elif tensor == 1.0 / np.sqrt(old_parameters["size_per_head"]):
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          1.0 / np.sqrt(old_parameters["size_per_head"]), "=>",
                          1.0 / np.sqrt(new_parameters["size_per_head"]))
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(1.0 / np.sqrt(new_parameters["size_per_head"]), dtype=dtype),
                                                initializer.name))
                elif tensor == np.sqrt(old_parameters["size_per_head"]):
                    print("initializer type={}".format(initializer.data_type), initializer.name,
                          np.sqrt(old_parameters["size_per_head"]), "=>", np.sqrt(new_parameters["size_per_head"]))
                    initializer.CopyFrom(
                        numpy_helper.from_array(np.asarray(np.sqrt(new_parameters["size_per_head"]), dtype=dtype),
                                                initializer.name))

            new_shape = []
            shape_changed = False
            for dim in tensor.shape:
                if (dim == old_parameters["hidden_size"]):
                    new_shape.append(new_parameters["hidden_size"])
                    shape_changed = True
                elif (dim == 4 * old_parameters["hidden_size"]):
                    new_shape.append(4 * new_parameters["hidden_size"])
                    shape_changed = True
                elif (dim == 3 * old_parameters["hidden_size"]):
                    new_shape.append(3 * new_parameters["hidden_size"])
                    shape_changed = True
                elif (dim in old_parameters["word_dict_size"]):
                    new_shape.append(new_parameters["word_dict_size"])
                    shape_changed = True
                elif (dim == old_parameters["max_word_position"]):
                    new_shape.append(new_parameters["max_word_position"])
                    shape_changed = True
                else:
                    new_shape.append(dim)
            if shape_changed:
                reshapes[initializer.name] = new_shape
                print("initializer", initializer.name, tensor.shape, "=>", new_shape)

        for initializer_name in reshapes:
            self.replace_input_of_all_nodes(initializer_name, initializer_name + '_resize')
            tensor = self.resize_weight(initializer_name, reshapes[initializer_name])
            self.model.graph.initializer.extend([tensor])

        # Add node name, replace split node attribute.
        nodes_to_add = []
        nodes_to_remove = []
        for i, node in enumerate(graph.node):
            if node.op_type == "Split":
                nodes_to_add.append(
                    onnx.helper.make_node('Split',
                                          node.input,
                                          node.output,
                                          name="Split_{}".format(i),
                                          axis=2,
                                          split=[
                                              new_parameters["hidden_size"], new_parameters["hidden_size"],
                                              new_parameters["hidden_size"]
                                          ]))
                nodes_to_remove.append(node)
                print("update split",
                      [new_parameters["hidden_size"], new_parameters["hidden_size"], new_parameters["hidden_size"]])
            if node.op_type == "Constant":
                for att in node.attribute:
                    if att.name == 'value':
                        if numpy_helper.to_array(att.t) == old_parameters["num_heads"]:
                            nodes_to_add.append(
                                onnx.helper.make_node('Constant',
                                                      inputs=node.input,
                                                      outputs=node.output,
                                                      value=onnx.helper.make_tensor(name=att.t.name,
                                                                                    data_type=TensorProto.INT64,
                                                                                    dims=[],
                                                                                    vals=[new_parameters["num_heads"]
                                                                                          ])))
                            print("constant", att.t.name, old_parameters["num_heads"], "=>",
                                  new_parameters["num_heads"])
                        if numpy_helper.to_array(att.t) == np.sqrt(old_parameters["size_per_head"]):
                            nodes_to_add.append(
                                onnx.helper.make_node('Constant',
                                                      inputs=node.input,
                                                      outputs=node.output,
                                                      value=onnx.helper.make_tensor(
                                                          name=att.t.name,
                                                          data_type=TensorProto.FLOAT,
                                                          dims=[],
                                                          vals=[np.sqrt(new_parameters["size_per_head"])])))
                            print("constant", att.t.name, np.sqrt(old_parameters["size_per_head"]), "=>",
                                  np.sqrt(new_parameters["size_per_head"]))
            else:
                node.name = node.op_type + "_" + str(i)
        for node in nodes_to_remove:
            graph.node.remove(node)
        graph.node.extend(nodes_to_add)

    def remove_past_outputs(self):
        keep_output_names = [self.model.graph.output[0].name]  # remove past state outputs which is not needed.
        print(f"Prune graph to keep the first output and drop past state outputs:{keep_output_names}")
        self.prune_graph(keep_output_names)


def generate_test_data(onnx_file,
                       output_path,
                       batch_size,
                       sequence_length,
                       use_cpu=True,
                       input_tensor_only=False,
                       dictionary_size=DICT_SIZE,
                       test_cases=1,
                       output_optimized_model=False):

    input_data_type = np.int64
    for test_case in range(test_cases):
        input_1 = np.random.randint(dictionary_size, size=(batch_size, sequence_length), dtype=input_data_type)
        tensor_1 = numpy_helper.from_array(input_1, 'input_ids')

        path = os.path.join(output_path, 'test_data_set_' + str(test_case))
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        if input_tensor_only:
            return

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options, providers=['CPUExecutionProvider'])

        input1_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        inputs = {input1_name: input_1}
        result = sess.run(output_names, inputs)

        with open(os.path.join(path, 'input_{}.pb'.format(0)), 'wb') as f:
            f.write(tensor_1.SerializeToString())

        for i, output_name in enumerate(output_names):
            if i == 0:
                tensor_result = numpy_helper.from_array(
                    np.asarray(result[i]).reshape((batch_size, sequence_length, new_parameters["hidden_size"])),
                    output_names[i])
                with open(os.path.join(path, 'output_{}.pb'.format(i)), 'wb') as f:
                    f.write(tensor_result.SerializeToString())
            else:
                tensor_result = numpy_helper.from_array(
                    np.asarray(result[i]).reshape(
                        (2, batch_size, new_parameters["num_heads"], sequence_length, new_parameters["size_per_head"])),
                    output_names[i])
                with open(os.path.join(path, 'output_{}.pb'.format(i)), 'wb') as f:
                    f.write(tensor_result.SerializeToString())

        start_time = timeit.default_timer()

        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        if output_optimized_model:
            path_prefix = onnx_file[:-5]  #remove .onnx suffix
            if use_cpu:
                sess_options.optimized_model_filepath = path_prefix + "_optimized_cpu.onnx"
            else:
                sess_options.optimized_model_filepath = path_prefix + "_optimized_gpu.onnx"

        session = onnxruntime.InferenceSession(onnx_file, sess_options)
        if use_cpu:
            session.set_providers(['CPUExecutionProvider'])  # use cpu
        else:
            if 'CUDAExecutionProvider' not in session.get_providers():
                print("Warning: GPU not found")
                continue
        outputs = session.run(None, inputs)
        evalTime = timeit.default_timer() - start_time
        if not np.allclose(outputs[0], result[0], rtol=1e-04, atol=1e-05):
            print("Error: not same result after optimization. use_cpu={}, no_opt_output={}, opt_output={}".format(
                use_cpu, result[0].tolist(), outputs[0].tolist()))
        print("** Evaluation done in total {} secs".format(evalTime))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--float16', required=False, action='store_true')
    parser.set_defaults(float16=False)
    parser.add_argument('--no_past_outputs', required=False, action='store_true')
    parser.set_defaults(no_past_outputs=False)
    parser.add_argument('--output_optimized_model', required=False, action='store_true')
    parser.set_defaults(output_optimized_model=False)
    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    bert_model = TinyBertOnnxModel(model)

    if args.float16:
        bert_model.convert_model_float32_to_float16()

    if args.no_past_outputs:
        bert_model.remove_past_outputs()

    bert_model.update_graph()
    bert_model.remove_unused_constant()

    print("opset verion", bert_model.model.opset_import[0].version)

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

    p = Path(args.output)
    data_path = p.parent

    batch_size = 1
    sequence_length = SEQ_LEN

    generate_test_data(args.output,
                       data_path,
                       batch_size,
                       sequence_length,
                       use_cpu=not args.float16,
                       output_optimized_model=args.output_optimized_model)


if __name__ == "__main__":
    main()
