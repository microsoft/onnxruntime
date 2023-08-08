# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Convert a Bert large model exported by Keras2Onnx to a tiny model for test purpose.
The input model is generated like the following (need install keras2onnx from source):

    import numpy
    import keras2onnx
    from transformers import (TFBertForQuestionAnswering, BertTokenizer)

    tokenizer = BertTokenizer.from_pretrained'bert-large-uncased-whole-word-masking-finetuned-squad', do_lower_case=True, cache_dir=cache_dir)
    model = TFBertForQuestionAnswering.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    question, text = "What is ONNX Runtime?", "ONNX Runtime is a performance-focused inference engine for ONNX models."
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors='tf')

    output_model_path =  os.path.join(output_dir, 'keras_{}.onnx'.format(model_name_or_path))
    if not os.path.exists(output_model_path):
        model.predict(inputs)
        onnx_model = keras2onnx.convert_keras(model, model.name)
        keras2onnx.save_model(onnx_model, output_model_path)
"""

import argparse
import os
import random
import sys  # noqa: F401
import timeit
from pathlib import Path

import numpy as np
import onnx
import onnx.utils
from onnx import ModelProto, TensorProto, numpy_helper
from onnxruntime_tools.transformers.onnx_model import OnnxModel

import onnxruntime

DICT_SIZE = 20
SEQ_LEN = 7
""" This class creates a tiny bert model for test purpose. """


class TinyBertOnnxModel(OnnxModel):
    def __init__(self, model, verbose):
        super().__init__(model, verbose)
        self.resize_model()

    def resize_weight(self, initializer_name, target_shape):
        weight = self.get_initializer(initializer_name)
        w = numpy_helper.to_array(weight)

        target_w = w
        if len(target_shape) == 1:
            target_w = w[: target_shape[0]]
        elif len(target_shape) == 2:
            target_w = w[: target_shape[0], : target_shape[1]]
        elif len(target_shape) == 3:
            target_w = w[: target_shape[0], : target_shape[1], : target_shape[2]]
        else:
            print("at most 3 dimensions")

        tensor = onnx.helper.make_tensor(
            name=initializer_name + "_resize",
            data_type=TensorProto.FLOAT,
            dims=target_shape,
            vals=target_w.flatten().tolist(),
        )

        return tensor

    def resize_model(self):
        graph = self.model.graph
        initializers = graph.initializer

        # parameters of input base model.
        old_parameters = {
            "seq_len": 26,
            "hidden_size": 1024,
            "num_heads": 16,
            "size_per_head": 64,
            "word_dict_size": [28996, 30522],  # list of supported dictionary size.
            "max_word_position": 512,
        }

        # parameters of output tiny model.
        new_parameters = {
            "seq_len": SEQ_LEN,
            "hidden_size": 8,
            "num_heads": 2,
            "size_per_head": 4,
            "word_dict_size": DICT_SIZE,
            "max_word_position": 10,
        }

        for input in graph.input:
            if input.type.tensor_type.shape.dim[1].dim_value == old_parameters["seq_len"]:
                print("input", input.name, input.type.tensor_type.shape)
                input.type.tensor_type.shape.dim[1].dim_value = new_parameters["seq_len"]
                print("=>", input.type.tensor_type.shape)

        reshapes = {}
        for initializer in initializers:
            tensor = numpy_helper.to_array(initializer)
            dtype = np.float32 if initializer.data_type == 1 else np.int32
            if len(tensor.shape) == 1 and tensor.shape[0] == 1:
                if tensor == old_parameters["num_heads"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["num_heads"],
                        "=>[",
                        new_parameters["num_heads"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([new_parameters["num_heads"]], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["seq_len"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["seq_len"],
                        "=>[",
                        new_parameters["seq_len"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([new_parameters["seq_len"]], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["size_per_head"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["size_per_head"],
                        "=>[",
                        new_parameters["size_per_head"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([new_parameters["size_per_head"]], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["hidden_size"],
                        "=>[",
                        new_parameters["hidden_size"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([new_parameters["hidden_size"]], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == 4 * old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        4 * old_parameters["hidden_size"],
                        "=>[",
                        4 * new_parameters["hidden_size"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([4 * new_parameters["hidden_size"]], dtype=dtype),
                            initializer.name,
                        )
                    )
            elif len(tensor.shape) == 0:
                if tensor == old_parameters["num_heads"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["num_heads"],
                        "=>",
                        new_parameters["num_heads"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(new_parameters["num_heads"], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["seq_len"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["seq_len"],
                        "=>",
                        new_parameters["seq_len"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(new_parameters["seq_len"], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["size_per_head"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["size_per_head"],
                        "=>",
                        new_parameters["size_per_head"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(new_parameters["size_per_head"], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        old_parameters["hidden_size"],
                        "=>",
                        new_parameters["hidden_size"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(new_parameters["hidden_size"], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == 4 * old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        4 * old_parameters["hidden_size"],
                        "=>",
                        4 * new_parameters["hidden_size"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(4 * new_parameters["hidden_size"], dtype=dtype),
                            initializer.name,
                        )
                    )
                elif tensor == 1.0 / np.sqrt(old_parameters["size_per_head"]):
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        1.0 / np.sqrt(old_parameters["size_per_head"]),
                        "=>",
                        1.0 / np.sqrt(new_parameters["size_per_head"]),
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(
                                1.0 / np.sqrt(new_parameters["size_per_head"]),
                                dtype=dtype,
                            ),
                            initializer.name,
                        )
                    )

            new_shape = []
            shape_changed = False
            for dim in tensor.shape:
                if dim == old_parameters["hidden_size"]:
                    new_shape.append(new_parameters["hidden_size"])
                    shape_changed = True
                elif dim == 4 * old_parameters["hidden_size"]:
                    new_shape.append(4 * new_parameters["hidden_size"])
                    shape_changed = True
                elif dim in old_parameters["word_dict_size"]:
                    new_shape.append(new_parameters["word_dict_size"])
                    shape_changed = True
                elif dim == old_parameters["max_word_position"]:
                    new_shape.append(new_parameters["max_word_position"])
                    shape_changed = True
                else:
                    new_shape.append(dim)
            if shape_changed:
                reshapes[initializer.name] = new_shape
                print("initializer", initializer.name, tensor.shape, "=>", new_shape)

        for initializer_name in reshapes:
            self.replace_input_of_all_nodes(initializer_name, initializer_name + "_resize")
            tensor = self.resize_weight(initializer_name, reshapes[initializer_name])
            self.model.graph.initializer.extend([tensor])

        self.use_dynamic_axes()

    def use_dynamic_axes(self, dynamic_batch_dim="batch_size", seq_len=7):
        """
        Update input and output shape to use dynamic axes.
        """
        for input in self.model.graph.input:
            dim_proto = input.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim
            dim_proto = input.type.tensor_type.shape.dim[1]
            dim_proto.dim_value = seq_len

        for output in self.model.graph.output:
            dim_proto = output.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim
            dim_proto = output.type.tensor_type.shape.dim[1]
            dim_proto.dim_value = seq_len


def generate_test_data(
    onnx_file,
    output_path,
    batch_size,
    sequence_length,
    use_cpu=True,
    input_tensor_only=False,
    dictionary_size=DICT_SIZE,
    test_cases=3,
):
    input_data_type = np.int32
    for test_case in range(test_cases):
        input_1 = np.random.randint(dictionary_size, size=(batch_size, sequence_length), dtype=input_data_type)
        tensor_1 = numpy_helper.from_array(input_1, "input_ids")

        actual_seq_len = random.randint(sequence_length - 3, sequence_length)
        input_2 = np.zeros((batch_size, sequence_length), dtype=input_data_type)
        temp = np.ones((batch_size, actual_seq_len), dtype=input_data_type)
        input_2[: temp.shape[0], : temp.shape[1]] = temp
        tensor_2 = numpy_helper.from_array(input_2, "attention_mask")

        input_3 = np.zeros((batch_size, sequence_length), dtype=input_data_type)
        tensor_3 = numpy_helper.from_array(input_3, "token_type_ids")

        path = os.path.join(output_path, "test_data_set_" + str(test_case))
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
        sess = onnxruntime.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        output_names = [output.name for output in sess.get_outputs()]
        inputs = {
            "input_ids": input_1,
            "attention_mask": input_2,
            "token_type_ids": input_3,
        }
        print("inputs", inputs)
        result = sess.run(output_names, inputs)

        with open(os.path.join(path, f"input_{0}.pb"), "wb") as f:
            f.write(tensor_1.SerializeToString())
        with open(os.path.join(path, f"input_{1}.pb"), "wb") as f:
            f.write(tensor_2.SerializeToString())
        with open(os.path.join(path, f"input_{2}.pb"), "wb") as f:
            f.write(tensor_3.SerializeToString())

        for i, _output_name in enumerate(output_names):
            tensor_result = numpy_helper.from_array(
                np.asarray(result[i]).reshape((batch_size, sequence_length)),
                output_names[i],
            )
            with open(os.path.join(path, f"output_{i}.pb"), "wb") as f:
                f.write(tensor_result.SerializeToString())

        start_time = timeit.default_timer()

        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        path_prefix = onnx_file[:-5]  # remove .onnx suffix
        if use_cpu:
            sess_options.optimized_model_filepath = path_prefix + "_optimized_cpu.onnx"
        else:
            sess_options.optimized_model_filepath = path_prefix + "_optimized_gpu.onnx"

        session = onnxruntime.InferenceSession(
            onnx_file,
            sess_options=sess_options,
            providers=onnxruntime.get_available_providers(),
        )
        if use_cpu:
            session.set_providers(["CPUExecutionProvider"])  # use cpu
        else:
            if "CUDAExecutionProvider" not in session.get_providers():
                print("Warning: GPU not found")
                continue
        outputs = session.run(None, inputs)
        evalTime = timeit.default_timer() - start_time  # noqa: N806
        if outputs[0].tolist() != result[0].tolist():
            print(
                "Error: not same result after optimization. use_cpu={}, no_opt_output={}, opt_output={}".format(
                    use_cpu, result[0].tolist(), outputs[1].tolist()
                )
            )
        print(f"** Evaluation done in total {evalTime} secs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--float16", required=False, action="store_true")
    parser.set_defaults(float16=False)
    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    bert_model = TinyBertOnnxModel(model, False)

    if args.float16:
        bert_model.convert_model_float32_to_float16()

    bert_model.update_graph()
    bert_model.remove_unused_constant()

    print("opset version", bert_model.model.opset_import[0].version)

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

    p = Path(args.output)
    data_path = p.parent

    batch_size = 1
    sequence_length = SEQ_LEN

    generate_test_data(args.output, data_path, batch_size, sequence_length, use_cpu=not args.float16)


if __name__ == "__main__":
    main()
