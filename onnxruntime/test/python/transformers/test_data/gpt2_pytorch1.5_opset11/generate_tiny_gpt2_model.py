# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This tool generates a tiny GPT2 model for testing fusion script.
# You can use benchmark_gpt2.py to get a gpt2 ONNX model as input of this tool.

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import onnx.utils
from onnx import ModelProto, TensorProto, numpy_helper
from onnxruntime_tools.transformers.onnx_model import OnnxModel

import onnxruntime

DICT_SIZE = 20
SEQ_LEN = 5
""" This class creates a tiny bert model for test purpose. """

# parameters of input base model.
old_parameters = {
    "seq_len": 5,
    "hidden_size": 768,
    "num_heads": 12,
    "size_per_head": 64,
    "word_dict_size": [50257],  # list of supported dictionary size.
    "max_word_position": 1024,
}

# parameters of output tiny model.
new_parameters = {
    "seq_len": SEQ_LEN,
    "hidden_size": 4,
    "num_heads": 2,
    "size_per_head": 2,
    "word_dict_size": DICT_SIZE,
    "max_word_position": 8,
}


class TinyGpt2Model(OnnxModel):
    def __init__(self, model):
        super().__init__(model)
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
        elif len(target_shape) == 4:
            target_w = w[
                : target_shape[0],
                : target_shape[1],
                : target_shape[2],
                : target_shape[3],
            ]
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

        for input in graph.input:
            if input.type.tensor_type.shape.dim[1].dim_value == old_parameters["seq_len"]:
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
                elif tensor == 3 * old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        3 * old_parameters["hidden_size"],
                        "=>[",
                        3 * new_parameters["hidden_size"],
                        "]",
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray([3 * new_parameters["hidden_size"]], dtype=dtype),
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
                elif tensor == 3 * old_parameters["hidden_size"]:
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        3 * old_parameters["hidden_size"],
                        "=>",
                        3 * new_parameters["hidden_size"],
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(3 * new_parameters["hidden_size"], dtype=dtype),
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
                elif tensor == np.sqrt(old_parameters["size_per_head"]):
                    print(
                        f"initializer type={initializer.data_type}",
                        initializer.name,
                        np.sqrt(old_parameters["size_per_head"]),
                        "=>",
                        np.sqrt(new_parameters["size_per_head"]),
                    )
                    initializer.CopyFrom(
                        numpy_helper.from_array(
                            np.asarray(np.sqrt(new_parameters["size_per_head"]), dtype=dtype),
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
                elif dim == 3 * old_parameters["hidden_size"]:
                    new_shape.append(3 * new_parameters["hidden_size"])
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

        # Add node name, replace split node attribute.
        nodes_to_add = []
        nodes_to_remove = []
        for i, node in enumerate(graph.node):
            if node.op_type == "Split":
                nodes_to_add.append(
                    onnx.helper.make_node(
                        "Split",
                        node.input,
                        node.output,
                        name=f"Split_{i}",
                        axis=2,
                        split=[
                            new_parameters["hidden_size"],
                            new_parameters["hidden_size"],
                            new_parameters["hidden_size"],
                        ],
                    )
                )
                nodes_to_remove.append(node)
                print(
                    "update split",
                    [
                        new_parameters["hidden_size"],
                        new_parameters["hidden_size"],
                        new_parameters["hidden_size"],
                    ],
                )
            if node.op_type == "Constant":
                for att in node.attribute:
                    if att.name == "value":
                        if numpy_helper.to_array(att.t) == old_parameters["num_heads"]:
                            nodes_to_add.append(
                                onnx.helper.make_node(
                                    "Constant",
                                    inputs=node.input,
                                    outputs=node.output,
                                    value=onnx.helper.make_tensor(
                                        name=att.t.name,
                                        data_type=TensorProto.INT64,
                                        dims=[],
                                        vals=[new_parameters["num_heads"]],
                                    ),
                                )
                            )
                            print(
                                "constant",
                                att.t.name,
                                old_parameters["num_heads"],
                                "=>",
                                new_parameters["num_heads"],
                            )
                        if numpy_helper.to_array(att.t) == np.sqrt(old_parameters["size_per_head"]):
                            nodes_to_add.append(
                                onnx.helper.make_node(
                                    "Constant",
                                    inputs=node.input,
                                    outputs=node.output,
                                    value=onnx.helper.make_tensor(
                                        name=att.t.name,
                                        data_type=TensorProto.FLOAT,
                                        dims=[],
                                        vals=[np.sqrt(new_parameters["size_per_head"])],
                                    ),
                                )
                            )
                            print(
                                "constant",
                                att.t.name,
                                np.sqrt(old_parameters["size_per_head"]),
                                "=>",
                                np.sqrt(new_parameters["size_per_head"]),
                            )
            else:
                node.name = node.op_type + "_" + str(i)
        for node in nodes_to_remove:
            graph.node.remove(node)
        graph.node.extend(nodes_to_add)

        for i, input in enumerate(self.model.graph.input):
            if i > 0:
                dim_proto = input.type.tensor_type.shape.dim[2]
                dim_proto.dim_value = new_parameters["num_heads"]
                dim_proto = input.type.tensor_type.shape.dim[4]
                dim_proto.dim_value = new_parameters["size_per_head"]

        for i, output in enumerate(self.model.graph.output):
            if i == 0:
                dim_proto = output.type.tensor_type.shape.dim[2]
                dim_proto.dim_value = new_parameters["hidden_size"]
            if i > 0:
                dim_proto = output.type.tensor_type.shape.dim[2]
                dim_proto.dim_value = new_parameters["num_heads"]
                dim_proto = output.type.tensor_type.shape.dim[4]
                dim_proto.dim_value = new_parameters["size_per_head"]


def generate_test_data(
    onnx_file,
    output_path,
    batch_size=1,
    use_cpu=True,
    input_tensor_only=False,
    dictionary_size=DICT_SIZE,
    test_cases=1,
    output_optimized_model=False,
):
    for test_case in range(test_cases):
        sequence_length = 3
        input_1 = np.random.randint(dictionary_size, size=(batch_size, 1), dtype=np.int64)
        tensor_1 = numpy_helper.from_array(input_1, "input_ids")

        path = os.path.join(output_path, "test_data_set_" + str(test_case))
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = onnxruntime.InferenceSession(onnx_file, sess_options, providers=["CPUExecutionProvider"])

        input1_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        inputs = {input1_name: input_1}

        with open(os.path.join(path, f"input_{0}.pb"), "wb") as f:
            f.write(tensor_1.SerializeToString())

        for i in range(12):
            input_name = f"past_{i}"
            input = np.random.rand(
                2,
                batch_size,
                new_parameters["num_heads"],
                sequence_length,
                new_parameters["size_per_head"],
            ).astype(np.float32)
            tensor = numpy_helper.from_array(input, input_name)
            inputs.update({input_name: input})

            with open(os.path.join(path, f"input_{1 + i}.pb"), "wb") as f:
                f.write(tensor.SerializeToString())

        if input_tensor_only:
            return

        result = sess.run(output_names, inputs)
        print("result 0 shape:", result[0].shape)
        print("result 1 shape:", result[1].shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--output_optimized_model", required=False, action="store_true")
    parser.set_defaults(output_optimized_model=False)
    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    bert_model = TinyGpt2Model(model)

    bert_model.update_graph()
    bert_model.remove_unused_constant()

    print("opset version", bert_model.model.opset_import[0].version)

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

    p = Path(args.output)
    data_path = p.parent

    generate_test_data(
        args.output,
        data_path,
        batch_size=1,
        use_cpu=True,
        output_optimized_model=args.output_optimized_model,
    )


if __name__ == "__main__":
    main()
