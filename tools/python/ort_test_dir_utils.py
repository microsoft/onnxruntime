import numpy as np
import onnx
import onnxruntime as ort
import os
import shutil

from onnx import numpy_helper


def _get_numpy_type(model_info, name):
    for i in model_info:
        if i.name == name:
            type_name = i.type.WhichOneof('value')
            if type_name == 'tensor_type':
                return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
            else:
                raise ValueError(f"Type is not handled: {type_name}")

    raise ValueError(f"{name} was not found in the model info.")


def create_missing_input_data(model_inputs, name_input_map, symbolic_dim_values_map):
    """
    Update name_input_map with random input for any missing values in the model inputs.

    :param model_inputs: model.graph.input from an onnx model
    :param name_input_map: Map of input names to values to update. Can be empty. Existing values are preserved.
    :param symbolic_dim_values_map: Map of symbolic dimension names to values to use if creating data.
    """
    for input in model_inputs:
        if input.name in name_input_map and name_input_map[input.name] is not None:
            continue

        input_type = input.type.WhichOneof('value')
        if input_type != 'tensor_type':
            raise ValueError(f'Unsupported model. Need to handle input type of {input_type}')

        shape = input.type.tensor_type.shape
        dims = []
        for dim in shape.dim:
            dim_type = dim.WhichOneof('value')
            if dim_type == 'dim_value':
                dims.append(dim.dim_value)
            elif dim_type == 'dim_param':
                if dim.dim_param not in symbolic_dim_values_map:
                    raise ValueError(f"Value for symbolic dim {dim.dim_param} was not provided.")

                dims.append(symbolic_dim_values_map[dim.dim_param])
            else:
                # TODO: see if we need to provide a way to specify these values. could ask for the whole
                # shape for the input name instead.
                raise ValueError("Unsupported model. Unknown dim with no value or symbolic name.")

        # create random data. give it range -10 to 10 so if we convert to an integer type it's not all 0s and 1s
        # TODO: consider if the range should be configurable
        np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]
        data = (np.random.standard_normal(dims) * 10).astype(np_type)
        name_input_map[input.name] = data


def create_test_dir(model_path, root_path, test_name,
                    name_input_map={}, symbolic_dim_values_map={},
                    name_output_map=None):
    """
    Create a test directory that can be used with onnx_test_runner or onnxruntime_perf_test.
    Generates random input data for any missing inputs.
    Saves output from running the model if name_output_map is not provided.

    :param model_path: Path to the onnx model file to use.
    :param root_path: Root path to create the test directory in.
    :param test_name: Name for test. Will be added to the root_path to create the test directory name.
    :param name_input_map: Map of input names to numpy ndarray data for each input.
    :param symbolic_dim_values_map: Map of symbolic dimension names to values to use for the input data if creating
                                    using random data.
    :param name_output_map: Optional map of output names to numpy ndarray expected output data.
                            If not provided, the model will be run with the input to generate output data to save.
    :return: None
    """

    model_path = os.path.abspath(model_path)
    root_path = os.path.abspath(root_path)
    test_dir = os.path.join(root_path, test_name)
    test_data_dir = os.path.join(test_dir, f"test_data_set_0")

    if not os.path.exists(test_dir) or not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    model_filename = os.path.split(model_path)[-1]
    test_model_filename = os.path.join(test_dir, model_filename)
    shutil.copy(model_path, test_model_filename)

    model = onnx.load(model_path)
    model_inputs = model.graph.input
    model_outputs = model.graph.output

    def save_data(prefix, name_data_map, model_info):
        idx = 0
        for name, data in name_data_map.items():
            if isinstance(data, dict):
                # ignore. map<T1, T2> from traditional ML ops
                pass
            elif isinstance(data, list):
                # ignore. vector<map<T1,T2>> from traditional ML ops. e.g. ZipMap output
                pass
            else:
                np_type = _get_numpy_type(model_info, name)
                tensor = numpy_helper.from_array(data.astype(np_type), name)
                filename = os.path.join(test_data_dir, f"{prefix}_{idx}.pb")
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())

            idx += 1

    create_missing_input_data(model_inputs, name_input_map, symbolic_dim_values_map)

    save_data("input", name_input_map, model_inputs)

    # save expected output data if provided. run model to create if not.
    if not name_output_map:
        output_names = [o.name for o in model_outputs]
        sess = ort.InferenceSession(test_model_filename)
        outputs = sess.run(output_names, name_input_map)
        name_output_map = {}
        for name, data in zip(output_names, outputs):
            name_output_map[name] = data

    save_data("output", name_output_map, model_outputs)

