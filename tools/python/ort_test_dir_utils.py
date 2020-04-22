import os
import shutil
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy


def _get_numpy_type(model_info, name):
    for i in model_info:
        if i.name == name:
            type_name = i.type.WhichOneof('value')
            if type_name == 'tensor_type':
                return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
            else:
                raise ValueError(f"Type is not handled: {type_name}")

    raise ValueError(f"{name} was not found in the model info.")


def create_test_dir(model_path, root_path, test_name, name_input_map, name_output_map=None):
    """
    Create a test directory that can be used with onnx_test_runner or onnxruntime_perf_test.

    :param model_path: Path to the onnx model file to use.
    :param root_path: Root path to create the test directory in.
    :param test_name: Name for test. Will be added to the root_path to create the test directory name.
    :param name_input_map: Map of input names to numpy ndarray data for each input.
    :param name_output_map: Optional map of output names to numpy ndarray expected output data.
                            If not provided, the model will be run with the input to generate output data to save.
    :return: None
    """
    test_dir = os.path.join(root_path, test_name)
    test_data_dir = os.path.join(test_dir, f"test_data_set_0")

    if not os.path.exists(test_dir) or not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    model_filename = model_path.split('\\')[-1]
    test_model_filename = os.path.join(test_dir, model_filename)
    shutil.copy(model_path, test_model_filename)

    model = onnx.load(test_model_filename)
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
