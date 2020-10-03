import glob
import numpy as np
import onnx
import onnx_test_data_utils
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
                raise ValueError("Type is not handled: {}".format(type_name))

    raise ValueError("{} was not found in the model info.".format(name))


def _create_missing_input_data(model_inputs, name_input_map, symbolic_dim_values_map):
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
            raise ValueError('Unsupported model. Need to handle input type of {}'.format(input_type))

        shape = input.type.tensor_type.shape
        dims = []
        for dim in shape.dim:
            dim_type = dim.WhichOneof('value')
            if dim_type == 'dim_value':
                dims.append(dim.dim_value)
            elif dim_type == 'dim_param':
                if dim.dim_param not in symbolic_dim_values_map:
                    raise ValueError("Value for symbolic dim {} was not provided.".format(dim.dim_param))

                dims.append(symbolic_dim_values_map[dim.dim_param])
            else:
                # TODO: see if we need to provide a way to specify these values. could ask for the whole
                # shape for the input name instead.
                raise ValueError("Unsupported model. Unknown dim with no value or symbolic name.")

        np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]
        # create random data. give it range -10 to 10 so if we convert to an integer type it's not all 0s and 1s
        data = (np.random.standard_normal(dims) * 10).astype(np_type)

        name_input_map[input.name] = data


def create_test_dir(model_path, root_path, test_name,
                    name_input_map=None, symbolic_dim_values_map=None,
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
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # add to existing test data sets if present
    test_num = 0
    while True:
        test_data_dir = os.path.join(test_dir, "test_data_set_" + str(test_num))
        if not os.path.exists(test_data_dir):
            os.mkdir(test_data_dir)
            break

        test_num += 1

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
                filename = os.path.join(test_data_dir, "{}_{}.pb".format(prefix, idx))
                with open(filename, 'wb') as f:
                    f.write(tensor.SerializeToString())

            idx += 1

    if not name_input_map:
        name_input_map = {}

    if not symbolic_dim_values_map:
        symbolic_dim_values_map = {}

    _create_missing_input_data(model_inputs, name_input_map, symbolic_dim_values_map)

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


def read_test_dir(dir_name):
    """
    Read the input and output .pb files from the provided directory.
    Input files should have a prefix of 'input_'
    Output files, which are optional, should have a prefix of 'output_'
    :param dir_name: Directory to read files from
    :return: tuple(dictionary of input name to numpy.ndarray of data,
                   dictionary of output name to numpy.ndarray)
    """

    inputs = {}
    outputs = {}
    input_files = glob.glob(os.path.join(dir_name, 'input_*.pb'))
    output_files = glob.glob(os.path.join(dir_name, 'output_*.pb'))

    for i in input_files:
        name, data = onnx_test_data_utils.read_tensorproto_pb_file(i)
        inputs[name] = data

    for o in output_files:
        name, data = onnx_test_data_utils.read_tensorproto_pb_file(o)
        outputs[name] = data

    return inputs, outputs


def run_test_dir(model_or_dir):
    """
    Run the test/s from a directory in ONNX test format.
    All subdirectories with a prefix of 'test' are considered test input for one test run.

    :param model_or_dir: Path to onnx model in test directory,
                         or the test directory name if the directory only contains one .onnx model.
    :return: None
    """

    if os.path.isdir(model_or_dir):
        model_dir = os.path.abspath(model_or_dir)
        # check there's only one onnx file
        onnx_models = glob.glob(os.path.join(model_dir, '*.onnx'))
        ort_models = glob.glob(os.path.join(model_dir, '*.ort'))
        models = onnx_models + ort_models
        if len(models) > 1:
            raise ValueError("'Multiple .onnx and/or .ort files found in {}. '"
                             "'Please provide specific .onnx or .ort file as input.".format(model_dir))
        elif len(models) == 0:
            raise ValueError("'No .onnx or .ort files found in {}.".format(model_dir))

        model_path = models[0]
    else:
        model_path = os.path.abspath(model_or_dir)
        model_dir = os.path.dirname(model_path)

    print('Running tests in {} for {}'.format(model_dir, model_path))

    test_dirs = [d for d in glob.glob(os.path.join(model_dir, 'test*')) if os.path.isdir(d)]
    if not test_dirs:
        raise ValueError("No directories with name starting with 'test' were found in {}.".format(model_dir))

    sess = ort.InferenceSession(model_path)

    for d in test_dirs:
        print(d)
        inputs, expected_outputs = read_test_dir(d)

        if expected_outputs:
            output_names = list(expected_outputs.keys())
            # handle case where there's a single expected output file but no name in it (empty string for name)
            # e.g. ONNX test models 20190729\opset8\tf_mobilenet_v2_1.4_224
            if len(output_names) == 1 and output_names[0] == '':
                output_names = [o.name for o in sess.get_outputs()]
                assert(len(output_names) == 1)
                expected_outputs[output_names[0]] = expected_outputs['']
                expected_outputs.pop('')

        else:
            output_names = [o.name for o in sess.get_outputs()]

        run_outputs = sess.run(output_names, inputs)
        failed = False
        if expected_outputs:
            for idx in range(len(output_names)):
                expected = expected_outputs[output_names[idx]]
                actual = run_outputs[idx]

                if expected.dtype.char in np.typecodes['AllFloat']:
                    if not np.isclose(expected, actual, rtol=1.e-3, atol=1.e-3).all():
                        print('Mismatch for {}:\nExpected:{}\nGot:{}'.format(output_names[idx], expected, actual))
                        failed = True
                else:
                    if not np.equal(expected, actual).all():
                        print('Mismatch for {}:\nExpected:{}\nGot:{}'.format(output_names[idx], expected, actual))
                        failed = True
        if failed:
            raise ValueError('FAILED due to output mismatch.')
        else:
            print('PASS')
