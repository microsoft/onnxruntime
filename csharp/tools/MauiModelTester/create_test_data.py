import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# set to the directory the ONNX Runtime repo is in
# `git checkout https://github.com/microsoft/onnxruntime.git` if needed.
ORT_ROOT_DIR = Path(__file__).parents[3]
SOLUTION_DIR = Path(__file__).parent

# add path for test data/dir generation utils
sys.path.append(str(ORT_ROOT_DIR / "tools" / "python"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Setup the model and test data for usage with the MAUI model tester app.
        Input data will be randomly generated as needed.
        The model will be run locally and the output saved as expected output.
        Explicit input data or expected output data can be specified by providing .pb files with the input/output name
        and tensor. These can be created with /tools/python/onnx_test_data_utils.py.

        See https://github.com/microsoft/onnxruntime/blob/main/tools/python/PythonTools.md#creating-a-test-directory-for-a-model  # noqa
        for info on creating specific input or expected output""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbolic_dims",
        "-s",
        help="Symbolic dimension values if the model inputs have symbolic dimensions and the input data is being "
        "generated. Format is `name=value {name2=value2 ...}.",
        type=str,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--input_data",
        "-i",
        help="Input data pb files created with onnx_test_data_utils.py. Multiple can be specified.",
        type=Path,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--output_data",
        "-o",
        help="Expected output data pb files created with onnx_test_data_utils.py. Multiple can be specified.",
        type=Path,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--model_path",
        "-m",
        help="Path to ONNX model to use. Model will be copied into the test app",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.model_path.resolve(strict=True)

    # convert symbolic dims to dictionary
    symbolic_dims = None
    if args.symbolic_dims:
        symbolic_dims = {}
        for value in args.symbolic_dims:
            pieces = value.split("=")
            assert len(pieces) == 2
            name = pieces[0].strip()
            dim_value = int(pieces[1].strip())
            symbolic_dims[name] = dim_value

    args.symbolic_dims = symbolic_dims

    return args


def create_existing_data_map(pb_files: List[Path]):
    import onnx_test_data_utils as data_utils

    data_map = {}
    for file in pb_files:
        file.resolve(strict=True)
        name, data = data_utils.read_tensorproto_pb_file(str(file))
        data_map[name] = data

    return data_map


def add_model_and_test_data_to_app(
    model_path: Path,
    symbolic_dims: Optional[Dict[str, int]] = None,
    input_map: Optional[Dict[str, np.ndarray]] = None,
    output_map: Optional[Dict[str, np.ndarray]] = None,
):
    import ort_test_dir_utils as utils

    output_path = SOLUTION_DIR / "Resources" / "Raw"
    test_name = "test_data"

    test_path = output_path / test_name
    # remove existing data
    if test_path.exists():
        shutil.rmtree(test_path)

    # If you want to directly create input data without using onnx_test_data_utils you can edit the input map here
    # if not input_map:
    #     input_map = {}
    #
    # input_map['Input3'] = np.random.rand(1, 1, 28, 28).astype(np.float32)

    utils.create_test_dir(
        str(model_path),
        str(output_path),
        test_name,
        # Explicit input data. Any missing required inputs will have data generated for them.
        name_input_map=input_map,
        # Optional map for any symbolic values.
        symbolic_dim_values_map=symbolic_dims,
        # Expected output can be provided if you want to validate model output against this.
        name_output_map=output_map,
    )

    # create_test_dir will copy the model to the output directory.
    # rename the copied model to the generic name the app expects.
    copied_model = output_path / test_name / model_path.name
    copied_model.rename(copied_model.with_name("model.onnx"))

    # add a text file with the original model path just so there's some info on where it came from
    with open(test_path / "model_info.txt", "w") as model_info_file:
        model_info_file.write(str(model_path))


def create_test_data():
    args = parse_args()

    input_map = None
    output_map = None

    if args.input_data:
        input_map = create_existing_data_map(args.input_data)

    if args.output_data:
        output_map = create_existing_data_map(args.output_data)

    add_model_and_test_data_to_app(args.model_path, args.symbolic_dims, input_map, output_map)


if __name__ == "__main__":
    create_test_data()
