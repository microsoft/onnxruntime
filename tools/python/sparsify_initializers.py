# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This script opens an existing model in onnx format and attempts to
# move initializers from model.graph.initializer field to model.graph.sparse_initializer field
# and convert them into ONNX COO flat index format.

import argparse
import logging
import sys
from typing import List, Tuple  # noqa: F401

import numpy as np
import onnx
from onnx import ModelProto, SparseTensorProto, TensorProto, numpy_helper  # noqa: F401

logger = logging.getLogger(__name__)

real_types = {int(TensorProto.FLOAT), int(TensorProto.DOUBLE)}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="input model path")
    parser.add_argument("--output", required=True, type=str, help="output model path")
    parser.add_argument(
        "--exclude", required=False, type=str, help="semicolon separated list of initializer names to exclude"
    )
    parser.add_argument("--tolerance", required=False, type=float, default=1e-6, help="FP absolute tolerance.")
    parser.add_argument(
        "--sparsity_threshold",
        required=False,
        type=float,
        default=0.5,
        help="convert to sparse initializers if sparsity is at least this much",
    )
    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    return args


def setup_logging(verbose):  # type: (bool)  -> None
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter("%(filename)20s: %(message)s"))
        logging_level = logging.INFO
    log_handler.setLevel(logging_level)
    logger.addHandler(log_handler)
    logger.setLevel(logging_level)


def convert_tensor_to_sparse(
    tensor, sparsity_threshold, tolerance
):  # type: (TensorProto, float, float) -> Tuple[SparseTensorProto, float]
    """returns a tuple of sparse_tensor and sparsity level"""
    values = []
    indices = []
    nnz_count = 0
    tensor_data = numpy_helper.to_array(tensor).flatten()
    data_len = len(tensor_data)
    if tensor_data.dtype in real_types:
        for index in range(data_len):
            el = tensor_data[index]
            if abs(el) <= tolerance:
                values.append(el)
                indices.append(index)
                nnz_count += 1
    else:
        for index in range(data_len):
            el = tensor_data[index]
            if el != 0:
                values.append(el)
                indices.append(index)
                nnz_count += 1

    sparsity = 1.0 - float(nnz_count) / data_len

    ind_data_type = TensorProto.INT8
    ind_dtype = np.int8
    ind_len = len(indices)
    max_indices_value = 0
    if ind_len > 0:
        max_indices_value = indices[-1]
        if max_indices_value <= np.iinfo(np.int8).max:
            ind_data_type = TensorProto.INT8
            ind_dtype = np.int8
        elif max_indices_value <= np.iinfo(np.int16).max:
            ind_data_type = TensorProto.INT16
            ind_dtype = np.int16
        elif max_indices_value <= np.iinfo(np.int32).max:
            ind_data_type = TensorProto.INT32
            ind_dtype = np.int32
        else:
            ind_data_type = TensorProto.INT64
            ind_dtype = np.int64

    logger.debug(
        f"initializer={tensor.name}, dtype={tensor_data.dtype}, \
                 data_len={data_len}, nnz={nnz_count}, sparsity={sparsity}, \
                 max_indices_value={max_indices_value}, sparse_indices_type={ind_dtype}"
    )

    if sparsity < sparsity_threshold:
        return (object(), sparsity)

    tensor_data_bytes = tensor_data.nbytes
    # create np array and cast data to the appropriate type
    np_values = np.array(values).astype(tensor_data.dtype)
    # create np array and cast data to the inferred index type
    np_indices = np.array(indices).astype(ind_dtype)
    total_sparse_bytes = np_values.nbytes + np_indices.nbytes

    logger.debug(
        f"initializer={tensor.name}, initializer_bytes={tensor_data_bytes}, \
                sparse_initializer_bytes={total_sparse_bytes}"
    )

    # This check is usually useful for sparsity_threshold=0.5 where much
    # depends on the size of the indices entries and the size of the original tensor.
    # Big dense tensors command larger indices data type and for large float32 tensors
    # int32 indices are often selected, thus we really want to guard against loosing
    # rather than winning.
    if tensor_data_bytes <= total_sparse_bytes:
        sparsity = 1.0 - float(tensor_data_bytes) / total_sparse_bytes
        logger.debug(f"initializer={tensor.name}, adjusted_sparsity={sparsity}")
        return (object(), sparsity)

    values_tensor = onnx.helper.make_tensor(tensor.name, tensor.data_type, [len(values)], np_values.tobytes(), raw=True)

    indicies_tensor = onnx.helper.make_tensor(
        tensor.name + "_indicies", ind_data_type, [ind_len], np_indices.tobytes(), raw=True
    )

    sparse_tensor = onnx.helper.make_sparse_tensor(values_tensor, indicies_tensor, tensor.dims)
    return (sparse_tensor, sparsity)


def convert_initializers(
    model, exclude_names, sparsity_threshold, tolerance
):  # type: (ModelProto, List[str], float, float) -> None
    graph = model.graph
    converted_sparse = []
    remaining_initializers = []
    for initializer in graph.initializer:
        if initializer.name in exclude_names:
            logger.info(f"initializer={initializer.name} was excluded")
            continue
        if initializer.data_type == TensorProto.BOOL:
            logger.info(f"initializer={initializer.name} contains bool, not converted")
            remaining_initializers.append(initializer)
            continue
        sparse_tensor, sparsity = convert_tensor_to_sparse(initializer, sparsity_threshold, tolerance)
        if sparsity >= sparsity_threshold:
            logger.info(f"initializer={initializer.name} converted. sparsity={sparsity}")
            converted_sparse.append(sparse_tensor)
        else:
            remaining_initializers.append(initializer)
            logger.info(f"initializer={initializer.name} is not converted. sparsity={sparsity}")

    graph.sparse_initializer.extend(converted_sparse)
    del graph.initializer[:]
    graph.initializer.extend(remaining_initializers)


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    exclude_names = set() if args.exclude is None else set(args.exclude.split(";"))

    model = ModelProto()
    with open(args.input, "rb") as input_file:
        model.ParseFromString(input_file.read())

    convert_initializers(model, exclude_names, args.sparsity_threshold, args.tolerance)

    with open(args.output, "wb") as output_file:
        s = model.SerializeToString()
        output_file.write(s)


if __name__ == "__main__":
    main()
