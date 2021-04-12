# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This script opens an existing model in onnx format and attempts to
# move initializers from model.graph.initializer field to model.graph.sparse_initializer field
# and convert them into ONNX COO flat index format.

import argparse
import logging
import numpy as np
import sys
from typing import Tuple, List
import onnx
from onnx import ModelProto, SparseTensorProto, TensorProto, numpy_helper

logger = logging.getLogger(__name__)

real_types = set((int(TensorProto.FLOAT), int(TensorProto.DOUBLE)))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='input model path')
    parser.add_argument('--output', required=True, type=str, help='output model path')
    parser.add_argument('--exclude', required=False, type=str,
                        help='semicolon separated list of initializer names to exclude')
    parser.add_argument('--tolerance', required=False, type=float, default=1e-6,
                        help='FP absolute tolerance. If not given simple compare to 0')
    parser.add_argument('--sparsity_threshold', required=False,
                        type=float, default=0.5,
                        help='convert to sparse initializers if sparsity is at least this much')
    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    return args


def setup_logging(verbose):  # type: (bool)  -> None
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s'))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter('%(filename)20s: %(message)s'))
        logging_level = logging.INFO
    log_handler.setLevel(logging_level)
    logger.addHandler(log_handler)
    logger.setLevel(logging_level)


def convert_tensor_to_sparse(tensor, tolerance):  # type: (TensorProto) -> Tuple[SparseTensorProto, float]
    """ returns a tuple of sparse_tensor and sparsity level
    """
    values = []
    indicies = []
    nnz_count = 0
    tensor_data = numpy_helper.to_array(tensor).flatten()
    data_len = len(tensor_data)
    if tensor_data.dtype in real_types:
        for index in range(data_len):
            el = tensor_data[index]
            if abs(el) <= tolerance:
                values.append(el)
                indicies.append(index)
                nnz_count += 1
    else:
        for index in range(data_len):
            el = tensor_data[index]
            if el != 0:
                values.append(el)
                indicies.append(index)
                nnz_count += 1

    sparsity = float(1.) - float(nnz_count)/data_len
    logger.debug(f"initializer={tensor.name}, dtype={tensor_data.dtype}, \
                 len={data_len}, nnz={nnz_count}, sparsity={sparsity}")

    values_tensor = onnx.helper.make_tensor(tensor.name, tensor.data_type,
                                            [len(values)], np.array(values).astype(tensor_data.dtype))
    indicies_tensor = onnx.helper.make_tensor(tensor.name + '_indicies',
                                              TensorProto.INT64,
                                              [len(indicies)], np.array(indicies).astype(np.int64))
    sparse_tensor = onnx.helper.make_sparse_tensor(values_tensor, indicies_tensor, tensor.dims)
    return (sparse_tensor, sparsity)


def convert_initializers(model,
                         exclude_names,
                         sparsity_threshold,
                         tolerance):  # type: (ModelProto, List[str], float) -> None
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
        sparse_tensor, sparsity = convert_tensor_to_sparse(initializer, tolerance)
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

    exclude_names = set() if args.exclude is None else set(args.exclude.split(';'))

    model = ModelProto()
    with open(args.input, "rb") as input_file:
        model.ParseFromString(input_file.read())

    convert_initializers(model, exclude_names, args.sparsity_threshold, args.tolerance)

    with open(args.output, "wb") as output_file:
        s = model.SerializeToString()
        output_file.write(s)


if __name__ == "__main__":
    main()
