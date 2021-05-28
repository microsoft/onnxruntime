# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from torch.onnx import symbolic_helper


def _export(g, n, *args, **kwargs):
    try:
        name = kwargs['name']
        inplace = kwargs['inplace']
        training_mode = symbolic_helper._training_mode
        cconv = n.cconv()
        input_tensor_types = []
        input_tensor_requires_grads = []
        input_tensor_ranks = []

        input_int_scalars = []
        input_int_scalar_positions = []

        input_float_scalars = []
        input_float_scalar_positions = []

        input_int_tuples = []
        input_int_tuple_positions = []
        input_int_tuple_begins = []

        input_float_tuples = []
        input_float_tuple_positions = []
        input_float_tuple_begins = []

        input_pointer_scalars = []
        input_pointer_scalar_positions = []

        tensor_args = []
        for i, arg, call_type in zip(range(len(args)), args, cconv):
            if call_type == 'd':
                # Got a tensor variable.
                tensor_args.append(arg)

                requires_grad = 1 if arg.requires_grad() else 0
                input_tensor_requires_grads.append(requires_grad)

                scalar_type = int(symbolic_helper.cast_pytorch_to_onnx[arg.type(
                ).scalarType()])
                input_tensor_types.append(scalar_type)
                input_tensor_ranks.append(arg.type().dim())
            elif call_type == 'c':
                # Got a non-tensor variable.
                # Non-tensor can't have gradient.
                input_tensor_requires_grads.append(0)
                if isinstance(arg, float):
                    # A float.
                    input_float_scalar_positions.append(i)
                    input_float_scalars.append(arg)
                elif isinstance(arg, int):
                    # A int.
                    input_int_scalar_positions.append(i)
                    input_int_scalars.append(arg)
                elif isinstance(arg, tuple):
                    assert len(arg) > 0
                    # A tuple of int or float.
                    if all(isinstance(ele, int) for ele in arg):
                        # A tuple of ints.
                        input_int_tuple_positions.append(i)
                        input_int_tuple_begins.append(len(input_int_tuples))
                        input_int_tuples.extend(list(arg))
                    elif all(isinstance(ele, float) for ele in arg):
                        # A tuple of floats.
                        input_float_tuple_positions.append(i)
                        input_float_tuple_begins.append(
                            len(input_float_tuples))
                        input_float_tuples.extend(list(arg))
                    else:
                        raise Exception(
                            f'Unknown argument type found: {type(arg)}.')
                else:
                    # All other inputs are accessed via "pointers".
                    input_pointer_scalar_positions.append(i)
                    input_pointer_scalars.append(id(arg))
            else:
                raise Exception(
                    f'Unknown calling convention found: {i}. Only \'d\' and \'c\' are supported')

        output_tensor_types = []
        output_tensor_ranks = []
        output_tensor_requires_grads = []
        for arg in n.outputs():
            # Type of tensor's elements.
            scalar_type = int(symbolic_helper.cast_pytorch_to_onnx[arg.type(
            ).scalarType()])
            output_tensor_types.append(scalar_type)
            output_tensor_ranks.append(arg.type().dim())
            # If output has gradient.
            requires_grad = 1 if arg.requires_grad() else 0
            output_tensor_requires_grads.append(requires_grad)

        # TODO: add fully-qualified name.
        attrs = {
            'name_s': name,
            'inplace_i': inplace,
            'call_convention_s': cconv,
            'outputs': n.outputsSize(),
            'input_tensor_types_i': input_tensor_types,
            'input_tensor_ranks_i': input_tensor_ranks,
            'input_tensor_requires_grads_i': input_tensor_requires_grads,
            'output_tensor_types_i': output_tensor_types,
            'output_tensor_ranks_i': output_tensor_ranks,
            'output_tensor_requires_grads_i': output_tensor_requires_grads,
            'training_mode_i': 1 if training_mode else 0
        }

        if len(input_int_scalars) > 0:
            attrs['input_int_scalars_i'] = input_int_scalars
            attrs['input_int_scalar_positions_i'] = input_int_scalar_positions
        if len(input_float_scalars) > 0:
            attrs['input_float_scalars_f'] = input_float_scalars
            attrs['input_float_scalar_positions_i'] = input_float_scalar_positions
        if len(input_int_tuples) > 0:
            attrs['input_int_tuples_i'] = input_int_tuples
            attrs['input_int_tuple_positions_i'] = input_int_tuple_positions
            attrs['input_int_tuple_begins_i'] = input_int_tuple_begins
        if len(input_float_tuples) > 0:
            attrs['input_float_tuples_f'] = input_float_tuples
            attrs['input_float_tuple_positions_i'] = input_float_tuple_positions
            attrs['input_float_tuple_begins_i'] = input_float_tuple_begins
        if len(input_pointer_scalars) > 0:
            attrs['input_pointer_scalars_i'] = input_pointer_scalars
            attrs['input_pointer_scalar_positions_i'] = input_pointer_scalar_positions

        returned_args = g.op("com.microsoft::PythonOp", *tensor_args, **attrs)

        return returned_args
    except:
        sys.stdout.flush()
        sys.stderr.flush()
        raise
