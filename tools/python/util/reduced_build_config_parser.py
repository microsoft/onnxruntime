# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

# Check if the flatbuffers module is available. If not we cannot handle type reduction information in the config.
try:
    import flatbuffers  # noqa
    have_flatbuffers = True
    from .ort_format_model import OperatorTypeUsageManager  # noqa
except ImportError:
    have_flatbuffers = False


def parse_config(config_file: str, enable_type_reduction: bool = False):
    '''
    Parse the configuration file and return the required operators dictionary, and possibly either an
    OperatorTypeUsageManager or a globally allowed types list.

    Configuration file lines can do the following:
    1. specify required operators
    2. specify globally allowed types for all operators

    The basic format for specifying required operators (1) is `domain;opset;op1,op2...`
    e.g. `ai.onnx;11;Add,Cast,Clip,...

    If the configuration file is generated from ORT format models it may optionally contain JSON for per-operator
    type reduction. The required types are generally listed per input and/or output of the operator.
    The type information is in a map, with 'inputs' and 'outputs' keys. The value for 'inputs' or 'outputs' is a map
    between the index number of the input/output and the required list of types.

    For example, both the input and output types are relevant to ai.onnx:Cast.
    Type information for input 0 and output 0 could look like this:
        `{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}}`

    which is added directly after the operator name in the configuration file.
    e.g.
        `ai.onnx;12;Add,Cast{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}},Concat`

    If for example the types of inputs 0 and 1 were important, the entry may look like this (e.g. ai.onnx:Gather):
        `{"inputs": {"0": ["float", "int32_t"], "1": ["int32_t"]}}`

    Finally some operators do non-standard things and store their type information under a 'custom' key.
    ai.onnx.OneHot is an example of this, where 3 type names from the inputs are combined into a string.
        `{"custom": ["float_int64_t_int64_t", "int64_t_string_int64_t"]}`

    The format for specifying globally allowed types for all operators (2) is `!globally_allowed_types;T0,T1,...`
    Ti should be a C++ scalar type supported by ONNX and ORT.
    At most one globally allowed types specification is allowed.

    Specifying per-operator type information and specifying globally allowed types are mutually exclusive - it is an
    error to specify both.

    :param config_file: Configuration file to parse
    :param enable_type_reduction: Set to True to use the type information in the config.
                                  If False the type information will be ignored.
                                  If the flatbuffers module is unavailable op-specific type information will be ignored
                                  as the per-op usage of the type information is via OperatorTypeUsageManager which has
                                  a dependency on the ORT flatbuffers python schema.
    :return: required_ops, op_type_usage_manager, globally_allowed_types:
             Dictionary of domain:opset:[ops] for required operators
             OperatorTypeUsageManager manager with operator specific type usage information if available. None if
             type reduction was disabled, per-op type reduction was not specified, or the flatbuffers module is not
             available.
             List of globally allowed types. None if type reduction was disabled, or globally allowed types were not
             specified.
             At most one of the op_type_usage_manager and globally_allowed_types tuple elements will not be None.
    '''

    if not os.path.isfile(config_file):
        raise ValueError('Configuration file {} does not exist'.format(config_file))

    required_ops = {}
    op_type_usage_manager = OperatorTypeUsageManager() if enable_type_reduction and have_flatbuffers else None
    has_op_type_reduction_info = False
    globally_allowed_types = None

    with open(config_file, 'r') as config:
        for line in [orig_line.strip() for orig_line in config.readlines()]:
            if not line or line.startswith("#"):  # skip empty lines and comments
                continue

            if line.startswith("!globally_allowed_types;"):  # handle globally allowed types
                if enable_type_reduction:
                    if globally_allowed_types is not None:
                        raise RuntimeError("Globally allowed types were already specified.")
                    globally_allowed_types = [segment.strip() for segment in line.split(';')[1].split(',')]
                continue

            domain, opset_str, operators_str = [segment.strip() for segment in line.split(';')]
            opset = int(opset_str)

            # any type reduction information is serialized json that starts/ends with { and }.
            # type info is optional for each operator.
            if '{' in operators_str:
                has_op_type_reduction_info = True

                # parse the entries in the json dictionary with type info
                operators = set()
                cur = 0
                end = len(operators_str)
                while cur < end:
                    next_comma = operators_str.find(',', cur)
                    next_open_brace = operators_str.find('{', cur)

                    if next_comma == -1:
                        next_comma = end

                    # the json string starts with '{', so if that is found (next_open_brace != -1)
                    # before the next comma (which would be the start of the next operator if there is no type info
                    # for the current operator), we have type info to parse.
                    # e.g. need to handle extracting the operator name and type info for OpB and OpD,
                    #      and just the operator names for OpA and OpC from this example string
                    #      OpA,OpB{"inputs": {"0": ["float", "int32_t"]}},OpC,OpD{"outputs": {"0": ["int32_t"]}}
                    if 0 < next_open_brace < next_comma:
                        operator = operators_str[cur:next_open_brace].strip()
                        operators.add(operator)

                        # parse out the json dictionary with the type info by finding the closing brace that matches
                        # the opening brace
                        i = next_open_brace + 1
                        num_open_braces = 1
                        while num_open_braces > 0 and i < end:
                            if operators_str[i] == '{':
                                num_open_braces += 1
                            elif operators_str[i] == '}':
                                num_open_braces -= 1
                            i += 1

                        if num_open_braces != 0:
                            raise RuntimeError('Mismatched { and } in type string: ' + operators_str[next_open_brace:])

                        if op_type_usage_manager:
                            type_str = operators_str[next_open_brace:i]
                            op_type_usage_manager.restore_from_config_entry(domain, operator, type_str)

                        cur = i + 1
                    else:
                        # comma or end of line is next
                        end_str = next_comma if next_comma != -1 else end
                        operators.add(operators_str[cur:end_str].strip())
                        cur = end_str + 1

            else:
                operators = set([op.strip() for op in operators_str.split(',')])

            if domain not in required_ops:
                required_ops[domain] = {opset: operators}
            elif opset not in required_ops[domain]:
                required_ops[domain][opset] = operators
            else:
                required_ops[domain][opset].update(operators)

    if enable_type_reduction:
        if not has_op_type_reduction_info:
            op_type_usage_manager = None
        if globally_allowed_types is not None and op_type_usage_manager is not None:
            raise RuntimeError(
                "Specifying globally allowed types and per-op type reduction info together is unsupported.")

    return required_ops, op_type_usage_manager, globally_allowed_types
