# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import ort_flatbuffers_py.experimental.fbs as fbs

from abc import ABC, abstractmethod
from .types import value_name_to_typestr


def _create_op_key(domain: str, optype: str):
    return '{}:{}'.format(domain, optype)


def _ort_constant_for_domain(domain: str):
    '''
    Map a string domain value to the internal ONNX Runtime constant for that domain.
    :param domain: Domain string to map.
    :return: Internal ONNX Runtime constant
    '''

    # constants are defined in <ORT root>/include/onnxruntime/core/graph/constants.h
    # This list is limited to just the domains we have processors for
    domain_to_constant_map = {'ai.onnx': 'kOnnxDomain',
                              'ai.onnx.ml': 'kMLDomain',
                              'com.microsoft': 'kMSDomain'}

    if domain not in domain_to_constant_map:
        raise ValueError('Domain {} not found in map to ONNX Runtime constant. Please update map.'.format(domain))

    return domain_to_constant_map[domain]


class TypeUsageProcessor(ABC):
    '''
    Abstract base class for processors which implement operator specific logic to determine the type or types required.
    '''
    def __init__(self, domain: str, optype: str):
        self.domain = domain
        self.optype = optype
        self.name = _create_op_key(domain, optype)

    @abstractmethod
    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        pass

    def is_typed_registration_needed(self, type_in_registration):
        '''
        Given the string from a kernel registration, determine if the registration is required or not.
        :param type_in_registration: Type string from kernel registration
        :return: True is required. False if not.
        '''
        # Not all operators have typed registrations, so this is optionally implemented by derived classes
        raise RuntimeError('Did not expect processor for {} to have typed registrations.'.format(self.name))

    @abstractmethod
    def get_cpp_entry(self):
        '''
        Get the C++ code that specifies this operator's required types.
        :return: List with any applicable C++ code for this operator's required types. One line per entry.
        '''
        pass

    @abstractmethod
    def to_config_entry(self):
        '''
        Generate a configuration file entry in JSON format with the required types for the operator.
        :return: JSON string with required type information.
        '''
        pass

    @abstractmethod
    def from_config_entry(self, entry: str):
        '''
        Re-create the types required from a configuration file entry created with to_config_entry.
        NOTE: Any existing type information should be cleared prior to re-creating from a config file entry.
        :param entry: Configuration file entry
        '''
        pass


class DefaultTypeUsageProcessor(TypeUsageProcessor):
    '''
    Operator processor which tracks the types used for selected input/s and/or output/s.
    '''

    def __init__(self, domain: str, optype: str, inputs: [int] = [0], outputs: [int] = []):
        '''
        Create DefaultTypeUsageProcessor. Types for one or more inputs and/or outputs can be tracked by the processor.
        The default is to track the types required for input 0, as this is the most common use case in ONNX.
        :param domain: Operator domain.
        :param optype: Operator name.
        :param inputs: Inputs to track. Zero based index. May be empty.
        :param outputs: Outputs to track. Zero based index. May be empty.
        '''
        super().__init__(domain, optype)
        self._input_types = {}
        self._output_types = {}

        for i in inputs:
            self._input_types[i] = set()

        for o in outputs:
            self._output_types[o] = set()

        if not inputs and not outputs:
            raise ValueError('At least one input or output must be tracked')

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        for i in self._input_types.keys():
            if i >= node.InputsLength():
                # Some operators have fewer inputs in earlier versions where data that was as an attribute
                # become an input in later versions to allow it to be dynamically provided. Allow for that.
                # e.g. Slice-1 had attributes for the indices, and Slice-10 moved those to be inputs
                # raise RuntimeError('Node has {} outputs. Tracker for {} incorrectly configured as it requires {}.'
                #                    .format(node.OutputsLength(), self.name, o))
                pass
            else:
                type_str = value_name_to_typestr(node.Inputs(i), value_name_to_typeinfo)
                self._input_types[i].add(type_str)

        for o in self._output_types.keys():
            # Don't know of any ops where the number of outputs changed across versions, so require a valid length
            if o >= node.OutputsLength():
                raise RuntimeError('Node has {} outputs. Tracker for {} incorrectly configured as it requires {}.'
                                   .format(node.OutputsLength(), self.name, o))

            type_str = value_name_to_typestr(node.Outputs(o), value_name_to_typeinfo)
            self._output_types[o].add(type_str)

    def is_typed_registration_needed(self, type_in_registration: str):
        if 0 not in self._input_types.keys():
            # currently all standard typed registrations are for input 0.
            # custom registrations can be handled by operator specific processors (e.g. OneHotProcessor below).
            raise RuntimeError('Expected typed registration to use type from input 0. Node:{}'.format(self.name))

        return type_in_registration in self._input_types[0]

    def get_cpp_entry(self):
        entries = []
        domain = _ort_constant_for_domain(self.domain)
        for i in sorted(self._input_types.keys()):
            if self._input_types[i]:
                entries.append('ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES({}, {}, Input, {}, {});'
                               .format(domain, self.optype, i, ', '.join(sorted(self._input_types[i]))))

        for o in sorted(self._output_types.keys()):
            if self._output_types[o]:
                entries.append('ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES({}, {}, Output, {}, {});'
                               .format(domain, self.optype, o, ', '.join(sorted(self._output_types[o]))))

        return entries

    def to_config_entry(self):
        # convert the sets of types to lists so they can easily written out using the json model
        aggregate_info = {'inputs': {}, 'outputs': {}}

        # filter out empty entries and sort the types
        for i in sorted(self._input_types.keys()):
            if self._input_types[i]:
                aggregate_info['inputs'][i] = sorted(self._input_types[i])

        for o in sorted(self._output_types.keys()):
            if self._output_types[o]:
                aggregate_info['outputs'][o] = sorted(self._output_types[o])

        # remove any empty keys
        if not aggregate_info['inputs']:
            aggregate_info.pop('inputs')
        if not aggregate_info['outputs']:
            aggregate_info.pop('outputs')

        entry = json.dumps(aggregate_info) if aggregate_info else None
        return entry

    def from_config_entry(self, entry: str):
        self._input_types.clear()
        self._output_types.clear()

        aggregate_info = json.loads(entry)
        if 'inputs' in aggregate_info:
            for i_str, values in aggregate_info['inputs'].items():
                self._input_types[int(i_str)] = set(values)

        if 'outputs' in aggregate_info:
            for o_str, values in aggregate_info['outputs'].items():
                self._output_types[int(o_str)] = set(values)


class Output0TypedRegistrationProcessor(DefaultTypeUsageProcessor):
    '''
    Processor for operators where the first output type is used in a typed kernel registration.
    '''
    def __init__(self, domain: str, optype: str):
        # init with tracking of output 0 only.
        super().__init__(domain, optype, inputs=[], outputs=[0])

    def is_typed_registration_needed(self, type_in_registration: str):
        return type_in_registration in self._output_types[0]


class OneHotProcessor(TypeUsageProcessor):
    '''
    Processor for the OneHot operator, which requires custom logic as the type registration key is a concatenation of
    the three types involved instead of a single type name.
    '''
    def __init__(self):
        super().__init__('ai.onnx', 'OneHot')
        self._triples = set()

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        type0 = value_name_to_typestr(node.Inputs(0), value_name_to_typeinfo)
        type1 = value_name_to_typestr(node.Inputs(1), value_name_to_typeinfo)
        type2 = value_name_to_typestr(node.Inputs(2), value_name_to_typeinfo)
        key = '{}_{}_{}'.format(type0, type1, type2)
        self._triples.add(key)

    def is_typed_registration_needed(self, type_in_registration):
        # the OneHot registration involves a concatenation of the 3 types involved, in the format we match
        # when adding values in process_node
        return type_in_registration in self._triples

    def get_cpp_entry(self):
        # exclusion is via commenting out the registration entry, so don't need to write any #defines
        # to disable type support for the OneHot operator
        return None

    def to_config_entry(self):
        if not self._triples:
            return None

        aggregate_info = {'custom': sorted(self._triples)}
        entry = json.dumps(aggregate_info)
        return entry

    def from_config_entry(self, entry: str):
        self._triples.clear()
        aggregate_info = json.loads(entry)
        if 'custom' in aggregate_info:
            self._triples = set(aggregate_info['custom'])


def _create_operator_type_usage_processors():
    '''
    Create a set of processors that determine the required types for all enabled operators.
    :return: Dictionary of operator key to processor. Key is 'domain:operator (e.g. ai.onnx:Cast)'.
    '''
    operator_processors = {}

    def add(processor):
        if processor.name in operator_processors:
            raise RuntimeError('Duplicate processor for ' + processor.name)

        operator_processors[processor.name] = processor

    # Starting with ops from:
    #   - Priority 1P models
    #   - Mobilenet + SSD Mobilenet + MobileBert
    #   - some known large kernels
    #
    # Ops we are ignoring currently so as not to produce meaningless/unused output:
    # - Implementation is type agnostic:
    #    ai.onnx: If, Loop, Reshape, Scan, Shape, Squeeze, Unsqueeze
    #    com.microsoft: DynamicQuantizeMatMul, MatMulIntegerToFloat
    # - Only one type supported in the ORT implementation:
    #    com.microsoft: FusedConv, FusedGemm, FusedMatMul, TransposeMatMul
    # - Implementation does not have any significant type specific code:
    #    ai.onnx: Concat, Flatten, Not, QLinearConv, Reshape, Shape, Squeeze, Unsqueeze
    #
    default_processor_onnx_ops = ['Abs', 'Add', 'ArgMax', 'ArgMin', 'AveragePool',
                                  'BatchNormalization', 'BitShift',
                                  'Ceil', 'Clip', 'Conv', 'CumSum',
                                  'DequantizeLinear', 'Div',
                                  'Equal', 'Exp', 'Expand',
                                  'Floor',
                                  'Gemm', 'Greater',
                                  'IsNaN'
                                  'Less', 'Log', 'LogSoftmax', 'LpNormalization',
                                  'MatMul', 'Max', 'Min', 'Mul',
                                  'Neg', 'NonMaxSuppression', 'NonZero',
                                  'Pad',
                                  'Range', 'Reciprocal', 'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp',
                                  'ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd', 'ReduceSum', 'ReduceSumSquare',
                                  'Relu', 'Resize', 'RoiAlign', 'Round',
                                  'Sigmoid', 'Sin', 'Softmax', 'Split', 'Sqrt', 'Sub',
                                  'Tanh', 'Tile', 'TopK', 'Transpose',
                                  'Where']

    internal_ops = ['QLinearAdd', 'QLinearMul']

    # TODO - review and add ML ops as needed
    # ML Op notes.
    #  CastMap: Switch on value type of input map type, and output type
    #  DictVectorizer: Templatized on key+value of input so need to handle like OneHot with custom processor
    #  LabelEncoder: Implementation switches on input and output types (only supports string and int64 in T1 and T2)
    #  LinearClassifier: Internal switch on input type and also switch on output type
    #  SVMClassifier: ditto
    #  TreeEnsembleClassifier: Templatized on input type and also switch on output type
    #  ZipMap: Switch on output type (derived from attributes)
    default_processor_onnxml_ops = []

    [add(DefaultTypeUsageProcessor('ai.onnx', op)) for op in default_processor_onnx_ops]
    [add(DefaultTypeUsageProcessor('ai.onnx.ml', op)) for op in default_processor_onnxml_ops]
    [add(DefaultTypeUsageProcessor('com.microsoft', op)) for op in internal_ops]

    #
    # Operators that require custom handling
    #

    # Cast switches on types of input 0 and output 0
    add(DefaultTypeUsageProcessor('ai.onnx', 'Cast', inputs=[0], outputs=[0]))

    # Operators that switch on the type of input 0 and 1
    add(DefaultTypeUsageProcessor('ai.onnx', 'Gather', inputs=[0, 1]))
    add(DefaultTypeUsageProcessor('ai.onnx', 'GatherElements', inputs=[0, 1]))
    add(DefaultTypeUsageProcessor('ai.onnx', 'Pow', inputs=[0, 1]))
    add(DefaultTypeUsageProcessor('ai.onnx', 'Slice', inputs=[0, 1]))

    # Operators that switch on output type
    add(DefaultTypeUsageProcessor('ai.onnx', 'ConstantOfShape', inputs=[], outputs=[0]))

    # Random generator ops produce new data so we track the output type
    onnx_random_ops = ['RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike', 'Multinomial']
    [add(DefaultTypeUsageProcessor('ai.onnx', op, inputs=[], outputs=[0])) for op in onnx_random_ops]

    # we only support 'float' as input for [Dynamic]QuantizeLinear so just track the output type
    # as that's what is used in the typed registration
    add(Output0TypedRegistrationProcessor('ai.onnx', 'QuantizeLinear'))
    add(Output0TypedRegistrationProcessor('ai.onnx', 'DynamicQuantizeLinear'))

    # OneHot concatenates type strings into a triple in the typed registration
    #   e.g. float_int64_t_int64_t
    add(OneHotProcessor())

    return operator_processors


class OperatorTypeUsageManager:
    '''
    Class to manage the operator type usage processors.
    TODO: Currently the type tracking is not specific to a version of the operator.
    It's unclear how/where version specific logic could/should be added, and it would add significant complexity
    to track types on a per-version basis. Not clear there's enough benefit from doing so either.
    '''
    def __init__(self):
        self._all_operator_processors = _create_operator_type_usage_processors()  # all possible processors
        self._operator_processors = {}  # processors we have actually used so we can limit output to be meaningful

    def _get_op_processor(self, key):
        'Add the processor to _operator_processors as it is about to be used.'
        processor = None
        if key in self._all_operator_processors:
            if key not in self._operator_processors:
                self._operator_processors[key] = self._all_operator_processors[key]

            processor = self._operator_processors[key]

        return processor

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        '''
        Process a Node and record info on the types used.
        :param node: Node from ORT format model
        :param value_name_to_typeinfo: Map of value names to TypeInfo instances
        '''
        optype = node.OpType().decode()
        domain = node.Domain().decode() or 'ai.onnx'  # empty domain defaults to ai.onnx

        key = _create_op_key(domain, optype)
        op_processor = self._get_op_processor(key)
        if op_processor:
            op_processor.process_node(node, value_name_to_typeinfo)

    def is_typed_registration_needed(self, domain: str, optype: str, type_registration_str: str):
        '''
        Given the string from a kernel registration, determine if the registration is required or not.
        :param domain: Operator domain.
        :param optype: Operator type.
        :param type_registration_str: Type string from kernel registration
        :return: True is required. False if not.
        '''
        needed = True  # we keep the registration unless the per-operator processor says not to
        key = _create_op_key(domain, optype)
        if key in self._operator_processors:
            needed = self._operator_processors[key].is_typed_registration_needed(type_registration_str)

        return needed

    def get_cpp_entries(self):
        '''
        Get the C++ code that define the lists of types to enable for the operators we have type info for.
        :return: List of strings. One line of C++ code per entry.
        '''
        entries = []
        for key in sorted(self._operator_processors.keys()):
            entries.extend(self._operator_processors[key].get_cpp_entry())

        return entries

    def get_config_entry(self, domain: str, optype: str):
        '''
        Get the config entry specifying the types for this operator.
        :param domain: Operator domain.
        :param optype: Operator type.
        :return: JSON string with type info if available, else None
        '''
        key = _create_op_key(domain, optype)
        config_str = None
        if key in self._operator_processors:
            config_str = self._operator_processors[key].to_config_entry()

        return config_str

    def restore_from_config_entry(self, domain: str, optype: str, config_entry: str):
        '''
        Restore the per-operator type information from a configuration file entry.
        :param domain: Operator domain.
        :param optype: Operator type.
        :param config_entry: JSON string with type info as created by get_config_entry
        '''
        key = _create_op_key(domain, optype)
        op_processor = self._get_op_processor(key)
        if op_processor:
            op_processor.from_config_entry(config_entry)

    def debug_dump(self):

        print('C++ code that will be emitted:')
        [print(cpp_line) for cpp_line in self.get_cpp_entries()]

        print('Config file type information that will be returned by get_config_entry:')
        for key in sorted(self._operator_processors.keys()):
            entry = self._operator_processors[key].to_config_entry()
            if entry:
                print('{} -> {}'.format(key, entry))

                # roundtrip test to validate that we can initialize the processor from the entry and get the
                # same values back
                self._operator_processors[key].from_config_entry(entry)
                assert (entry == self._operator_processors[key].to_config_entry())
