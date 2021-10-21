import cerberus
from collections import namedtuple
import torch
from ._utils import static_vars


LEARNING_RATE_IO_DESCRIPTION_NAME = "__learning_rate"
ALL_FINITE_IO_DESCRIPTION_NAME = "__all_finite"
LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME = "__loss_scale_input_name"
GRADIENT_ACCUMULATION_IO_DESCRIPTION_NAME = "__gradient_accumulation_name"


class _ORTTrainerModelDesc(object):

    def __init__(self, model_desc):
        # Keep a copy of original input for debug
        self._original = dict(model_desc)

        # Global counter used to validate occurrences of 'is_loss=True' whithin 'model_desc.outputs'
        #   A stateless validator is used for each tuple, but validation accross the whole list of tuple is needed
        #       because just one 'is_loss=True' is allowed withing 'model_desc.outputs' list of tuples
        _model_desc_outputs_validation.loss_counter = 0

        # Used for logging purposes
        self._main_class_name = self.__class__.__name__

        # Validates user input
        self._validated = dict(self._original)
        validator = cerberus.Validator(MODEL_DESC_SCHEMA)
        self._validated = validator.validated(self._validated)
        if self._validated is None:
            raise ValueError(f'Invalid model_desc: {validator.errors}')

        # Normalize inputs to a list of namedtuple(name, shape)
        self._InputDescription = namedtuple('InputDescription', ['name', 'shape'])
        self._InputDescriptionTyped = namedtuple('InputDescriptionTyped', ['name', 'shape', 'dtype'])
        for idx, input in enumerate(self._validated['inputs']):
            self._validated['inputs'][idx] = self._InputDescription(*input)

        # Normalize outputs to a list of namedtuple(name, shape, is_loss)
        self._OutputDescription = namedtuple('OutputDescription', ['name', 'shape', 'is_loss'])
        self._OutputDescriptionTyped = namedtuple('OutputDescriptionTyped',
                                                  ['name', 'shape', 'is_loss', 'dtype', 'dtype_amp'])
        for idx, output in enumerate(self._validated['outputs']):
            if len(output) == 2:
                self._validated['outputs'][idx] = self._OutputDescription(*output, False)
            else:
                self._validated['outputs'][idx] = self._OutputDescription(*output)

        # Hard-code learning rate, all_finite descriptors
        self.learning_rate = self._InputDescriptionTyped(LEARNING_RATE_IO_DESCRIPTION_NAME, [1], torch.float32)

        # Convert dict in object
        for k, v in self._validated.items():
            setattr(self, k, self._wrap(v))

    def __repr__(self):
        '''Pretty representation for a model description class'''

        pretty_msg = 'Model description:\n'

        # Inputs
        inputs = []
        for i_desc in self.inputs:
            if isinstance(i_desc, self._InputDescription):
                inputs.append(f'(name={i_desc.name}, shape={i_desc.shape})')
            elif isinstance(i_desc, self._InputDescriptionTyped):
                inputs.append(f'(name={i_desc.name}, shape={i_desc.shape}, dtype={i_desc.dtype})')
            else:
                raise ValueError(f'Unexpected type {type(i_desc)} for input description')

        pretty_msg += '\nInputs:'
        for idx, item in enumerate(inputs):
            pretty_msg += f'\n\t{idx}: {item}'

        # Outputs
        outputs = []
        for o_desc in self.outputs:
            if isinstance(o_desc, self._OutputDescription):
                outputs.append(f'(name={o_desc.name}, shape={o_desc.shape})')
            elif isinstance(o_desc, self._OutputDescriptionTyped):
                outputs.append(f'(name={o_desc.name}, shape={o_desc.shape}, dtype={o_desc.dtype}, dtype_amp={o_desc.dtype_amp})')
            else:
                raise ValueError(f'Unexpected type {type(o_desc)} for output description')
        pretty_msg += '\nOutputs:'
        for idx, item in enumerate(outputs):
            pretty_msg += f'\n\t{idx}: {item}'

        # Learning rate
        if self.learning_rate:
            pretty_msg += '\nLearning rate: '
            pretty_msg += f'(name={self.learning_rate.name}, shape={self.learning_rate.shape}, dtype={self.learning_rate.dtype})'

        # Mixed precision
        if getattr(self, ALL_FINITE_IO_DESCRIPTION_NAME, None) or getattr(self, LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME, None):
            pretty_msg += '\nMixed Precision:'
            if getattr(self, ALL_FINITE_IO_DESCRIPTION_NAME, None):
                pretty_msg += '\n\tis gradients finite: '
                pretty_msg += f'(name={self.all_finite.name}, shape={self.all_finite.shape}, dtype={self.all_finite.dtype})'
            if getattr(self, LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME, None):
                pretty_msg += '\n\tloss scale input name: '
                pretty_msg += f'(name={self.loss_scale_input.name}, shape={self.loss_scale_input.shape}, dtype={self.loss_scale_input.dtype})'

        # Gradient Accumulation steps
        if self.gradient_accumulation:
            pretty_msg += '\nGradient Accumulation: '
            pretty_msg += f'(name={self.gradient_accumulation.name}, shape={self.gradient_accumulation.shape}, dtype={self.gradient_accumulation.dtype})'

        return pretty_msg

    def add_type_to_input_description(self, index, dtype):
        '''Updates an existing input description at position 'index' with 'dtype' type information

        Args:
            index (int): position within 'inputs' description
            dtype (torch.dtype): input data type
        '''

        assert isinstance(index, int) and index >= 0,\
            "input 'index' must be a positive int"
        assert isinstance(dtype, torch.dtype),\
            "input 'dtype' must be a torch.dtype type"
        existing_values = (*self.inputs[index],)
        if isinstance(self.inputs[index], self._InputDescriptionTyped):
            existing_values = (*existing_values[:-1],)
        self.inputs[index] = self._InputDescriptionTyped(*existing_values, dtype)

    def add_type_to_output_description(self, index, dtype, dtype_amp=None):
        '''Updates an existing output description at position 'index' with 'dtype' type information

        Args:
            index (int): position within 'inputs' description
            dtype (torch.dtype): input data type
            dtype_amp (torch.dtype, default is None): input data type for evaluation with mixed precision
        '''

        assert isinstance(index, int) and index >= 0,\
            "output 'index' must be a positive int"
        assert isinstance(dtype, torch.dtype),\
            "output 'dtype' must be a torch.dtype type"
        assert dtype_amp is None or isinstance(dtype_amp, torch.dtype),\
            "output 'dtype_amp' must be either None or torch.dtype type"
        existing_values = (*self.outputs[index],)
        if isinstance(self.outputs[index], self._OutputDescriptionTyped):
            existing_values = (*existing_values[:-2],)
        self.outputs[index] = self._OutputDescriptionTyped(*existing_values, dtype, dtype_amp)

    @property
    def gradient_accumulation(self):
        return getattr(self, GRADIENT_ACCUMULATION_IO_DESCRIPTION_NAME, None)

    @gradient_accumulation.setter
    def gradient_accumulation(self, name):
        self._add_output_description(self, name, [1], False, torch.bool, None, GRADIENT_ACCUMULATION_IO_DESCRIPTION_NAME, ignore_duplicate=True)

    @property
    def all_finite(self):
        return getattr(self, ALL_FINITE_IO_DESCRIPTION_NAME, None)

    @all_finite.setter
    def all_finite(self, name):
        self._add_output_description(self, name, [1], False, torch.bool, None, ALL_FINITE_IO_DESCRIPTION_NAME, ignore_duplicate=True)

    @property
    def loss_scale_input(self):
        return getattr(self, LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME, None)

    @loss_scale_input.setter
    def loss_scale_input(self, name):
        self._add_input_description(self, name, [], torch.float32, LOSS_SCALE_INPUT_IO_DESCRIPTION_NAME, ignore_duplicate=True)

    def _add_input_description(self, node, name, shape, dtype=None, attr_name=None, ignore_duplicate=False):
        '''Add a new input description into the node object

        If 'dtype' is specified, a typed input description namedtuple(name, shape, dtype) is created.
        Otherwise an untyped input description namedtuple(name, shape) is created instead.

        Args:
            node (list or object): node to append input description to. When 'node' is 'self.inputs',
                a new input description is appended to the list.
                Otherwise, a new input description is created as an attribute into 'node' with name 'attr_name'
            name (str): name of input description
            shape (list): shape of input description
            dtype (torch.dtype): input data type
            attr_name (str, default is None): friendly name to allow direct access to the output description
            ignore_duplicate (bool, default is False): silently skips addition of duplicate inputs
        '''

        assert isinstance(name, str) and len(name) > 0, "'name' is an invalid input name"
        not_found = True
        if not ignore_duplicate:
            if id(node) == id(self.inputs):
                not_found = all([name not in i_desc.name for i_desc in node])
                assert not_found, f"'name' {name} already exists in the inputs description"
            else:
                not_found = attr_name not in dir(self)
                assert not_found, f"'attr_name' {attr_name} already exists in the 'node'"
        elif not not_found:
            return
        assert isinstance(shape, list) and all([(isinstance(dim, int) or (isinstance(dim, str) and len(dim) > 0))\
            for dim in shape]), "'shape' must be a list of int or str with length at least 1"
        assert dtype is None or isinstance(dtype, torch.dtype), "'dtype' must be either None or a torch.dtype type"
        if dtype:
            new_input_desc = self._InputDescriptionTyped(name, shape, dtype)
        else:
            new_input_desc = self._InputDescription(name, shape)

        if id(node) == id(self.inputs):
            self.inputs.append(new_input_desc)
        else:
            assert isinstance(attr_name, str) and len(attr_name) > 0, "Invalid 'attr_name'"
            setattr(node, attr_name, new_input_desc)

    def _add_output_description(self, node, name, shape, is_loss, dtype=None, dtype_amp=None, attr_name=None, ignore_duplicate=False):
        '''Add a new output description into the node object as a tuple

        When (name, shape, is_loss, dtype) is specified, a typed output description is created
        Otherwise an untyped output description (name, shape, is_loss) is created instead

        Args:
            node (list or object): node to append output description to. When 'node' is 'self.outputs',
                a new output description is appended to the list.
                Otherwise, a new output description is created as an attribute into 'node' with name 'attr_name'
            name (str): name of output description
            shape (list): shape of output description
            is_loss (bool): specifies whether this output is a loss
            dtype (torch.dtype): input data type
            dtype_amp (torch.dtype, default is None): input data type for evaluation with mixed precision.
            attr_name (str, default is None): friendly name to allow direct access to the output description
            ignore_duplicate (bool, default is False): silently skips addition of duplicate outputs
        '''

        assert isinstance(name, str) and len(name) > 0, "'name' is an invalid output name"
        assert isinstance(shape, list) and all([(isinstance(dim, int) or (isinstance(dim, str) and len(dim) > 0))\
            for dim in shape]), "'shape' must be a list of int or str with length at least 1"
        assert isinstance(is_loss, bool), "'is_loss' must be a bool"

        not_found = True
        if not ignore_duplicate:
            if id(node) == id(self.outputs):
                not_found = all([name not in o_desc.name for o_desc in node])
                assert not_found, f"'name' {name} already exists in the outputs description"
                assert all([not o_desc.is_loss for o_desc in node]) if is_loss else True,\
                    "Only one 'is_loss' is supported at outputs description"
            else:
                not_found = attr_name not in dir(self)
                assert not_found, f"'attr_name' {attr_name} already exists in the 'node'"
        elif not not_found:
            return

        assert dtype is None or isinstance(dtype, torch.dtype), "'dtype' must be either None or a torch.dtype type"
        if dtype:
            new_output_desc = self._OutputDescriptionTyped(name, shape, is_loss, dtype, None)
        else:
            new_output_desc = self._OutputDescription(name, shape, is_loss)

        if id(node) == id(self.outputs):
            self.outputs.append(new_output_desc)
        else:
            assert isinstance(attr_name, str) and len(attr_name) > 0, "Invalid 'attr_name'"
            setattr(node, attr_name, new_output_desc)

    def _wrap(self, v):
        '''Add 'v' as self's attribute to allow direct access as self.v'''
        if isinstance(v, (list)):
            return type(v)([self._wrap(v) for v in v])
        elif isinstance(v, (self._InputDescription, self._InputDescriptionTyped,
                            self._OutputDescription, self._OutputDescriptionTyped)):
            return v
        elif isinstance(v, (tuple)):
            return type(v)([self._wrap(v) for v in v])
        elif isinstance(v, (dict, int, float, bool, str)):
            return _ORTTrainerModelDescInternal(self._main_class_name, v) if isinstance(v, dict) else v
        else:
            raise ValueError(f"Unsupported type for model_desc ({v})."
                             "Only int, float, bool, str, list, tuple and dict are supported")


class _ORTTrainerModelDescInternal(_ORTTrainerModelDesc):
    r"""Internal class used by ONNX Runtime training backend for input validation

    NOTE: Users MUST NOT use this class in any way!
    """

    def __init__(self, main_class_name, model_desc):
        # Used for logging purposes
        self._main_class_name = main_class_name

        # Convert dict in object
        for k, v in dict(model_desc).items():
            setattr(self, k, self._wrap(v))


def _model_desc_inputs_validation(field, value, error):
    r'''Cerberus custom check method for 'model_desc.inputs'

    'model_desc.inputs' is a list of tuples.
    The list has variable length, but each tuple has size 2

    The first element of the tuple is a string which represents the input name
    The second element is a list of shapes. Each shape must be either an int or string.
        Empty list represents a scalar output

    Validation is done within each tuple to enforce the schema described above.

    Example:

        .. code-block:: python

            model_desc['inputs'] = [('input1', ['batch', 1024]),
                                    ('input2', [])
                                    ('input3', [512])]
    '''

    if not isinstance(value, tuple) or len(value) != 2:
        error(field, "must be a tuple with size 2")
    if not isinstance(value[0], str):
        error(field, "the first element of the tuple (aka name) must be a string")
    if not isinstance(value[1], list):
        error(field, "the second element of the tuple (aka shape) must be a list")
    else:
        for shape in value[1]:
            if not isinstance(shape, str) and not isinstance(shape, int) or isinstance(shape, bool):
                error(field, "each shape must be either a string or integer")


@static_vars(loss_counter=0)
def _model_desc_outputs_validation(field, value, error):
    r'''Cerberus custom check method for 'model_desc.outputs'

    'model_desc.outputs' is a list of tuples with variable length.
    The first element of the tuple is a string which represents the output name
    The second element is a list of shapes. Each shape must be either an int or string.
        Empty list represents a scalar output
    The third element is optional and is a flag that signals whether the output is a loss value

    Validation is done within each tuple to enforce the schema described above, but also
    throughout the list of tuples to ensure a single 'is_loss=True' occurrence.

    Example:

        .. code-block:: python

            model_desc['outputs'] = [('output1', ['batch', 1024], is_loss=True),
                                     ('output2', [], is_loss=False)
                                     ('output3', [512])]
    '''

    if not isinstance(value, tuple) or len(value) < 2 or len(value) > 3:
        error(field, "must be a tuple with size 2 or 3")
    if len(value) == 3 and not isinstance(value[2], bool):
        error(field, "the third element of the tuple (aka is_loss) must be a boolean")
    elif len(value) == 3:
        if value[2]:
            _model_desc_outputs_validation.loss_counter += 1
        if _model_desc_outputs_validation.loss_counter > 1:
            error(field, "only one is_loss can bet set to True")
    if not isinstance(value[0], str):
        error(field, "the first element of the tuple (aka name) must be a string")
    if not isinstance(value[1], list):
        error(field, "the second element of the tuple (aka shape) must be a list")
    else:
        for shape in value[1]:
            if not isinstance(shape, str) and not isinstance(shape, int) or isinstance(shape, bool):
                error(field, "each shape must be either a string or integer")


# Validation schema for model description dictionary
MODEL_DESC_SCHEMA = {
    'inputs': {
        'type': 'list',
        'required': True,
        'minlength': 1,
        'schema': {
            'check_with': _model_desc_inputs_validation
        },
    },
    'outputs': {
        'type': 'list',
        'required': True,
        'minlength': 1,
        'schema': {

            'check_with': _model_desc_outputs_validation
        },
    }
}
