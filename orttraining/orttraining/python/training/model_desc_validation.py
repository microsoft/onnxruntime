import cerberus

from ._utils import static_vars


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

        # Convert dict in object
        for k, v in self._validated.items():
            setattr(self, k, self._wrap(v))

        # Keep this in the last line
        # After this point, this class becomes immutable
        self._initialized = True

    def __repr__(self):
        return '{%s}' % str(', '.join("'%s': %s" % (k, repr(v))
                                      for (k, v) in self.__dict__.items()
                                      if k not in ['_main_class_name', '_original', '_validated', '_initialized']))

    def __setattr__(self, k, v):
        if hasattr(self, '_initialized'):
            raise Exception(f"{self._main_class_name} is an immutable class")
        return super().__setattr__(k, v)

    def _wrap(self, v):
        if isinstance(v, (tuple, list, set, frozenset)):
            return type(v)([self._wrap(v) for v in v])
        else:
            return _ORTTrainerModelDescInternal(self._main_class_name, v) if isinstance(v, dict) else v


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

        # Keep this in the last line
        # After this point, this class becomes immutable
        self._initialized = True


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
