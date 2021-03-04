# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import collections.abc
import os

from onnxruntime.capi import _pybind_state as C
# TODO: This create a circular dependency
# from onnxruntime.capi.onnxruntime_inference_collection import IOBinding
from onnxruntime.capi._pybind_state import TrainingAgent as C_TrainingAgent

# TODO: Duplicate from onnxruntime/python/onnxruntime_inference_collection.py to break circular dependency


class IOBinding:
    '''
    This class provides API to bind input/output to a specified device, e.g. GPU.
    '''

    def __init__(self, agent):
        self._iobinding = C.SessionIOBinding(agent.get_session())
        self._numpy_obj_references = []

    def bind_cpu_input(self, name, arr_on_cpu):
        '''
        bind an input to array on CPU
        :param name: input name
        :param arr_on_cpu: input values as a python array on CPU
        '''
        # Hold a reference to the numpy object as the bound OrtValue is backed
        # directly by the data buffer of the numpy object and so the numpy object
        # must be around until this IOBinding instance is around
        self._numpy_obj_references.append(arr_on_cpu)
        self._iobinding.bind_input(name, arr_on_cpu)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        '''
        :param name: input name
        :param device_type: e.g. cpu, cuda
        :param device_id: device id, e.g. 0
        :param element_type: input element type
        :param shape: input shape
        :param buffer_ptr: memory pointer to input data
        '''
        self._iobinding.bind_input(name,
                                   C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                               device_id),
                                   element_type, shape, buffer_ptr)

    def bind_ortvalue_input(self, name, ortvalue):
        '''
        :param name: input name
        :param ortvalue: OrtValue instance to bind
        '''
        self._iobinding.bind_ortvalue_input(name, ortvalue._ortvalue)

    def bind_output(self, name, device_type='cpu', device_id=0, element_type=None, shape=None, buffer_ptr=None):
        '''
        :param name: output name
        :param device_type: e.g. cpu, cuda, cpu by default
        :param device_id: device id, e.g. 0
        :param element_type: output element type
        :param shape: output shape
        :param buffer_ptr: memory pointer to output data
        '''

        # Follow the `if` path when the user has not provided any pre-allocated buffer but still
        # would like to bind an output to a specific device (e.g. cuda).
        # Pre-allocating an output buffer may not be an option for the user as :
        # (1) They may not want to use a custom allocator specific to the device they want to bind the output to,
        # in which case ORT will allocate the memory for the user
        # (2) The output has a dynamic shape and hence the size of the buffer may not be fixed across runs
        if buffer_ptr is None:
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id))
        else:
            if element_type is None or shape is None:
                raise ValueError(
                    "`element_type` and `shape` are to be provided if pre-allocated memory is provided")
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id),
                                        element_type, shape, buffer_ptr)

    def bind_ortvalue_output(self, name, ortvalue):
        '''
        :param name: output name
        :param ortvalue: OrtValue instance to bind
        '''
        self._iobinding.bind_ortvalue_output(name, ortvalue._ortvalue)

    def get_outputs(self):
        '''
        Returns the output OrtValues from the Run() that preceded the call.
        The data buffer of the obtained OrtValues may not reside on CPU memory
        '''
        returned_ortvalues = []

        for ortvalue in self._iobinding.get_outputs():
            returned_ortvalues.append(OrtValue(ortvalue))

        return returned_ortvalues

    def copy_outputs_to_cpu(self):
        '''Copy output contents to CPU (if on another device). No-op if already on the CPU.'''
        return self._iobinding.copy_outputs_to_cpu()

    def clear_binding_inputs(self):
        self._iobinding.clear_binding_inputs()

    def clear_binding_outputs(self):
        self._iobinding.clear_binding_outputs()

# TODO: Duplicate from onnxruntime/python/onnxruntime_inference_collection.py to break circular dependency


def check_and_normalize_provider_args(providers, provider_options, available_provider_names):
    """
    Validates the 'providers' and 'provider_options' arguments and returns a
        normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if name not in available_provider_names:
            raise ValueError("Specified provider '{}' is unavailable. Available providers: '{}'".format(
                name, ", ".join(available_provider_names)))

        if name in provider_name_to_options:
            warnings.warn(
                "Duplicate provider '{}' encountered, ignoring.".format(name))
            return

        normalized_options = {str(key): str(value)
                              for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError(
                "'providers' and 'provider_options' should be the same length if both are given.")

        if not all([isinstance(provider, str) for provider in providers]):
            raise ValueError(
                "Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all([isinstance(options_for_provider, dict) for options_for_provider in provider_options]):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, dict())
            elif isinstance(provider, tuple) and len(provider) == 2 and \
                    isinstance(provider[0], str) and isinstance(provider[1], dict):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError(
                    "'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


class TrainingAgent(object):
    """
    This is the main class used to run a ORTModule model.
    """

    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None):
        """
        :param path_or_bytes: filename or serialized ONNX or ORT format model in a byte string
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """

        if isinstance(path_or_bytes, str):
            print('from string')
            self._model_path = path_or_bytes
            self._model_bytes = None
        elif isinstance(path_or_bytes, bytes):
            print('from bytes')
            self._model_path = None
            # TODO: This is bad as we're holding the memory indefinitely
            self._model_bytes = path_or_bytes
        else:
            raise TypeError(
                "Unable to load from type '{0}'".format(type(path_or_bytes)))

        self._sess_options = sess_options
        self._sess_options_initial = sess_options
        self._enable_fallback = True
        self._read_config_from_model = os.environ.get(
            'ORT_LOAD_CONFIG_FROM_MODEL') == '1'

        try:
            self.create_training_agent(providers, provider_options)
        except RuntimeError:
            if self._enable_fallback:
                print("EP Error using {}".format(providers))
                print("Falling back to {} and retrying.".format(
                    self._fallback_providers))
                self.create_training_agent(self._fallback_providers, None)
                # Fallback only once.
                self._enable_fallback = False
            else:
                raise

    def create_training_agent(self, providers, provider_options):
        available_providers = C.get_available_providers()

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(providers,
                                                                        provider_options,
                                                                        available_providers)

        # Tensorrt can fall back to CUDA. All others fall back to CPU.
        if 'TensorrtExecutionProvider' in available_providers:
            self._fallback_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self._fallback_providers = ['CPUExecutionProvider']

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        agent = C_TrainingAgent(session_options, self._model_bytes)

        # import pdb; pdb.set_trace()
        # initialize the C++ InferenceSession
        agent.initialize_session(providers, provider_options)

        self._agent = agent
        self._sess_options = agent.get_session_options()
        self._providers = agent.get_providers()
        self._provider_options = agent.get_provider_options()

    def _reset_training_agent(self, providers, provider_options):
        "release underlying session object."
        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._sess_options = None
        self._providers = None
        self._provider_options = None

        # create a new C.InferenceSession
        self._agent = None
        self._sess_options = self._sess_options_initial
        self._create_training_agent(providers, provider_options)

    def io_binding(self):
        "Return an onnxruntime.IOBinding object`."
        return IOBinding(self._agent.get_session())
