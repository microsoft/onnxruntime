#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import numpy as np
import onnxruntime as onnxrt

ort_float_set = set([np.float32, np.float64])

pd_float_set = set(['float64'])

ort_int_set = set([np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])

pd_int_set = set(['int64'])

types_dict = {
    'tensor(float16)': np.float16,
    'tensor(float)'  : np.float32,
    'tensor(double)' : np.float64,

    'tensor(int8)'   : np.int8,
    'tensor(uint8)'  : np.uint8,
    'tensor(int16)'  : np.int16,
    'tensor(uint16)' : np.uint16,
    'tensor(int32)'  : np.int32,
    'tensor(uint32)' : np.uint32,
    'tensor(int64)'  : np.int64,
    'tensor(uint64)' : np.uint64,

    'tensor(bool)'   : np.bool,
    'tensor(string)' : np.object
}

class DataFrameTool():
    """
    This is a utility class used to run a model with pandas.DataFrame input
    """
    def __init__(self, model_path, sess_options=None):
        """
        :param model_path: path to the model to be loaded
        :param sess_options: see onnxruntime.SessionsOptions
        """
        self._model_path = model_path
        self._sess_options = sess_options
        self._sess = onnxrt.InferenceSession(self._model_path, self._sess_options)

    def _process_input_list(self, df, input_metas, require):
        """
        Return a dictionary of input_name : a typed and shaped np.array of values for a given input_meta
        The function does the heavy lifting for _get_input_feeds()

        :param df: See :class:`pandas.DataFrame`. 
        :param input_metas: a list of name/type pairs
        :require is a boolean. If True this helper throws on a missing input.
        
        """
        feeds = {}
        # Process mandadory inputs. Raise an error if anything is not present
        for input_meta in input_metas:
            shape = [dim if dim else 1 for dim in input_meta.shape]
            # We fully expect all the types are in the above dictionary
            assert input_meta.type in types_dict, "Update types_dict for the new type"
            if input_meta.name in df.columns:
                expected_type = types_dict[input_meta.type]
                # float16 and bool will always require exact match
                # We attempt to convert any type to a string if it is required.
                # With strings we always want to put this into a flat array, cast to np.object and then reshape as object
                if input_meta.type == 'tensor(string)':
                    #print('Col: {} processed as string type: {} '.format(input_meta.name, df[input_meta.name].dtype))
                    feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
                elif expected_type == df[input_meta.name].dtype: # If there is an exact match we take as is
                    #print('Col: {} processed exact match type: {} '.format(input_meta.name, df[input_meta.name].dtype))
                    feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
                elif expected_type in ort_float_set and str(df[input_meta.name].dtype) in pd_float_set:
                    #print('Col: {} processed as floating type: {} '.format(input_meta.name, df[input_meta.name].dtype))
                    feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
                elif expected_type in ort_int_set and str(df[input_meta.name].dtype) in pd_int_set:
                    #print('Col: {} processed as integer type: {} '.format(input_meta.name, df[input_meta.name].dtype))
                    feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
                else:
                    raise TypeError("Input {} expected to be of type: {} got {} ".format(
                                input_meta.name, expected_type, df[input_meta.name].dtype))
            elif require:
                raise RuntimeError("This model requires input {} of type {} but it is not found in the DataFrame".format(
                               input_meta.name, types_dict[input_meta.type]))
        return feeds


    def _get_input_feeds(self, df, sess):
        """
        Return a dictionary of input_name : a typed and shaped np.array of values
        This function accepts Pandas DataFrame as the first argument and onnxruntime
        session with a loaded model. The function interrogates the model for the inputs
        and matches the model input names to the DataFrame instance column names.
        It requires exact matches for bool and float16 types. It attempts to convert to
        string any input type if string is required.
        It attempts to convert floating types to each other and does the same for all of the
        integer types without requiring an exact match.

        :param df: See :class:`pandas.DataFrame`. The function only considers the first row (0) of each column
            and feeds the data to the appropriate model inputs.

        :param sess: See :class:`onnxruntime.InferenceSession`.
        
        ::
        For example: pd.DataFrame([[0], [4],[20]],index=[0], columns=['A', 'B', 'C'])

        """
        if df.empty:
            raise RuntimeError('input DataFrame is empty')

        # Process mandadory inputs. Raise an error if anything is not present
        feeds = self._process_input_list(df, sess.get_inputs(), True)
        # Process optional overridable initializers. If present the initialzier value
        # is overriden by the input. If not, the initialzier value embedded in the model takes effect.
        initializers = self._process_input_list(df, sess.get_overridable_initializers(), False)

        feeds.update(initializers)

        return feeds

    def execute(self, df, output_names, run_options=None):
        "Return a list of output values restricted to output names if not empty"
        """
        Compute the predictions.

        :param df: See :class:`pandas.DataFrame`.
        :param output_names: name of the outputs that we are interested in
        :param run_options: See :class:`onnxruntime.RunOptions`.

        ::

        sess.run([output_name], {input_name: x})
        """
        input_feed = self._get_input_feeds(df, self._sess);
        return self._sess.run(output_names, input_feed, run_options)

