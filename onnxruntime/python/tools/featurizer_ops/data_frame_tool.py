#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
# This script is an experimental feature. Use at your own risk.
# Contributions are welcome.

from datetime import datetime
import numpy as np
import pandas as pd
import onnxruntime as onnxrt

ort_float_set = set([np.float32, np.float64])

pd_float_set = set(['float64'])

ort_int_set = set([np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])

pd_int_set = set(['int64'])

types_dict = {
    'tensor(float16)': np.float16,
    'tensor(float)': np.float32,
    'tensor(double)': np.float64,
    'tensor(int8)': np.int8,
    'tensor(uint8)': np.uint8,
    'tensor(int16)': np.int16,
    'tensor(uint16)': np.uint16,
    'tensor(int32)': np.int32,
    'tensor(uint32)': np.uint32,
    'tensor(int64)': np.int64,
    'tensor(uint64)': np.uint64,
    'tensor(bool)': np.bool,
    'tensor(string)': np.object
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

    def _reshape_input(self, input_array, expected_shape):
        """
        :param - input_array numpy array. This one is obtained from DataFrame and expected to have
        :      a rank if 1.
        :expected_shape - shape fetched from the model which may include dynamic elements.
        :  expected_shape may at most have one -1, None or zero which will be computed from
        :  the size of the input_array. We replace None and zeros to -1 and let np.ndarray.reshape deal with it.
        """
        # expected_shape rank is one, we will let onnxruntime to deal with it
        if len(expected_shape) == 1:
            return input_array

        inferred_shape = [dim if dim else -1 for dim in expected_shape]
        return input_array.reshape(inferred_shape)

    def _validate_type(self, input_meta, col_type):
        """
        : input_meta - meta info obtained from the model for the given input
        : col_type - dtype of the column
        : throws if conditions are not met

         float16 and bool will always require exact match
         We attempt to convert any type to a string if it is required.
         With strings we always want to put this into a flat array, cast to np.object and then reshape as object
         Any other type to qualify for casting must match either integer or floating point types
         Python datetime which is denoted in Pandas as datetime64[ns] is cast to int64
        """
        expected_type = types_dict[input_meta.type]
        if input_meta.type == 'tensor(string)':
           return
        elif expected_type == col_type:
           return
        elif expected_type == np.int64 and str(col_type) == 'datetime64[ns]':
           return
        elif expected_type == np.uint32 and str(col_type) == 'category':
           return
        elif expected_type in ort_float_set and str(col_type) in pd_float_set:
           return
        elif expected_type in ort_int_set and str(col_type) in pd_int_set:
           return

        raise TypeError("Input {} requires type {} unable to cast column type {} ".format(
            input_meta.name, expected_type, col_type))

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
            # We fully expect all the types are in the above dictionary
            assert input_meta.type in types_dict, "Update types_dict for the new type"
            if input_meta.name in df.columns:
                self._validate_type(input_meta, df[input_meta.name].dtype)
                if (df[input_meta.name].dtype) == 'datetime64[ns]':
                    input_array = np.array([dt.timestamp() for dt in df[input_meta.name]]).astype(np.int64)
                elif (str(df[input_meta.name].dtype)) == 'category':
                    # ONNX models trained in ML.NET input from "categorical columns" is 1 based indices,
                    # whereas Categorical columns are 0 based and need to be retrieved from .array.codes
                    input_array = np.array([key + 1 for key in df[input_meta.name].array.codes]).astype(np.uint32)
                else:
                    # With strings we must cast first to np.object then then reshape
                    # so we do it for everything
                    input_array = np.array(df[input_meta.name]).astype(types_dict[input_meta.type])
                feeds[input_meta.name] = self._reshape_input(input_array, input_meta.shape)

            elif require:
                raise RuntimeError(
                    "This model requires input {} of type {} but it is not found in the DataFrame".format(
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

    def execute(self, df, output_names=None, output_types=None, run_options=None):
        "Return a list of output values restricted to output names if not empty"
        """
        Compute the predictions.

        :param df: See :class:`pandas.DataFrame`.
        :param output_names: name of the outputs that we are interested in
        :param output_types: { output_name : dtype } cast output to the colum type
        :param run_options: See :class:`onnxruntime.RunOptions`.
        ::
        sess.run([output_name], {input_name: x})
        """
        input_feed = self._get_input_feeds(df, self._sess);
        if not output_names:
            output_names = [output.name for output in self._sess._outputs_meta]

        results = self._sess.run(output_names, input_feed, run_options)

        df = None
        for i, r in enumerate(results):
            # ML.NET specific columns that needs to be removed.
            if output_names[i].startswith('mlnet.') and \
               output_names[i].endswith('.unusedOutput') and \
               r.shape == (1,1):
                continue

            r = pd.DataFrame(r)
            col_names = []
            for suffix in range(0, len(r.columns)):
                if output_types and output_names[i] in output_types:
                    dtype = output_types[output_names[i]]
                    if dtype == np.dtype('datetime64'):
                        r[suffix]= r[suffix].astype(np.int64)
                        r[suffix] = [datetime.utcfromtimestamp(ts) for ts in r[suffix]]
                    else:
                        r[suffix] = r[suffix].astype(dtype)

                col_name = output_names[i] if len(r.columns) == 1 else \
                           output_names[i] + '.' + str(suffix)
                col_names.append(col_name)

            r.columns = col_names
            if df is None:
                df = r
            else:
                df = df.join(r)

        return df
