#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import numpy as np

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


# !\brief This class hosts
#   various utility functions.
class DataFrameTool():
  # @param sess is expected to be a loaded onnx model
  #        which can be executed repeatedly
  def __init__(self, sess):
    self.sess_ = sess

  # \!brief This function accepts Pandas DataFrame as the first argument and onnxruntime
  # session with a loaded model. The function interrogates the model for the inputs
  # and matches the model input names to the DataFrame instance column names.
  # It then performs the type matching according to its internal mapping. (Should we accept
  # an extra, possibly optional argument for mapping?
  #
  # @param df - an instance of DataFrame
  #   the function only considers the first row of each column and feeds the data to the
  #   appropriate model inputs.
  #   For example: pd.DataFrame([[0], [4],[20]],index=[0], columns=['A', 'B', 'C'])
  #   Since we require a single row to be present, we currently refer to it by index 0
  #   The input types are expected to match the expected input types. 'O' are taken as strings.
  #
  # @return feeds that has feeds populated for session run
  #
  # This dict maps input data type to a np.dtype
  # The inputs for onnx models are currently always tensors
  # but we can make maps and sequences as well
  def feed_nputs(self, df):
    if df.empty:
      raise RuntimeError('input DataFrame is empty')

    feeds = {}
    meta = self.sess_.get_modelmeta()
    # Combine input_meta and overridable initializers_meta together
    input_descriptors = self.sess_.get_inputs() + self.sess_.get_overridable_initializers()
    for input_meta in input_descriptors:
      shape = [dim if dim else 1 for dim in input_meta.shape]
      # We fully expect all the types are in the above dictionary
      assert input_meta.type in types_dict
      print("Processing: " + input_meta.name)
      if input_meta.name in df.columns:
        expected_type = types_dict[input_meta.type]
        if input_meta.type == 'tensor(string)':
          # With strings we always want to put this into a flat array, cast to np.object and then reshape
          feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
        elif expected_type == df[input_meta.name].dtype:  # Everything else map according to the dictionary
            feeds[input_meta.name] = np.array([df[input_meta.name][0]]).astype(expected_type).reshape(shape)
        else:
          raise TypeError("Input {} expected to be of type: {} got {} ".format(
                          input_meta.name, expected_type, df[input_meta.name].dtype))
      else:
        raise RuntimeError("This model requires input {} of type {} but it is not found in the DataFrame".format(
                             input_meta.name, types_dict[input_meta.type]))

    return feeds

  # Returns a list of onnx types that this class can process
  def get_onnx_types_list():
    return list(types_dict.keys())
