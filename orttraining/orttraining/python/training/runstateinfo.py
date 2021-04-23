# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class RunStateInfo(object):
   def __init__(self, state, output_info):
      """
      :param state: State of partial run that contains intermediate tensors needed to resume the run later.
      :param output_info: Output info.
      """
      self.state = state
      self.output_info = output_info
