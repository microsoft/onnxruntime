# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class RunStateInfo(object):
   def __init__(self, run_options, state, output_info):
      self.run_options = run_options
      self.state = state
      self.output_info = output_info
