# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class RunStateInfo(object):
   def __init__(self, run_id, run_options, io_binding, output_info):
      self.run_id = run_id
      self.run_options = run_options
      self.io_binding = io_binding
      self.output_info = output_info
