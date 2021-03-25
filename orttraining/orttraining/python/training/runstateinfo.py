# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

class RunStateInfo(object):
   def __init__(self, run_id, run_options, forward_io_binding, backward_io_binding, output_info):
      self.run_id = run_id
      self.run_options = run_options
      self.forward_io_binding = forward_io_binding
      self.backward_io_binding = backward_io_binding
      self.output_info = output_info
