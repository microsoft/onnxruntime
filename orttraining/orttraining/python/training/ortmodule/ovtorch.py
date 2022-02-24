import torch
from onnxruntime.training.ortmodule import ORTModule

class OVTorch(ORTModule):

   '''TODO: OVTorch module to add OpenVINO related APIs ''' 

   def __init__(self,module, debug_options=None):
       super().__init__(module)

   # Dummy get_backend function
   def get_backend():
       print("OpenVINO APIs")

       

