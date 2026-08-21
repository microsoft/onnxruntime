import time
import torch
from torch._C import device
from .base import BaseModelWrapper
from transformers import GPT2LMHeadModel
import myutils

"""
Implement pt version of the model
"""

class PtModelWrapper(BaseModelWrapper):
    def __init__(
        self,
        model_path:str,
        pad_token_id=0,
        device='cuda:0',
        num_layers = 1,
        head_size = 64,
        num_heads = 12):
        #TODO num_layers are 1 by default, other models need more than 1
        # by passing this as command line argument
        # similarly with head_size and number of heads
        super().__init__(pad_token_id=pad_token_id)
        self._model = GPT2LMHeadModel.from_pretrained(model_path)
        if 'cuda' in device:
            device = torch.device('cuda')
            self._model.to(device)
        self._model.eval()
        self.pad_token_id = pad_token_id
        self._float_type = None
        self._num_layers = num_layers
        self._head_size = head_size
        self._num_heads = num_heads

    def _get_outputs(self, input_ids, position_ids, attention_mask, *past):
        try:
            start_time = time.perf_counter()
            outputs = self._model(
                        input_ids = input_ids,
                        position_ids = position_ids,
                        attention_mask = attention_mask,
                        past = past)
            end_time = time.perf_counter()

            infer_time = (end_time - start_time) * 1000
            myutils.total_infer_time += infer_time

            return outputs
        except Exception as e:
            # TODO Create a class to represent this as a custom Exception
            raise Exception(f"Exception inside pytorch run {str(e)}")

    def forward(self, input_ids, model_state, generator_state):
        """
        The inputs are made in BaseModelWrapper and can be used as is
        """
        try:
            _, _, cur_len = input_ids.shape
            input_ids = input_ids.view(-1, cur_len)

            model_inputs = self.get_model_inputs(
                                input_ids,
                                model_state,
                                generator_state)
        
            past_list = []
            if "past" in model_inputs.keys():
                past_list = [p.to(dtype=self._float_type) for p in model_inputs["past"]]
            else:
                s = 0
                b, _  = model_inputs['input_ids'].size()
                shape = [2, b, self._num_heads, s, self._head_size]
                zeros = torch.zeros(shape).type_as(model_inputs["attention_mask"])
                #TODO here with 1 layer or n layers when the input tensor in zero, only one 
                # zero tensor is supplied and it seems to be working. May be inside pytorch this 
                # optimization is already present
                past_list.extend([zeros for i in range(self._num_layers)])

            return self._get_outputs(
                            model_inputs["input_ids"],
                            model_inputs["position_ids"],
                            model_inputs["attention_mask"],
                            *past_list)
        except Exception as e:
            raise e

    def run(self, **kargs):
        """
        write this function to complete running the model by itself
        Probably needs a new Wrapper of this. Currently not supported
        """
        raise Exception(f"Not implemented yet")