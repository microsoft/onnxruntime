import torch

class BaseModelWrapper(torch.nn.Module):
    def __init__(self, pad_token_id=0, float_type=None, *args, **kwargs):
        super().__init__()
        self._pad_token_id = pad_token_id
        self._float_type = float_type or torch.float32

    def _preprocess_inputs(self, input_ids):
        '''It automatically infer `position_ids` and `attention_mask` for `GPT2LMHeadModel`
        '''
        try:
            mask = (input_ids != self._pad_token_id)
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(position_ids < 0, 0)
            attention_mask = mask.to(self._float_type)
            
            return {'input_ids': input_ids, 'position_ids': position_ids, 'attention_mask': attention_mask}
        except Exception as e:
            raise Exception(f"Exception in preprocessing inputs {str(e)}")

    def get_model_inputs(self, input_ids, past, generator_state):
        try:
            model_inputs = self._preprocess_inputs(input_ids)
            if past is not None:
                for k in model_inputs:
                    if k in ('input_ids', 'position_ids'):
                        model_inputs[k] = model_inputs[k][:, -1:]
                input_seq_index = generator_state['input_seq_index']
                past = tuple(layer_past.index_select(1, input_seq_index) for layer_past in past)
                model_inputs['past'] = past

            return model_inputs
        except Exception as e:
            raise e

    def forward(self, input_ids, model_state=None, generator_state={}):
        NotImplemented
