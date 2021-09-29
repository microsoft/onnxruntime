import torch


class BaseModelWrapper(torch.nn.Module):
    def __init__(self, model, pad_token_id=0, float_type=None, *args, **kwargs):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.float_type = float_type or torch.float32

    @staticmethod
    def preprocess_inputs(input_ids, pad_token_id=-1, float_type=torch.float32):
        '''It automatically infer `position_ids` and `attention_mask` for `GPT2LMHeadModel`

        :param pad_token_id: -1 means not padding. Raise error if `input_ids` has more than 1 sample.
        '''
        # if input_ids.size(0) > 1:
        #     assert pad_token_id >= 0, '"pad_token_id" should be provided if using batch mode.'
        mask = (input_ids != pad_token_id)

        # the following was meant to remove unnecessary paddings for left/right padding
        # but it is now invalid as it will cause sequence length inconsistency between input_ids and past states

        # nonzero = mask.any(0).nonzero()
        # maxlen = nonzero.max() + 1 if nonzero.numel() > 0 else 1
        # input_ids = input_ids[:, :maxlen].contiguous()
        # mask = mask[:, :maxlen].contiguous()

        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        attention_mask = mask.to(float_type)
        # if position_ids.max().item() >= 1023:
        #     import ipdb; ipdb.set_trace()
        return {'input_ids': input_ids, 'position_ids': position_ids, 'attention_mask': attention_mask}

    def get_model_inputs(self, input_ids, past, generator_state):
        model_inputs = self.preprocess_inputs(input_ids, self.pad_token_id, float_type=self.float_type)
        if past is not None:
            for k in model_inputs:
                if k in ('input_ids', 'position_ids'):
                    # input last token is needed. But mask should contains all historical input masks
                    model_inputs[k] = model_inputs[k][:, -1:]
            input_seq_index = generator_state['input_seq_index']
            past = tuple(layer_past.index_select(1, input_seq_index) for layer_past in past)
            model_inputs['past'] = past
            '''
            if generator_state:
                batch_size, num_sequences_per_sample = input_ids.shape
                input_seq_index = generator_state['input_seq_index']
                input_seq_index = input_seq_index + torch.arange(batch_size).unsqueeze(1).to(input_ids.device) * generator_state['last_num_sequences_per_sample']
                input_seq_index = input_seq_index.view(-1)
                model_status = tuple(layer_past.index_select(1, input_seq_index) for layer_past in past)
                model_inputs['past'] = model_status
            '''

        return model_inputs

    def forward(self, input_ids, model_state=None, generator_state={}):
        NotImplemented
