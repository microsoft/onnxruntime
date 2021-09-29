import torch
from torch.nn import functional as F
import numpy as np
import os
import time
import ipdb
import myutils
_b = ipdb.set_trace

TIME_INFO = {'total_model_runtime': 0,
             'first_model_runtime': 0,
             'total_search_time': 0,
             'search_steps': 0,
             'topk_time' : 0,
             'topkn_time' : 0}

class  LMGenerator(object):
    # construct once for one task
    def __init__(self,
        max_length,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=0,
        eos_token_ids=[],
        length_penalty=1.,
        num_return_sequences=1,
        num_beams = 1,
        enable_ort = False
        ):
        self.max_length = max_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.pad_token_id = pad_token_id
        self.eos_token_ids = eos_token_ids
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.enable_ort = enable_ort

    def generate_stream(self, model, input_ids, first_token_masks=None, **kwargs):
        global common_fh
        generator_status = None
        model_status = None
        TIME_INFO.update({'total_model_runtime': 0,
                          'first_model_runtime': 0,
                          'total_search_time': 0,
                          'search_steps': 0,
                          'topk_time' : 0,
                          'topkn_time' : 0})
        myutils.total_infer_time = 0
                              
        # current position and vocab size
        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(1) # (batch_size, 1, sequence_len), the 2nd dim indicate number of sequences has been generated per sample.

        cur_len = input_ids.size(2)
        batch_size = input_ids.size(0)

        # current position / max lengths / length of generated sentences / unfinished sentences
        self.done = False
        self.unfinished_sents = input_ids.new(batch_size, 1).fill_(1)
        self.log_probs = input_ids.new(batch_size, 1).fill_(0).float()

        counter = 0
        inference_start_time = time.perf_counter()
        while not self.done:
            model_start_time = time.perf_counter()
            next_token_logits, model_status = model(input_ids, model_status, generator_status)
            model_end_time = time.perf_counter()
            TIME_INFO['total_model_runtime'] += (
                model_end_time - model_start_time
            )
            # next_token_logits = next_token_logits.view(input_ids.shape + (-1,))
            if counter == 0: # first token
                for eos_token_id in set(self.eos_token_ids):
                    next_token_logits[..., eos_token_id] = -1000

                if first_token_masks is not None and counter == 0:
                    next_token_logits = next_token_logits - 1000 * (1 - first_token_masks).unsqueeze(1)

            input_ids, generator_status = self.search_next(input_ids, next_token_logits, **kwargs)

            if self.enable_ort: 
                # vish generator state is updated 
                last_n_seq = generator_status['last_num_sequences_per_sample']
                input_seq_index = generator_status['input_seq_index']
                input_seq_index = (
                    input_seq_index +
                    (last_n_seq *
                    torch.arange(batch_size).unsqueeze(1).to('cuda:0')))
                    
                generator_status['input_seq_index'] = input_seq_index.view(-1)

            yield input_ids[:, :self.num_return_sequences], generator_status['log_probs'][:, :self.num_return_sequences].exp()

            counter += 1
        
        if counter != 0:
            myutils.counterset = True
        TIME_INFO['search_steps'] = counter
        TIME_INFO['total_search_time'] = time.perf_counter() - inference_start_time - TIME_INFO['total_model_runtime']
        myutils.common_fh.write(str(myutils.total_infer_time) + "\t")
        myutils.common_fh.write(str(counter) + "\t")
        myutils.common_fh.write(str(TIME_INFO['total_model_runtime'] * 1000) + "\t")
        myutils.common_fh.write(str(TIME_INFO['total_search_time'] * 1000) + "\t")

    def generate(self, model, input_ids, stream=False, *args, **kwargs):
        # return tuple (output_ids, probs)
        iterator = self.generate_stream(model, input_ids, *args, **kwargs)
        if stream:
            return iterator
        else:
            for x in iterator:
                pass
            return x

    def search_next(
        self,
        input_ids,
        next_token_logits,
        max_length=None,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        length_penalty=None,
        num_return_sequences=None,
        num_beams=None):

        max_length = max_length or self.max_length
        do_sample = do_sample or self.do_sample
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        top_p = top_p or self.top_p
        repetition_penalty = repetition_penalty or self.repetition_penalty
        length_penalty = length_penalty or self.length_penalty
        num_return_sequences = num_return_sequences or self.num_return_sequences
        num_beams = num_beams or self.num_beams

        # `input_ids` is 3D as following. `next_token_logits` is also 3D with shape (batch_size, input_num_sequences, vocab_size)
        batch_size, input_num_sequences, cur_len = input_ids.shape  # input is 3D.

        if self.enable_ort:
            next_token_logits = next_token_logits[:,-1]
            next_token_logits = next_token_logits.view(batch_size, input_num_sequences, -1)

        vocab_size = next_token_logits.size(-1)

        if cur_len >= max_length or self.done:
            raise ValueError('Finished')

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            _pen = next_token_logits.gather(2, input_ids)
            # num_repeats = (x.unique(return_counts=True, return_inverse=True) for x in input_ids.view(-1, input_ids.size(0)))
            # num_repeats = torch.stack([x[2].gather(0, x[1]) for x in num_repeats])
            _pen = (_pen > 0).float() * _pen / repetition_penalty + (_pen < 0).float() * _pen * repetition_penalty
            next_token_logits.scatter_(2, input_ids, _pen)

        if (next_token_logits.shape[2] > 50257):
            next_token_logits[:,:, 50257:] = next_token_logits[:,:, 50257:] - 1000

        # similar way to encourage short sentence
        if length_penalty != 1.0:
            for eos_token_id in set(self.eos_token_ids):
                _pen = next_token_logits[..., eos_token_id]
                _pen = (_pen < 0).float() * (_pen / length_penalty) + (_pen > 0).float() * (_pen * length_penalty)
                next_token_logits[..., eos_token_id] = _pen

        if self.temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        num_samples = num_return_sequences * num_beams if input_ids.shape[1] == 1 else num_beams
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        next_token_log_probs, next_token = torch.topk(next_token_log_probs, num_samples, dim=-1, largest=True, sorted=True)

        next_token_log_probs = next_token_log_probs * self.unfinished_sents.float().unsqueeze(-1)

        if num_beams > 1:
            # when one beam has finished in previous rounds, at this current round of search, only keep first one/beam alive, while others are to be removed (via heavy penalty) as they are duplicates
            next_token_log_probs[:, :, 1:].masked_fill_((1 - self.unfinished_sents).bool().unsqueeze(-1).repeat(1, 1, next_token_log_probs.size(-1)- 1), -1000)

        log_probs = self.log_probs.unsqueeze(-1) + next_token_log_probs

        # select #(self.num_return_sequences * num_beams) sequences from beams of all sequences, sorted by sequence probability
        self.log_probs, index = log_probs.view(batch_size, -1).topk(self.num_return_sequences * num_beams, dim=-1, largest=True, sorted=True)

        # select the correspondent sentences/next tokens
        input_seq_index = index // next_token.size(-1)
        next_token = next_token.view(batch_size, -1).gather(-1, index)
        input_ids = input_ids.gather(1, input_seq_index.unsqueeze(-1).repeat(1, 1, cur_len))
        self.unfinished_sents = self.unfinished_sents.gather(1, input_seq_index)

        # append next token to inputs and check if sentences finished
        tokens_to_add = next_token * self.unfinished_sents + self.pad_token_id * (1 - self.unfinished_sents)
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        for eos_token_id in self.eos_token_ids:
            self.unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
        cur_len += 1

        # stop when there is a end-of-text token in each sentence, or if we exceed the maximum length
        if cur_len >= max_length or self.unfinished_sents.max() == 0:
            self.done = True

        ## The following code were originally in Hugging'face code, but seems very artificial
        ## add eos_token_ids to unfinished sentences
        # if self.done or cur_len == max_length:
        #     input_ids[:, :, -1].masked_fill_(self.unfinished_sents.to(dtype=torch.bool), self.eos_token_ids[0])

        generator_status = {'input_seq_index': input_seq_index,
                            'log_probs': self.log_probs,
                            'unfinished_sents': self.unfinished_sents,
                            'last_num_sequences_per_sample': input_num_sequences}
        return input_ids, generator_status