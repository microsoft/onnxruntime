import torch
from torch.nn import functional as F
import time
import ipdb
from tokenizer import Tokenizer
import myutils
_b = ipdb.set_trace

TIME_INFO = {'total_model_runtime': 0,
             'first_model_runtime': 0,
             'total_search_time': 0,
             'search_steps': 0,
             'topk_time' : 0,
             'topkn_time' : 0}

class  Generator(object):
    def __init__(
        self, max_length, num_return_sequences=1, num_beams = 0, pad_token_id=0, eos_token_ids=[],
        length_penalty=1.0, tokenizer: Tokenizer = None, temperature = 1.0,
        repetition_penalty = 1.0, num_layers = 1, device = 'cuda'):
        self._max_length = max_length
        self._num_return_sequences = num_return_sequences
        self._num_beams = num_beams
        self._pad_token_id = pad_token_id
        self._eos_token_ids = eos_token_ids
        self._length_penalty = length_penalty
        self._tokenizer = tokenizer
        self._beam_search = num_beams > 0
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._device = device

    def _update_generator_status(self, generator_status, batch_size):
        last_n_seq = generator_status['last_num_sequences_per_sample']
        input_seq_index = generator_status['input_seq_index']
        input_seq_index = (
            input_seq_index +
            (last_n_seq *
                torch.arange(batch_size).unsqueeze(1).to(self._device)))
                    
        generator_status['input_seq_index'] = input_seq_index.view(-1)

    def _update_logits_with_first_token_masks(self, next_token_logits, first_token_masks):
        for eos_token_id in set(self._eos_token_ids):
            next_token_logits[..., eos_token_id] = -1000

        if first_token_masks is not None:
            next_token_logits = next_token_logits - 1000 * (1 - first_token_masks).unsqueeze(1)
        
        return next_token_logits
        
    def _prediction_generator(self, model, input_ids, first_token_masks=None, **kwargs):
        generator_status = None
        model_status = None

        try:
            if input_ids.ndim == 2:
                input_ids = input_ids.unsqueeze(1)

            batch_size = input_ids.size(0)
            self.done = False
            self.unfinished_sents = input_ids.new(batch_size, 1).fill_(1)
            self.log_probs = input_ids.new(batch_size, 1).fill_(0).float()

            counter = 0
            inference_start_time = time.perf_counter()

            while not self.done:
                model_start_time = time.perf_counter()
                next_token_logits, model_status = model(input_ids, model_status, generator_status)
                model_end_time = time.perf_counter()

                TIME_INFO['total_model_runtime'] += (model_end_time - model_start_time)

                if counter == 0: # first token
                        next_token_logits = self._update_logits_with_first_token_masks(next_token_logits, first_token_masks)

                input_ids, generator_status = self._search_next(input_ids, next_token_logits, **kwargs)
                self._update_generator_status(generator_status, batch_size)
                yield input_ids[:, :self._num_return_sequences], generator_status['log_probs'][:, :self._num_return_sequences].exp()

                counter += 1

            TIME_INFO['search_steps'] = counter
            TIME_INFO['total_search_time'] = time.perf_counter() - inference_start_time - TIME_INFO['total_model_runtime']
        except Exception as e:
            raise e

    def generate(self, model, input_ids, *args, **kwargs):
        """
        Entry point for suggestions generator.
        
        If self._beam_search is set to True, prepares the inputs for beam search and processes outputs after each iteration
        as required by next step. 

        If self._beam_search is set to False, this simply runs the model and returns the outputs 
        """
        try :
            myutils.total_infer_time = 0
            TIME_INFO.update({
                        'total_model_runtime': 0,
                        'first_model_runtime': 0,
                        'total_search_time': 0,
                        'search_steps': 0,
                        'topk_time' : 0,
                        'topkn_time' : 0})
            
            if self._beam_search is False:
                #TODO this place needs to be handle the case of beam search OP and single entry into the model
                return model(input_ids, None, None, self._beam_search)
            else:
                iterator = self._prediction_generator(model, input_ids, *args, **kwargs)
                for x in iterator:
                    pass
            
                myutils.common_fh.write(str(myutils.total_infer_time) + "\t")
                myutils.common_fh.write(str(TIME_INFO['search_steps']) + "\t")
                myutils.common_fh.write(str(TIME_INFO['total_model_runtime'] * 1000) + "\t")
                myutils.common_fh.write(str(TIME_INFO['total_search_time'] * 1000) + "\t")

                return x
        except Exception as e:
            raise e

    def _apply_length_penalty(self, next_token_logits):
        '''
        Here eos_token_id is updated in order to have shorter sequences
        this benefits in e2e latency
        '''
        if self._length_penalty != 1.0:
            for eos_token_id in set(self._eos_token_ids):
                pen = next_token_logits[..., eos_token_id]
                pen = (pen < 0).float() * (pen / self._length_penalty) + (pen > 0).float() * (pen * self._length_penalty)
                next_token_logits[..., eos_token_id] = pen

    def _apply_repetition_penalty(self, input_ids, next_token_logits):
        if self.repetition_penalty != 1.0:
            pen = next_token_logits.gather(2, input_ids)
            pen = (pen > 0).float() * pen / self._repetition_penalty + (pen < 0).float() * pen * self._repetition_penalty
            next_token_logits.scatter_(2, input_ids, pen)

    def _first_top_k_sort(self, input_ids, next_token_logits):
        ''' First top K: extract top (_num_return_sequences * _num_beams)
         if _num_return_sequences = 8 and beam_size = 2:
         first iteration we want 16 outputs because batch_size would be 1,
         for next iterations, batch_size would be 16 and 2(_num_beams) from each batch would be extracted
        '''
        num_samples = self._num_return_sequences * self._num_beams if input_ids.shape[1] == 1 else self._num_beams
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

        next_token_log_probs, next_tokens = torch.topk(next_token_log_probs, num_samples, dim=-1, largest=True, sorted=True)
        next_token_log_probs = next_token_log_probs * self.unfinished_sents.float().unsqueeze(-1)

        return next_token_log_probs, next_tokens

    def _second_top_k_sort(self, log_probs, batch_size):
        ''' Second top K:
         First iteration:
               log_probs = (1, 1, 16) keep all of them
               self.log_probs = (1,16)
         Next iterations:
               log_probs = (1, 16, 2)
               extract 16 out of these
         select #(self._num_return_sequences * _num_beams) sequences from beams of all sequences, sorted by sequence probability
        '''
        log_probs, index = log_probs.view(batch_size, -1).topk(self._num_return_sequences * self._num_beams, dim=-1, largest=True, sorted=True)

        return  log_probs, index

    def _make_inputs_for_next_iteration(self, index, next_tokens, batch_size, input_ids, cur_len):
        # select the correspondent sentences/next tokens
        self.input_seq_index = index // next_tokens.size(-1)
        next_tokens = next_tokens.view(batch_size, -1).gather(-1, index)
        input_ids = input_ids.gather(1, self.input_seq_index.unsqueeze(-1).repeat(1, 1, cur_len))
        self.unfinished_sents = self.unfinished_sents.gather(1, self.input_seq_index)

        # append next token to inputs and check if sentences finished
        tokens_to_add = next_tokens * self.unfinished_sents + self._pad_token_id * (1 - self.unfinished_sents)
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        for eos_token_id in self._eos_token_ids:
            self.unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
        
        return input_ids

    def _mask_extra_tokens(self, next_token_logits):
        # the self._tokenizer has only extra tokens sometimes, subtract 1000 from all of them  to make it insignificant
        original_tokens_count = self._tokenizer.get_gpt2_token_count()
        if (next_token_logits.shape[2] > original_tokens_count):
            next_token_logits[:,:, original_tokens_count:] = next_token_logits[:,:, original_tokens_count:] - 1000

    def _search_next(self, input_ids, next_token_logits):
        # `input_ids` is 3D as following. `next_token_logits` is also 3D with shape (batch_size, input_num_sequences, vocab_size)
        batch_size, input_num_sequences, cur_len = input_ids.shape  # input is 3D.

        next_token_logits = next_token_logits[:,-1]
        next_token_logits = next_token_logits.view(batch_size, input_num_sequences, -1)

        if cur_len >= self._max_length or self.done:
            raise ValueError('Finished but still trying to beam search')

        self._mask_extra_tokens(next_token_logits)

        self._apply_length_penalty(next_token_logits)

        if self._temperature != 1.0:
            next_token_logits = next_token_logits / self._temperature

        next_token_log_probs, next_tokens = self._first_top_k_sort(input_ids, next_token_logits)

        if self._num_beams > 1:
            # when one beam has finished in previous rounds, at this current round of search, only keep first one/beam alive, while others are to be removed (via heavy penalty) as they are duplicates
            next_token_log_probs[:, :, 1:].masked_fill_((1 - self.unfinished_sents).bool().unsqueeze(-1).repeat(1, 1, next_token_log_probs.size(-1)- 1), -1000)
        
        log_probs = self.log_probs.unsqueeze(-1) + next_token_log_probs

        self.log_probs, index = self._second_top_k_sort(log_probs, batch_size)
        input_ids = self._make_inputs_for_next_iteration(index, next_tokens, batch_size, input_ids, cur_len)
        cur_len += 1

        # stop when there is a end-of-text token in each sentence, or if we exceed the maximum length
        if cur_len >= self._max_length or self.unfinished_sents.max() == 0:
            self.done = True

        generator_status = {'input_seq_index': self.input_seq_index, 'log_probs': self.log_probs,
                            'unfinished_sents': self.unfinished_sents, 'last_num_sequences_per_sample': input_num_sequences}

        return input_ids, generator_status