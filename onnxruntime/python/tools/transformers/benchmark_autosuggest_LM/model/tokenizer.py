from transformers import GPT2Tokenizer
import bisect
import numpy as np
import myutils
import torch

class Tokenizer:
    """ This class is to create GPT2Tokenizer required to encode and decode text and output ids of a model respectively.
    It only needs the path of tokenizer to initialize
    """
    def __init__(self, tokenizer_path:str):
        self._tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        #print(self._tokenizer.eos_token_id)
        self._initialize_prefix_vocab()
        self.eos_token_id = self._tokenizer.eos_token_id

    def _initialize_prefix_vocab(self):
        self._vocab = [self._tokenizer.decoder[i] if i < len(self._tokenizer.decoder) else self._tokenizer.added_tokens_decoder[i] for i in range(len(self._tokenizer))]
        output = [w.startswith('Ġ') for w in self._vocab]
        self._space_vec = np.asarray(output, dtype='float32')
        self._zero_vec = np.ones((len(self._tokenizer),), dtype='float32')

        sorted_vocab = sorted(enumerate(self._vocab), key=lambda x: x[1])
        keys, values = list(zip(*sorted_vocab))
        ranks = [x[0] for x in sorted(enumerate(keys), key=lambda x: x[1])]
        self._ranks = np.asarray(ranks)
        self._sorted_vocab = values

    def _get_range(self, prefix):
        if not prefix:
            return (0, 0)
        l = bisect.bisect_left(self._sorted_vocab, prefix)
        end_str = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        r = bisect.bisect_left(self._sorted_vocab, end_str, lo=l)
        return l, r

    def _get_mask(self, prefix):
        prefix = prefix.strip()
        if not prefix:
            return self._zero_vec
        if prefix == 'Ġ':
            return self._space_vec
        l, r = self._get_range(prefix)
        mask = np.zeros((len(self._vocab),), dtype=np.float32)
        mask[l:r] = 1
        return mask[self._ranks]
    
    def encode_text_with_partial_word(self, input_text, pad_token_id, device):
        """Encodes the given input using GPT2 tokenizer and returns the ids
        Params 
        input_text     : a string to encode 
        pad_token_id   : in case, string can't be encoded, token id that should be used for padding
        device         : device on which input should be created cuda or cpu
        """
        prefix = input_text.replace('  ', ' ').split(' ')[-1].strip()
        ids = self._tokenizer.encode(' '.join(input_text.replace('  ', ' ').split(' ')[:-1]))
        last_complete_word_pos = len(ids)

        mask = self._get_mask('Ġ' + prefix)
        if not mask.any():
            ids = self._tokenizer.encode(input_text)
            last_token = self._tokenizer.convert_ids_to_tokens([ids[-1]])
            
            mask = self._get_mask(last_token[0])
            if not mask.any(): # not partial word match,the incomplete-word seems to be a complete-token
                mask = np.ones_like(mask)
            else:
                ids = ids[:-1]
        
        input_ids = []
        first_token_masks = []

        if not ids:
            ids = [pad_token_id]  # NOT state token, but, this endup with empty list. This is a hack.
            last_complete_word_pos += 1

        input_ids.append(ids)
        first_token_masks.append(mask)

        input_ids = torch.from_numpy(np.asarray(input_ids, dtype='int64')).to(device)
        first_token_masks = torch.from_numpy(np.asarray(first_token_masks, dtype='float32')).to(device)

        return input_ids, first_token_masks, last_complete_word_pos

    def decode(self, ids):
        return self._tokenizer.decode(ids)
