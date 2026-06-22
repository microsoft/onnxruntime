from logging import exception
from transformers import GPT2Tokenizer
import bisect
import numpy as np
import myutils
import torch

# These are static limits to shave some kind of limitation on the processing
# can be increased as needed
MAX_TOKENS_LENGTH=1024

class Tokenizer:
    """ This class represents an object of GPT2Tokenizer required to encode and decode text and output ids of a model respectively.
    It only needs the path of tokenizer to initialize
    """
    def __init__(self, tokenizer_path:str):
        self._is_gpt2 = False
        if tokenizer_path == "gpt2":
            self._is_gpt2 = True

        self._tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
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
        try:
            prefix = input_text.replace('  ', ' ').split(' ')[-1].strip()
            ids = self._tokenizer.encode(' '.join(input_text.replace('  ', ' ').split(' ')[:-1]))
            last_complete_word_pos = len(ids)

            first_token_masks = []
            mask = self._get_mask('Ġ' + prefix)
            if not mask.any():
                ids = self._tokenizer.encode(input_text)
                last_token = self._tokenizer.convert_ids_to_tokens([ids[-1]])
                
                mask = self._get_mask(last_token[0])
                if not mask.any():
                    mask = np.ones_like(mask)
                else:
                    ids = ids[:-1]
            
            first_token_masks.append(mask)

            input_ids = []
            if not ids:
                ids = [pad_token_id]
                last_complete_word_pos += 1

            # is this possible? Input text is not an indicator of number of tokens it would
            # produce so checking the post encoded length             
            if len(ids) > MAX_TOKENS_LENGTH:
                raise Exception(f"Input{input_text} is tokenized into more than {MAX_TOKENS_LENGTH} tokens")

            input_ids.append(ids)

            first_token_masks = torch.from_numpy(np.asarray(first_token_masks, dtype='float32')).to(device)
            input_ids = torch.from_numpy(np.asarray(input_ids, dtype='int64')).to(device)
        except Exception as e:
            raise Exception(f"Caught exception during encoding {str(e)}")
        
        if self._is_gpt2:
            first_token_masks = first_token_masks[:, :self.get_gpt2_token_count()]
        return input_ids, first_token_masks, last_complete_word_pos

    def decode(self, ids):
        """
        decodes given ids into text
        """
        return self._tokenizer.decode(ids)

    def get_gpt2_token_count(self):
        """
        returns the original number of tokens. The extra tokens are in addition to gpt2 tokenizer.
        The model would be trained using the extra tokens for some benefit, but these have to be masked out during inference

        TODO This can be changed to configurable in future
        """
        return 50257
