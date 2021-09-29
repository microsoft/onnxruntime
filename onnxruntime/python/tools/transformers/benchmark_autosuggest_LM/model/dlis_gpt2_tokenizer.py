import ctypes
import os

"""
Optimized implementation of BERT tokenizer.
"""
class DLIS_Gpt2Tokenizer:
    """
    Tokenize query for BatchTokenize
    """
    class TokenizeQuery(ctypes.Structure):
        _fields_ = [("input", ctypes.c_char_p),
                    ("token_ids", ctypes.POINTER(ctypes.c_int)),
                    ("num_ids", ctypes.c_int),
                    ("tokenize_ok", ctypes.c_bool)]
    """
    Constructor

    Params
    ------
    vocab_file : str, path to vocab file
    bpe_file : str, path to bpe file
    max_num_tokens : int, maximum number of tokens
    num_contexts : int, the number of tokenizer threads
    max_batch_size : int, the maximum batch size per query
                        default is 0, means equal to num_contexts
    """
    def __init__(self, vocab_file, bpe_file, max_num_tokens, 
                 num_contexts = 1, max_batch_size = 0):
        if num_contexts <= 0:
            raise Exception("num_contexts should large or equal to 1")

        if max_batch_size == 0:
            max_batch_size = num_contexts
        if max_batch_size < num_contexts:
            raise Exception("max_batch_size should equal or larger than num_contexts.")

        self._max_num_ids = max_num_tokens
        self._max_batch_size = max_batch_size
        self._batch_token_ids = []

        dir_path = os.getcwd()
        os.environ['PATH'] = os.path.join(dir_path, './dlis_tokenizer') + ';' + os.environ['PATH']
        self._tokenizer = ctypes.cdll.LoadLibrary("E:\\mycode\\DeepSuggestChanged\\Deepsuggest\\dlis_tokenizer\\tokenizer.dll")

        # Init
        self._tokenizer.Gpt2TokenizerInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._tokenizer.Gpt2TokenizerInit.restype = ctypes.c_bool
        is_success = self._tokenizer.Gpt2TokenizerInit(
            ctypes.c_char_p(vocab_file.encode("utf-8")),
            ctypes.c_char_p(bpe_file.encode("utf-8")),
            num_contexts)

        if not is_success:
            raise Exception("Error in tokenizer initialization")

        self._batch_queries = (self.TokenizeQuery * max_batch_size)()        
        for i in range(max_batch_size):
            self._batch_token_ids.append((ctypes.c_int * max_num_tokens)())
            self._batch_queries[i].token_ids = self._batch_token_ids[i]

        # Tokenize
        self._tokenizer.Gpt2TokenizerTokenize.argtypes = [
            ctypes.POINTER(self.TokenizeQuery),
            ctypes.c_int]

        self._tokenizer.Gpt2TokenizerTokenize.restype = None
        
    """
    Tokenizes an input string, returning ids.

    Params
    ------
    input_str : str, input string to tokenize

    Returns
    -------
    token_ids
    """
    def Tokenize(self, input_str):
        self._batch_queries[0].tokenize_ok = False
        self._batch_queries[0].input = ctypes.c_char_p(input_str.encode("utf-8"))
        self._tokenizer.Gpt2TokenizerTokenize(self._batch_queries, self._max_num_ids)
        
        if not self._batch_queries[0].tokenize_ok:
            raise Exception("Error in tokenization: " + input_str)

        token_ids_ret = []
        for i in range(self._batch_queries[0].num_ids):
            token_ids_ret.append(self._batch_queries[0].token_ids[i])
        return token_ids_ret

