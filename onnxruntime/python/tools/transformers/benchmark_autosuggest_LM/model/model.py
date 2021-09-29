"""
Owner: isst
Please implement your own ModelImp
"""
import json
import os
import time
from torch import nn
#from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer

from wrapper.onnx import load_onnx
import utils
from modeling_gpt2 import GPT2LMHeadModel
from tnlg_s import (autocomplete, get_args, gpt2_wrapper_for_generate,
                    initialize_prefix_vocab, DEVICE)
from wrapper.onnx import OnnxModelWrapper

SupportedModels = ['onnx', 'dlis']
MaxSuggestions = 8

class ModelImp:
    def  __init__(self,  args = None):
        self._args = get_args()
        self._no_of_suggestions = args.no_of_suggestions if args != None else MaxSuggestions
        self._pad_token_id =0
        self._is_onnx_model = False

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path if args != None else 'model_files/')
        initialize_prefix_vocab(self.tokenizer)

        if (args != None and args.model_type == 'onnx') or os.getenv('ENABLE_ORT', 0) == '1':
            self._is_onnx_model = True
            print("Loading onnx model...")
            start_time = time.perf_counter()
            self.onnx_model = load_onnx(args.model_path if args != None else os.getenv('ONNX_MODEL_PATH'), device=DEVICE)
            end_time = time.perf_counter()
            print("Time taken for loading onnx model:" + str(end_time - start_time))
            print(str(end_time - start_time))
            io_binding = True if 'cuda' in DEVICE.lower() else False
            self.wrapped_model = OnnxModelWrapper(self.onnx_model, pad_token_id=self._pad_token_id, max_batch_size=1, io_binding=io_binding)
        else:
            print('Loading TNLG Model...')
            start_time = time.perf_counter()
            self.lm_model = GPT2LMHeadModel.from_pretrained(args.model_path if args != None else 'model_files/')
            end_time = time.perf_counter()
            print("Time taken for loading dlis model:" + str(end_time - start_time))

            if isinstance(self.lm_model, nn.Module):
                num_params = sum(w.numel() for w in self.lm_model.parameters())
                print(f'Number of Parameters: {num_params:,}')

            self.lm_model.to(self._args.device)
            self.lm_model.eval()
            self.wrapped_model = gpt2_wrapper_for_generate(self.lm_model, self.lm_model.config.pad_token_id, device=self._args.device)
            print('TNLG Model loaded.')

    # string version of Eval
    # data is a string
    def Eval(self, data):
        top = self._no_of_suggestions
        count_start = 0
        count_start, outputs, probs = autocomplete(self._args, self.wrapped_model, self.tokenizer, [data], count_start, pad_token_id=self._pad_token_id, is_onnx_model=self._is_onnx_model)
        outputs = outputs[0]
        probs = probs[0]
        outputs = [output.strip() for output, prob in zip(outputs, probs)]
        if not data.endswith(' '):
            prefix = data.split(' ')[-1]
            results = [ answer for answer in outputs if answer.lower() != prefix.lower() and len(answer) > 1][:top]
        else:
            results = [ answer for answer in outputs if len(answer) > 1][:top]
        results = json.dumps(results)
        return results

