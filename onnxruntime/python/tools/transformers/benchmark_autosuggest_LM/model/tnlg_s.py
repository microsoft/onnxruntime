import os
from tempfile import TemporaryFile
import numpy as np
import torch
from torch import nn
import argparse
import logging
import json
import time
import re
import copy
from generate import LMGenerator
#from transformers import GPT2Tokenizer,GPT2LMHeadModel
from transformers import GPT2Tokenizer
from tokenizer import Tokenizer
from modeling_gpt2 import GPT2LMHeadModel
from dlis_gpt2_tokenizer import DLIS_Gpt2Tokenizer
import myutils
import bisect

logging.basicConfig(level=logging.INFO)
logger = logging
DEVICE = 'cuda:0'
DlisTokenizer = DLIS_Gpt2Tokenizer("./dlis_tokenizer/vocab", "./dlis_tokenizer/merges.txt", 64)

def get_args():
    '''Cases
    - Finetune LM and save
    - Loaded an LM(either finetune or not), construct KNN-LM and Save
    - Loaded an saved KNN-LM / regular-LM and evaluate
    - Loaded an saved KNN-LM / regular-LM and do interactive test
    - Loaded an saved KNN-LM / regular-LM and flask service
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to load pickled GPT2 model')
    parser.add_argument('--gpt2_type', type=str, default='gpt2-large')
    parser.add_argument('--train', action='store_true', help='Train model: either finetune GPT2 or construct Knn-LM(--knn)')
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--save', type=str, default=None, help='Path to save trained model / predictions')
    parser.add_argument('--knn', action='store_true', help='Use Knn-LM. If --train, then construct Knn-LM')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--interactive', action='store_true', help='Start an on-terminal interation')
    parser.add_argument('--lamb', type=float, default=0.2, help='Proportion of KNN at interpolation')
    parser.add_argument('--knn_k', type=int, default=1000, help='K in Knn search')
    parser.add_argument('--knn_temp', type=float, default=1.0, help='Temperature for KNN probability estimation')
    parser.add_argument('--context_size', type=int, default=3, help='Number of prefixed words to trigger auto-suggestion')
    parser.add_argument('--num_words', type=int, default=3, help='Number next words to generate')
    parser.add_argument('--num_suggestions', type=int, default=1, help='Number of generated suggestions')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams to use for beam search. Default to disable beam search')
    parser.add_argument('--do_sample', action='store_true', help='If doing random sampling instead of greed search')
    parser.add_argument('--no_knn_cache', action='store_true', help='If to recompute contextualized world embedding for KNN')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training or inferencing')
    parser.add_argument('--epoch', type=int, default=2, help='Epoch to finetune the LM model')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for finetuning using Adam')
    parser.add_argument('--port', type=int, default=None, help='Port to run Flask service')
    parser.add_argument('--predict', action='store_true', help='Use --test_data as input file, and results saved in --save')
    parser.add_argument('--maxlen', type=int, default=16, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Multiplier/divisor to the amplify logits of end-of-text tokens')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Multiplier/divisor to reduce the logits of previous tokens')
    parser.add_argument('--sequence_prob_threshold', type=float, default=0., help='Hard thresholding on resulting sequences')
    parser.add_argument('--onnx', action='store_true', help='Input model is an ONNX object')
    parser.add_argument('--checkpoint_policy', type=str, default='epoch:1', help='''Save frequency. Example: epoch:2 for every 2 epochs. batch:1000 for every 1000 batches. Default: epoch:1. ''')
    parser.add_argument('--write_prob', action='store_true',help='Write probility to the prediction file')
    parser.add_argument('--disable_prefix_filter', action='store_true',help='Disable prefix filter for the prediction')
    #args = parser.parse_args()
    arg_list = get_config_args()
    if arg_list and len(arg_list) > 0:
        args = parser.parse_args(arg_list)
    else:
        args, unknown = parser.parse_known_args()
    assert re.match('(epoch|batch):\d+$', args.checkpoint_policy)
    logging.info(args)
    return args

def get_config_args():
    config_file_path = os.getenv("_TNLG_S_CONFIG_", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tnlg_s_config.json"))
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config_json = json.load(config_file)
        arg_list = []
        #[arg_list.extend(['--' + key, str(value)])for key, value in config_json.items()]
        [arg_list.extend(['--' + key, str(value)]) if value != "" else arg_list.extend(['--' + key]) for key, value in config_json.items()]
        print(arg_list)
        return arg_list
    else:
        print(f'not find {config_file_path}')
        return None

def preprocess_inputs(input_ids, pad_token_id):
    # trim over-pading. Assume any padding (left, middle, right)
    pid = pad_token_id
    mask = (input_ids != pid)
    if mask.all():
        return {'input_ids': input_ids}

    maxlen = 1
    mask = mask.long()
    mask_any = mask.sum(0).cpu().detach().numpy()
    for i in range(len(mask_any), 0, -1):
        if mask_any[i - 1] > 0:
            if maxlen < i:
                maxlen = i
            break

    input_ids = input_ids[:, :maxlen].contiguous()
    position_ids = mask.cumsum(-1) - 1
    position_ids = position_ids * (position_ids >= 0).long()

    attention_mask = mask.float()
    return {'input_ids': input_ids, 'position_ids': position_ids, 'attention_mask': attention_mask}

def gpt2_wrapper_for_generate(model, pad_token_id, device=DEVICE):
    '''Issue with batch mode: masking is impossible with past is not None in Huggingface's implementation.(TODO)
    '''
    enable_dlis = (os.getenv('ENABLE_DLIS', '0') == '1')

    def _model(input_ids, model_status, generator_status):
        # input_ids: (batch_size, num_sequences_per_sample, sequence_len)
        past = model_status  # model_status here is past
        batch_size, num_sequences_per_sample, cur_len = input_ids.shape
        input_ids = input_ids.view(-1, cur_len)
        # model_inputs = preprocess_inputs(input_ids, pad_token_id)
        model_inputs = {'input_ids': input_ids}
        if past is not None:
            for k in model_inputs:
                if k !=  'attention_mask':
                    # input last token is needed. But mask should contains all historical input masks
                    # This is actually impossible due to Huggingface's code problem with attention mask when `past` is not None.
                    model_inputs[k] = model_inputs[k][:, -1:]
            
            if enable_dlis:
                model_inputs['past'] = generator_status['input_seq_index']
            else:
                if generator_status:
                    input_seq_index = generator_status['input_seq_index']
                    input_seq_index = input_seq_index + torch.arange(batch_size).unsqueeze(1).to(device) * generator_status['last_num_sequences_per_sample']
                    input_seq_index = input_seq_index.view(-1)
                    model_status = tuple(layer_past.index_select(1, input_seq_index) for layer_past in past)
                model_inputs['past'] = model_status

        start_time = time.perf_counter()
        outputs = model(**model_inputs)
        end_time = time.perf_counter()
        infer_time = (end_time - start_time) * 1000
        myutils.total_infer_time += infer_time

        next_token_logits = outputs[0][:, -1].view((batch_size, num_sequences_per_sample, -1))
        return next_token_logits, outputs[1]

    return _model

def process_results(output_ids, output_probs, tokenizer, last_complete_word_position):
    #TODO why is this expanding to a 3D tensor
    if output_ids.dim() == 2: # one output
        output_ids = output_ids[:, None]

    returns = []
    returns_probs = []
    for i, output_i in enumerate(output_ids):
        returns_i = []
        probs_i = []
        for j, output_ij in enumerate(output_i):
            # TODO is args.sequence_prob_threshold really needed?
            if returns_i and output_probs[i, j] < 0:
                break
            output_ij = output_ij.tolist()
            for k in range(len(output_ij)):
                if output_ij[k] == tokenizer.eos_token_id:
                    output_ij = output_ij[:k]
                    break

            output_ij = output_ij[last_complete_word_position:]
            output_text = tokenizer.decode(output_ij).rstrip()
            if output_text not in returns_i:
                returns_i.append(output_text)
                probs_i.append(output_probs[i, j].item())

        returns.append(returns_i)
        returns_probs.append(probs_i)

    return returns, returns_probs


@torch.no_grad()
def autocomplete(args, model, tokenizer : Tokenizer, input_text, pad_token_id=0, verbose=False, is_onnx_model = False):
    start_time = time.perf_counter()

    input_ids, first_token_masks, last_complete_word_position = tokenizer.encode_text_with_partial_word(input_text, pad_token_id, args.device)

    generator = LMGenerator(max_length=input_ids.size(1) + args.num_words,
                            num_return_sequences=args.num_suggestions, do_sample=args.do_sample, num_beams=args.num_beams,
                            pad_token_id=tokenizer.eos_token_id, eos_token_ids=[tokenizer.eos_token_id], top_p=0.95,
                            length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty, enable_ort = is_onnx_model)

    output_ids, output_probs = generator.generate(model, input_ids, first_token_masks=first_token_masks)

    return process_results(output_ids, output_probs, tokenizer, last_complete_word_position)