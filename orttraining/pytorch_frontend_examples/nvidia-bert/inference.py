# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
""" BERT inference script. Does not depend on dataset. """

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from tqdm import tqdm, trange
from types import SimpleNamespace

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from run_squad import _get_best_indices, _compute_softmax, get_valid_prelim_predictions, get_answer_text

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


import math
import json
import numpy as np
import collections


def preprocess_tokenized_text(doc_tokens, query_tokens, tokenizer, 
                              max_seq_length, max_query_length):
    """ converts an example into a feature """
    
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    
    # truncate if too long
    length = len(all_doc_tokens)
    length = min(length, max_tokens_for_doc)
    
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    for i in range(length):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        token_is_max_context[len(tokens)] = True
        tokens.append(all_doc_tokens[i])
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    tensors_for_inference = {
                             'input_ids': input_ids, 
                             'input_mask': input_mask, 
                             'segment_ids': segment_ids
                            }
    tensors_for_inference = SimpleNamespace(**tensors_for_inference)
    
    tokens_for_postprocessing = {
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context
                                }
    tokens_for_postprocessing = SimpleNamespace(**tokens_for_postprocessing)
    
    return tensors_for_inference, tokens_for_postprocessing


RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])


def get_answer(doc_tokens, tokens_for_postprocessing, 
               start_logits, end_logits, args):
    
    result = RawResult(start_logits=start_logits, end_logits=end_logits)
    
    predictions = []
    Prediction = collections.namedtuple('Prediction', ['text', 'start_logit', 'end_logit'])
    
    if args.version_2_with_negative:
        null_val = (float("inf"), 0, 0)
    
    start_indices = _get_best_indices(result.start_logits, args.n_best_size)
    end_indices = _get_best_indices(result.end_logits, args.n_best_size)
    prelim_predictions = get_valid_prelim_predictions(start_indices, end_indices, 
                                                      tokens_for_postprocessing, result, args)
    prelim_predictions = sorted(
                                prelim_predictions,
                                key=lambda x: (x.start_logit + x.end_logit),
                                reverse=True
                                )
    if args.version_2_with_negative:
        score = result.start_logits[0] + result.end_logits[0]
        if score < null_val[0]:
            null_val = (score, result.start_logits[0], result.end_logits[0])
    
    doc_tokens_obj = {
                      'doc_tokens': doc_tokens, 
                     }
    doc_tokens_obj = SimpleNamespace(**doc_tokens_obj)

    curr_predictions = []
    seen_predictions = []
    for pred in prelim_predictions:
        if len(curr_predictions) == args.n_best_size:
            break
        if pred.end_index > 0: # this is a non-null prediction
            final_text = get_answer_text(doc_tokens_obj, tokens_for_postprocessing, pred, args)
            if final_text in seen_predictions:
                continue
        else:
            final_text = ""
        
        seen_predictions.append(final_text)
        curr_predictions.append(Prediction(final_text, pred.start_logit, pred.end_logit))
    predictions += curr_predictions
    
    # add empty prediction
    if args.version_2_with_negative:
        predictions.append(Prediction('', null_val[1], null_val[2]))
    
    nbest_answers = []
    answer = None
    nbest = sorted(predictions,
                   key=lambda x: (x.start_logit + x.end_logit),
                   reverse=True)[:args.n_best_size]
    
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry and entry.text:
            best_non_null_entry = entry
    probs = _compute_softmax(total_scores)
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_answers.append(output)
    if args.version_2_with_negative:
        score_diff = null_val[0] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
        if score_diff > args.null_score_diff_threshold:
            answer = ""
        else:
            answer = best_non_null_entry.text
    else:
        answer = nbest_answers[0]['text']
    
    return answer, nbest_answers


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")
    
    ## Other parameters
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. ")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--question", default="Most antibiotics target bacteria and don't affect what class of organisms? ", 
                                              type=str, help="question")
    parser.add_argument("--context", default="Within the genitourinary and gastrointestinal tracts, commensal flora serve as biological barriers by competing with pathogenic bacteria for food and space and, in some cases, by changing the conditions in their environment, such as pH or available iron. This reduces the probability that pathogens will reach sufficient numbers to cause illness. However, since most antibiotics non-specifically target bacteria and do not affect fungi, oral antibiotics can lead to an overgrowth of fungi and cause conditions such as a vaginal candidiasis (a yeast infection). There is good evidence that re-introduction of probiotic flora, such as pure cultures of the lactobacilli normally found in unpasteurized yogurt, helps restore a healthy balance of microbial populations in intestinal infections in children and encouraging preliminary data in studies on bacterial gastroenteritis, inflammatory bowel diseases, urinary tract infection and post-surgical infections. ", 
                                              type=str, help="context")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate. ")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, then the model can reply with "unknown". ')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=-11.0,
                        help="If null_score - best_non_null is greater than the threshold predict 'unknown'. ")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="use mixed-precision")
    parser.add_argument("--local_rank", default=-1, help="ordinal of the GPU to use")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    
    # Prepare model
    config = BertConfig.from_json_file(args.config_file)
    
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    
    # initialize model
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"])
    model.to(device)
    if args.fp16:
        model.half()
    model.eval()
    
    print("question: ", args.question)
    print("context: ", args.context)
    print()
    
    # preprocessing
    doc_tokens = args.context.split()
    query_tokens = tokenizer.tokenize(args.question)
    feature = preprocess_tokenized_text(doc_tokens, 
                                        query_tokens, 
                                        tokenizer, 
                                        max_seq_length=args.max_seq_length, 
                                        max_query_length=args.max_query_length)
    
    tensors_for_inference, tokens_for_postprocessing = feature
    
    input_ids = torch.tensor(tensors_for_inference.input_ids, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(tensors_for_inference.segment_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(tensors_for_inference.input_mask, dtype=torch.long).unsqueeze(0)
    
    # load tensors to device
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    
    # run prediction
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, segment_ids, input_mask)
    
    # post-processing
    start_logits = start_logits[0].detach().cpu().tolist()
    end_logits = end_logits[0].detach().cpu().tolist()
    answer, answers = get_answer(doc_tokens, tokens_for_postprocessing, 
                                 start_logits, end_logits, args)
    
    # print result
    print()
    print(answer)
    print()
    print(json.dumps(answers, indent=4))


if __name__ == "__main__":
    main()

