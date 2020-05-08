#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import argparse
import numpy as np
from builtins import range
from tensorrtserver.api import *
# 
import sys
sys.path.append('../')
from inference import preprocess_tokenized_text, get_answer
from tokenization import BertTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='batch size for inference. default: 1')
    parser.add_argument("--triton-model-name", type=str, default="model_name", 
                        help="the name of the model used for inference")
    parser.add_argument("--triton-model-version", type=int, default=-1, 
                        help="the version of the model used for inference")
    parser.add_argument("--triton-server-url", type=str, default="localhost:8000", 
                        help="Inference server URL. Default is localhost:8000.")
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')
    
    ## pre- and postprocessing parameters
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. ")
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
    # input texts
    parser.add_argument("--question", default="Most antibiotics target bacteria and don't affect what class of organisms? ", 
                                              type=str, help="question")
    parser.add_argument("--context", default="Within the genitourinary and gastrointestinal tracts, commensal flora serve as biological barriers by competing with pathogenic bacteria for food and space and, in some cases, by changing the conditions in their environment, such as pH or available iron. This reduces the probability that pathogens will reach sufficient numbers to cause illness. However, since most antibiotics non-specifically target bacteria and do not affect fungi, oral antibiotics can lead to an overgrowth of fungi and cause conditions such as a vaginal candidiasis (a yeast infection). There is good evidence that re-introduction of probiotic flora, such as pure cultures of the lactobacilli normally found in unpasteurized yogurt, helps restore a healthy balance of microbial populations in intestinal infections in children and encouraging preliminary data in studies on bacterial gastroenteritis, inflammatory bowel diseases, urinary tract infection and post-surgical infections. ", 
                                              type=str, help="context")
    
    args = parser.parse_args()
    args.protocol = ProtocolType.from_str(args.protocol)
    
    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(args.triton_server_url, args.protocol, 
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Health for model {}".format(args.triton_model_name))
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))
    
    # Create a status context and get server status
    status_ctx = ServerStatusContext(args.triton_server_url, args.protocol, args.triton_model_name, 
                                     http_headers=args.http_headers, verbose=args.verbose)
    print("Status for model {}".format(args.triton_model_name))
    print(status_ctx.get_server_status())
    
    # Create the inference context for the model.
    infer_ctx = InferContext(args.triton_server_url, args.protocol, args.triton_model_name, args.triton_model_version, 
                             http_headers=args.http_headers, verbose=args.verbose)
    
    print("question: ", args.question)
    print("context: ", args.context)
    print()
    
    # pre-processing
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    
    doc_tokens = args.context.split()
    query_tokens = tokenizer.tokenize(args.question)
    feature = preprocess_tokenized_text(doc_tokens, 
                                        query_tokens, 
                                        tokenizer, 
                                        max_seq_length=args.max_seq_length, 
                                        max_query_length=args.max_query_length)
    
    tensors_for_inference, tokens_for_postprocessing = feature
    
    dtype = np.int64
    input_ids = np.array(tensors_for_inference.input_ids, dtype=dtype)[None,...] # make bs=1
    segment_ids = np.array(tensors_for_inference.segment_ids, dtype=dtype)[None,...] # make bs=1
    input_mask = np.array(tensors_for_inference.input_mask, dtype=dtype)[None,...] # make bs=1
    
    assert args.batch_size == input_ids.shape[0]
    assert args.batch_size == segment_ids.shape[0]
    assert args.batch_size == input_mask.shape[0]
    
    # prepare inputs
    input_dict = {
                           "input__0" : tuple(input_ids[i] for i in range(args.batch_size)), 
                           "input__1" : tuple(segment_ids[i] for i in range(args.batch_size)), 
                           "input__2" : tuple(input_mask[i] for i in range(args.batch_size))
    }
    
    # prepare outputs
    output_keys = [
                           "output__0",
                           "output__1"
    ]
    
    output_dict = {}
    for k in output_keys:
        output_dict[k] = InferContext.ResultFormat.RAW
    
    # Send inference request to the inference server. 
    result = infer_ctx.run(input_dict, output_dict, args.batch_size)
    
    # get the result
    start_logits = result["output__0"][0].tolist()
    end_logits = result["output__1"][0].tolist()
    
    # post-processing
    answer, answers = get_answer(doc_tokens, tokens_for_postprocessing, 
                                 start_logits, end_logits, args)
    
    # print result
    print()
    print(answer)
    print()
    print(json.dumps(answers, indent=4))

