#!/usr/bin/python

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import numpy as np
import os
import sys
from builtins import range
import collections
from tqdm import tqdm
import time
from tensorrtserver.api import *

sys.path.append('.')
from run_squad import get_answers, convert_examples_to_features, read_squad_examples
from tokenization import BertTokenizer
import json
import pickle


args = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-H', dest='http_headers', metavar="HTTP_HEADER",
                        required=False, action='append',
                        help='HTTP headers to add to inference server requests. ' +
                        'Format is -H"Header:Value".')
    parser.add_argument('--synchronous', action='store_true', help="Wait for previous request to finish before sending next request.")
    
    parser.add_argument("--model_name",
                        type=str,
                        default='bert',
                        help="Specify model to run")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. \
                        True for uncased models, False for cased models.")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Maximal number of examples in a batch')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    
    args = parser.parse_args()
    
    # TRITON client setup
    protocol = ProtocolType.from_str(args.protocol)
    
    model_version = -1
    infer_ctx = InferContext(args.url, protocol, args.model_name, model_version,
                             http_headers=args.http_headers, verbose=args.verbose)
    
    # Preprocess input data
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    cached_features_file = args.predict_file + '_{}_{}.bin'.format(args.max_seq_length, args.doc_stride)

    eval_examples = read_squad_examples(
        input_file=args.predict_file,
        is_training=False,
        version_2_with_negative=args.version_2_with_negative)

    try:
        with open(cached_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        with open(cached_features_file, "wb") as writer:
            pickle.dump(eval_features, writer)
    
    dtype = np.int64
    
    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            unique_ids = ()
            example_indices = ()
            input_ids_data = ()
            input_mask_data = ()
            segment_ids_data = ()
            for i in range(0, min(n, l-ndx)):
                unique_ids = unique_ids + (iterable[ndx + i].unique_id,)
                example_indices = example_indices + (ndx + i,)
                input_ids_data = input_ids_data + (np.array(iterable[ndx + i].input_ids, dtype=dtype),)
                input_mask_data = input_mask_data + (np.array(iterable[ndx + i].input_mask, dtype=dtype),)
                segment_ids_data = segment_ids_data + (np.array(iterable[ndx + i].segment_ids, dtype=dtype),)
            
            inputs_dict = {'input__0': input_ids_data,
                           'input__1': segment_ids_data,
                           'input__2': input_mask_data}
            yield inputs_dict, example_indices, unique_ids
    
    
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    ExampleInfo = collections.namedtuple("ExampleInfo",
                                   ["start_time", "batch_size", "example_ids", "unique_ids"])
    all_results = []
    time_list = []
    outstanding = 0
    sent_prog = tqdm(desc="Sending Requests", total=len(eval_features), file=sys.stdout, unit='sentences')
    recv_prog = tqdm(desc="Processed Requests", total=len(eval_features), file=sys.stdout, unit='sentences')
    if args.synchronous:
        raw_results = []
    
    
    def process_result_cb(example_info, ctx, request_id):
        global outstanding
        
        result = infer_ctx.get_async_run_results(request_id)
        stop = time.time()
        outstanding -= 1
        
        time_list.append(stop - example_info.start_time)
        
        batch_count = example_info.batch_size
        
        for i in range(batch_count):
            unique_id = int(example_info.unique_ids[i])
            start_logits = [float(x) for x in result["output__0"][i].flat]
            end_logits = [float(x) for x in result["output__1"][i].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))
        
        recv_prog.update(n=batch_count)
    
    
    all_results_start = time.time()
    
    for input_dict, example_indices, unique_ids in batch(eval_features, args.batch_size):
        current_bs = len(input_dict['input__0'])
        outputs_dict = {'output__0': InferContext.ResultFormat.RAW,
                        'output__1': InferContext.ResultFormat.RAW}
        start = time.time()
        example_info = ExampleInfo(start_time=start,
                                   batch_size=current_bs,
                                   example_ids=example_indices,
                                   unique_ids=unique_ids
                                   )
        if not args.synchronous:
            outstanding += 1
            result_id = infer_ctx.async_run(partial(process_result_cb, example_info),
                                                    input_dict,
                                                    outputs_dict,
                                                    batch_size=current_bs)
        else:
            result = infer_ctx.run(input_dict, outputs_dict, batch_size=current_bs)
            raw_results.append((example_info, result))
        sent_prog.update(n=current_bs)
    
    # Make sure that all sent requests have been processed
    while outstanding > 0:
        pass
    
    all_results_end = time.time()
    all_results_total = (all_results_end - all_results_start) * 1000.0
    num_batches = (len(eval_features) + args.batch_size - 1) // args.batch_size
    
    if args.synchronous:
        for result in raw_results:
            example_info, batch = result
            for i in range(example_info.batch_size): 
                unique_id = int(example_info.unique_ids[i])
                start_logits = [float(x) for x in batch["output__0"][i].flat]
                end_logits = [float(x) for x in batch["output__1"][i].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
            recv_prog.update(n=example_info.batch_size)
    
    print("-----------------------------")
    print("Individual Time Runs")
    print("Total Time: {} ms".format(all_results_total))
    print("-----------------------------")
    
    print("-----------------------------")
    print("Total Inference Time = %0.2f for"
          "Sentences processed = %d" % (sum(time_list), len(eval_features)))
    print("Throughput Average (sentences/sec) = %0.2f" % (len(eval_features) / all_results_total * 1000.0))
    print("Throughput Average (batches/sec) = %0.2f" % (num_batches / all_results_total * 1000.0))
    print("-----------------------------")
    
    if not args.synchronous:
        time_list.sort()
        
        avg = np.mean(time_list)
        cf_95 = max(time_list[:int(len(time_list) * 0.95)])
        cf_99 = max(time_list[:int(len(time_list) * 0.99)])
        cf_100 = max(time_list[:int(len(time_list) * 1)])
        print("-----------------------------")
        print("Summary Statistics")
        print("Batch size =", args.batch_size)
        print("Sequence Length =", args.max_seq_length)
        print("Latency Confidence Level 95 (ms) =", cf_95 * 1000)
        print("Latency Confidence Level 99 (ms)  =", cf_99 * 1000)
        print("Latency Confidence Level 100 (ms)  =", cf_100 * 1000)
        print("Latency Average (ms)  =", avg * 1000)
        print("-----------------------------")
    
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    answers, nbest_answers = get_answers(eval_examples, eval_features, all_results, args)
    with open(output_prediction_file, "w") as f:
        f.write(json.dumps(answers, indent=4) + "\n")
    with open(output_nbest_file, "w") as f:
        f.write(json.dumps(nbest_answers, indent=4) + "\n")

