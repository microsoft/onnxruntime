#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved. 
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

import torch
import pickle
import argparse
import deployer_lib
# 
import sys
sys.path.append('../')
sys.path.append('.')
from modeling import BertForQuestionAnswering, BertConfig
from tokenization import BertTokenizer
from run_squad import convert_examples_to_features, read_squad_examples


def get_model_args(model_args):
    ''' the arguments initialize_model will receive '''
    parser = argparse.ArgumentParser()
    ## Required parameters by the model. 
    parser.add_argument("--checkpoint", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="The checkpoint of the model. ")
    parser.add_argument('--batch_size', 
                        default=8, 
                        type=int, 
                        help='Batch size for inference')
    parser.add_argument("--bert_model", default="bert-large-uncased", type=str, 
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", 
                        action='store_true', 
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
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
    parser.add_argument("--config_file", 
                        default=None, 
                        type=str, 
                        required=True, 
                        help="The BERT model config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="use mixed-precision")
    parser.add_argument('--nbatches', 
                        default=2, 
                        type=int, 
                        help='Number of batches in the inference dataloader. Default: 10. ')
    return parser.parse_args(model_args)


def initialize_model(args):
    ''' return model, ready to trace '''
    config = BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForQuestionAnswering(config)
    model.enable_apex(False)
    state_dict = torch.load(args.checkpoint, map_location='cpu')["model"]
    model.load_state_dict(state_dict)
    if args.fp16:
        model.half()
    return model


def get_dataloader(args):
    ''' return dataloader for inference '''
    
    # Preprocess input data
    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    
    cached_features_file = args.predict_file + '_{}_{}.bin'.format(args.max_seq_length, args.doc_stride)
    try:
        with open(cached_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_examples = read_squad_examples(
            input_file=args.predict_file,
            is_training=False,
            version_2_with_negative=args.version_2_with_negative)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        with open(cached_features_file, "wb") as writer:
            pickle.dump(eval_features, writer)
    
    data = []
    for feature in eval_features:
        input_ids = torch.tensor(feature.input_ids, dtype=torch.int64)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.int64)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.int64)
        inp = (input_ids, segment_ids, input_mask)
        data.append(inp)
    
    if args.nbatches > 0:
        data = data[:args.nbatches*args.batch_size]
    
    test_loader = torch.utils.data.DataLoader(
        data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True)
    
    return test_loader


if __name__=='__main__':
    # don't touch this!
    deployer, model_argv = deployer_lib.create_deployer(sys.argv[1:]) # deployer and returns removed deployer arguments
    
    model_args = get_model_args(model_argv)
    
    model = initialize_model(model_args)
    dataloader = get_dataloader(model_args)
    
    deployer.deploy(dataloader, model)

