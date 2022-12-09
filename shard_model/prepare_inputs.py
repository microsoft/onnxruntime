import onnx
import os
from onnx import helper, shape_inference
import argparse
import pickle
import numpy as np
import torch
import random
from transformers import AutoConfig


def generate_gpt2_dummy_inputs(batch,
        past_seq_len,
        seq_len,
        num_heads,
        hidden_size,
        num_layer,
        vocab_size,
        float16=True,
        has_position_ids=True,
        has_attention_mask=True,
        input_ids_dtype=torch.int32,
        position_ids_dtype=torch.int32,
        attention_mask_dtype=torch.int32
    ):
    ret = {}
    float_type = torch.float16 if float16 else torch.float32
    past_shape = [2, batch, num_heads, past_seq_len, int(hidden_size / num_heads)]
    for i in range(num_layer):
        past = torch.rand(past_shape, dtype=float_type) * 2.0 - 1.0
        ret[f'past_{i}'] = past.numpy()

    input_ids = torch.randint(low=0, high=vocab_size - 1, size=(batch, seq_len), dtype=input_ids_dtype)
    ret['input_ids'] = input_ids.numpy()

    if has_attention_mask:
        total_seq_len = past_seq_len + seq_len
        attention_mask = torch.ones((batch, total_seq_len), dtype=attention_mask_dtype)
        if total_seq_len >= 2:
            padding_position = random.randint(0, total_seq_len - 1)
            attention_mask[:,padding_position] = 0
            ret['attention_mask'] = attention_mask.numpy()

        if has_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(position_ids < 0, 0)
            position_ids = position_ids[:,past_seq_len:].to(position_ids_dtype)
            ret['position_ids'] = position_ids.numpy()

    return ret 


def main(args):
    # generate fake input data
    config = AutoConfig.from_pretrained(args.model_name)
    inputs = generate_gpt2_dummy_inputs(
                args.batch,
                args.past_seq_len, 
                args.seq_len,
                num_heads=config.n_head,
                hidden_size=config.n_embd,
                num_layer=config.n_layer,
                vocab_size=config.vocab_size
            )

    with open('inputs.pkl', 'wb') as fp:
        pickle.dump(inputs, fp)

    # past shape: (2, batch, num_heads, past_seq_len, q_hidden/num_heads)
    for i in range(config.n_layer):
        past = inputs[f'past_{i}']
        past = np.transpose(past, axes=(0,1,3,2,4)).reshape(2, args.batch, args.past_seq_len, -1)
        past = np.split(past, args.num_shards, axis=3)
        past = [np.transpose(p.reshape(2, args.batch, args.past_seq_len, config.n_head // args.num_shards, -1), axes=(0,1,3,2,4)) for p in past]
        inputs[f'past_{i}'] = past
    for i in range(args.num_shards):
        data = {}
        for k in inputs:
            if k.startswith('past'):
                data[k] = inputs[k][i]
            else:
                data[k] = inputs[k]
        with open(f'inputs-{i}.pkl', 'wb') as fp:
            pickle.dump(data, fp)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument('--batch', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--past-seq-len', type=int)
    parser.add_argument('--model-name', type=str, default='gpt2-large')
    parser.add_argument('--num-shards', type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
