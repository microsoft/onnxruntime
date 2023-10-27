import os
import logging

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dist_settings import init_dist, get_rank, get_size, barrier, print_out

logger = logging.getLogger("")

def setup_torch_model(args, location, use_auth_token, torch_dtype=torch.float32, use_cuda=True):
    world_size = get_size()
    logger.info(f'world_size: {world_size}')
    rank = get_rank()
    barrier()

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    for i in range(world_size):
        if i == rank:
            l_config = LlamaConfig.from_pretrained(location, use_auth_token=use_auth_token, cache_dir=args.cache_dir)
            l_config.use_cache = True
            #l_config.num_hidden_layers = 2
            llama = LlamaForCausalLM.from_pretrained(location, use_auth_token=use_auth_token, config=l_config, 
                                                     torch_dtype=torch_dtype, cache_dir=args.cache_dir)
            if world_size > 1:
                llama.parallel_model()
            if use_cuda:
                llama.to(torch.device(rank))
            llama.eval()
            llama.requires_grad_(False)
        barrier()
    return l_config, llama