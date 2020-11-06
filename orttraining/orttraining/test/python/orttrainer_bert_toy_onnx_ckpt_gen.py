import os
import torch

import onnxruntime
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_commons, _test_helpers

from orttrainer_bert_toy_onnx import bert_model_description, \
                                     load_bert_onnx_model, \
                                     generate_random_input_from_model_desc


###############################################################################
# This method is used to generate bert checkpoints for ZeRO
# eg: mpirun -n 4 python3 orttrainer_bert_toy_onnx_ckpt_gen.py
###############################################################################
def testToyBERTModelMixedPrecision(optimizer):
    # Common setup
    local_rank = max(0, int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']))
    world_size = max(1, int(os.environ["OMPI_COMM_WORLD_SIZE"]))
    total_steps = 3

    # set 0 instead of local_rank in order to run on a single GPU on CI.
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optimizer
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': str(device),
        },
        'mixed_precision': {
            'enabled': True,
        },
        'distributed': {
            'world_rank': local_rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'allreduce_post_accumulation': True,
            'deepspeed_zero_optimization': {'stage': 1}},
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, world_size * i+local_rank)
        losses.append(trainer.train_step(*sample_input).cpu().item())

    ckpt_dir = _test_helpers._get_name("ort_ckpt")
    ckpt_prefix = _test_helpers._get_bert_ckpt_prefix(optimizer.name)
    checkpoint.experimental_save_checkpoint(trainer, ckpt_dir, ckpt_prefix)

if __name__ == "__main__":
    testToyBERTModelMixedPrecision(optim.AdamConfig())
    testToyBERTModelMixedPrecision(optim.LambConfig())
