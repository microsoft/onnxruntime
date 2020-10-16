import sys
import collections
import subprocess

Config = collections.namedtuple(
    "Config",
    [
        "enable_mixed_precision",
        "sequence_length",
        "max_batch_size",
        "max_predictions_per_seq",
        "gelu_recompute",
        "attn_dropout_recompute",
        "transformer_layer_recompute"])

configs = [
    Config(True, 128, 46, 20, False, False, False),
    Config(True, 512, 8, 80, False, False, False),
    Config(False, 128, 26, 20, False, False, False),
    Config(False, 512, 4, 80, False, False, False),
    Config(True, 128, 55, 20, True, False, False),
    Config(True, 128, 51, 20, False, True, False),
    Config(True, 128, 78, 20, False, False, True),
    Config(True, 512, 9, 80, True, False, False),
    Config(True, 512, 10, 80, False, True, False),
    Config(True, 512, 16, 80, False, False, True),
]

for config in configs:
    print("##### testing name - {}-{} #####".format("fp16" if config.enable_mixed_precision else "fp32",
                                                        config.sequence_length))
    print("gelu_recompute: ", config.gelu_recompute)
    print("attn_dropout_recompute: ", config.attn_dropout_recompute)
    print("transformer_layer_recompute: ", config.transformer_layer_recompute)

    cmds = [
        sys.executable,
        'orttraining_run_bert_pretrain.py',
        "ORTBertPretrainTest.test_pretrain_throughput",
        "--sequence_length", str(config.sequence_length),
        "--max_batch_size", str(config.max_batch_size),
        "--max_predictions_per_seq", str(config.max_predictions_per_seq)]
    if config.enable_mixed_precision:
        cmds = [*cmds, "--enable_mixed_precision"]
    if config.gelu_recompute:
        cmds = [*cmds, "--gelu_recompute"]
    if config.attn_dropout_recompute:
        cmds = [*cmds, "--attn_dropout_recompute"]
    if config.transformer_layer_recompute:
        cmds = [*cmds, "--transformer_layer_recompute"]
    subprocess.run(cmds, timeout=1200).check_returncode()
