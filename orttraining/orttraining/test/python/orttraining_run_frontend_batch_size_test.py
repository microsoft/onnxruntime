import collections
import subprocess
import sys

Config = collections.namedtuple(
    "Config",
    [
        "enable_mixed_precision",
        "sequence_length",
        "max_batch_size",
        "max_predictions_per_seq",
        "gelu_recompute",
        "attn_dropout_recompute",
        "transformer_layer_recompute",
    ],
)

configs = [
    Config(True, 128, 46, 20, False, False, False),
    Config(True, 512, 8, 80, False, False, False),
    Config(False, 128, 26, 20, False, False, False),
    Config(False, 512, 4, 80, False, False, False),
    Config(True, 128, 50, 20, True, False, False),
    Config(True, 128, 50, 20, False, True, False),
    Config(True, 128, 76, 20, False, False, True),
    Config(True, 512, 8, 80, True, False, False),
    Config(True, 512, 9, 80, False, True, False),
    Config(True, 512, 15, 80, False, False, True),
]


def run_with_config(config):
    print(
        "##### testing name - {}-{} #####".format(
            "fp16" if config.enable_mixed_precision else "fp32", config.sequence_length
        )
    )
    print("gelu_recompute: ", config.gelu_recompute)
    print("attn_dropout_recompute: ", config.attn_dropout_recompute)
    print("transformer_layer_recompute: ", config.transformer_layer_recompute)

    cmds = [
        sys.executable,
        "orttraining_run_bert_pretrain.py",
        "ORTBertPretrainTest.test_pretrain_throughput",
        "--sequence_length",
        str(config.sequence_length),
        "--max_batch_size",
        str(config.max_batch_size),
        "--max_predictions_per_seq",
        str(config.max_predictions_per_seq),
    ]
    if config.enable_mixed_precision:
        cmds.append("--enable_mixed_precision")
    if config.gelu_recompute:
        cmds.append("--gelu_recompute")
    if config.attn_dropout_recompute:
        cmds.append("--attn_dropout_recompute")
    if config.transformer_layer_recompute:
        cmds.append("--transformer_layer_recompute")

    # access to azure storage shared disk is much slower so we need a longer timeout.
    subprocess.run(cmds, timeout=1200).check_returncode()


for config in configs:
    run_with_config(config)
