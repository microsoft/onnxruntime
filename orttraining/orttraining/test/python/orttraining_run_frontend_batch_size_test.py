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
        "--enable_mixed_precision",
        "--sequence_length", "128",
        "--max_batch_size", str(config.max_batch_size),
        "--max_predictions_per_seq", str(config.max_predictions_per_seq)]
    if config.gelu_recompute:
        cmds = [*cmds, "--gelu_recompute"]
    if config.attn_dropout_recompute:
        cmds = [*cmds, "--attn_dropout_recompute"]
    if config.transformer_layer_recompute:
        cmds = [*cmds, "--transformer_layer_recompute"]
    subprocess.run(cmds, timeout=1200).check_returncode()

# for max_batch_size in range(15, 50, 1):
#     cmds = [
#         sys.executable,
#         'orttraining_run_bert_pretrain.py',
#         "ORTBertPretrainTest.test_pretrain_throughput",
#         "--enable_mixed_precision",
#         "--sequence_length", "512",
#         "--max_batch_size", str(max_batch_size),
#         "--max_predictions_per_seq", "80",
#         "--transformer_layer_recompute"]

#     subprocess.run(cmds, timeout=12000).check_returncode()

# max_batch_size_fp16_128_20_gelu_recompute = 55
#max_batch_size_fp16_128_20_attn_dropout_recompute = 51
#max_batch_size_fp16_128_20_transformer_layer_recompute = 78

# max_batch_size_fp16_512_80_gelu_recompute = 9
# max_batch_size_fp16_512_80_attn_dropout_recompute = 10
# max_batch_size_fp16_512_80_transformer_layer_recompute = 16

