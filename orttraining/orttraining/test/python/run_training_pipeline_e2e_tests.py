import os
import argparse
import subprocess

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)

log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="cwd")
    return parser.parse_args()

def run_subprocess(args, cwd=None, env={}):
    log.info("Running subprocess in '{0}'\n{1}".format(cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    my_env.update(env)
    completed_process = subprocess.run(args, cwd=cwd, check=True, env=my_env)
    log.debug("Subprocess completed. Return code=" +
              str(completed_process.returncode))
    return completed_process

def run_training_pipeline_e2e_tests(cwd):
    # pipeline tests are to be added here:
    log.info("Running pipeline e2e tests.")

    import torch
    ngpus = torch.cuda.device_count()

    command = ['./onnxruntime_training_bert',
               '--ort_log_severity', '1',
               '--optimizer=Lamb',
               '--learning_rate=3e-3',
               '--max_seq_length=128',
               '--max_predictions_per_seq=20',
               '--warmup_ratio=0.2843',
               '--warmup_mode=Poly',
               '--model_name', '/bert_ort/bert_models/nv/bert-large/' +
               'bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12',
               '--train_data_dir', '/bert_data/128/books_wiki_en_corpus/train',
               '--test_data_dir', '/bert_data/128/books_wiki_en_corpus/test',
               '--display_loss_steps', '1',
               '--use_nccl',
               '--use_mixed_precision',
               '--allreduce_in_fp16',
               '--gradient_accumulation_steps', '48',
               '--num_train_steps', '96',
               '--train_batch_size', '50']

    # TODO: currently the CI machine only has 4 GPUs for parallel tests.
    # Fill in more pipeline partition options when the machine has different GPUs counts.
    if ngpus != 4:
        return

    # Test 4-way pipeline parallel
    pp_command = ['mpirun', '-n', str(ngpus)] + command + ['--pipeline_parallel_size', '4', '--cut_group_info',
                                                           '1149:407-1219/1341/1463/1585/1707/1829,' +
                                                           '1881:407-1951/2073/2195/2317/2439/2561,' +
                                                           '2613:407-2683/2805/2927/3049/3171/3293']
    command_str = ', '.join(pp_command)
    log.debug('RUN: ' + command_str)
    run_subprocess(pp_command, cwd=cwd)

    # Test 2-way data parallel + 2-way pipeline parallel
    pp_dp_command = ['mpirun', '-n', str(ngpus)]
    pp_dp_command = pp_dp_command + command
    pp_dp_command = pp_dp_command + ['--data_parallel_size', '2', '--pipeline_parallel_size',
                                     '2', '--cut_group_info',
                                     '1881:407-1951/2073/2195/2317/2439/2561/2683/2805/2927/3049/3171/3293']
    command_str = ', '.join(pp_dp_command)
    log.debug('RUN: ' + command_str)
    run_subprocess(pp_dp_command, cwd=cwd)


args = parse_arguments()
run_training_pipeline_e2e_tests(cwd=args.cwd)
