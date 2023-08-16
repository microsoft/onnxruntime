import argparse
import json
import logging
import os
import re
import subprocess

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger('time_ort')


class OrtPlayground:
    ort_executable = "./build/RelWithDebInfo/onnxruntime_perf_test" + \
        (".exe" if os.name == "nt" else "")

    def __init__(self, model):
        self.onnx_file = os.path.abspath(model)
        self.model_dir = os.path.dirname(self.onnx_file)

    def run(self, *args):
        launch_args = [
            OrtPlayground.ort_executable,
            "-I" , *args, self.onnx_file,
        ]

        try:
            r = subprocess.run(launch_args, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

            stdout = r.stdout.decode("UTF-8")
        except subprocess.CalledProcessError as e:
            logger.warning(f"run failed for model: {self.onnx_file}")
            stdout = e.stdout.decode("UTF-8")
        return launch_args, stdout

    def get_result(self, stdout):
        try:
            results = {
                "mean_end_to_end_latency_ms": float(re.findall(r'Average inference time cost: ([\d.]*) ms', stdout)[0]),
                "min_end_to_end_latency_ms": float(re.findall(r'Min Latency: ([\d.]*) s', stdout)[0]) * 1000,
                "max_end_to_end_latency_ms": float(re.findall(r'Max Latency: ([\d.]*) s', stdout)[0]) * 1000,
                "creation_time_ms": float(re.findall(r'Session creation time cost: ([\d.]*) s', stdout)[0]) * 1000,
            }

        except IndexError:
            logger.warning(
                "onnxruntime bench failed, error message will be returned")
            results = {
                "stdout": stdout
            }
        return results


def main(ofname, onnx_files):
    logger.setLevel(level=logging.INFO)

    summary = {}
    # modify this to your needs
    summary["experiments"] = {
        "cuda_ep_nchw": ["-e", "cuda", "-t", "5", "-q"],
        "cuda_ep_nhwc": ["-e", "cuda", "-t", "5", "-q", "-l"],
    }

    for onnx_file in tqdm(onnx_files):
        model = {}
        for name, additional_args in summary["experiments"].items():
            launcher = OrtPlayground(onnx_file)
            _, stdout_run = launcher.run(*additional_args)
            result = launcher.get_result(stdout_run)
            model[name] = result
        summary[onnx_file] = model
    with open(ofname, 'w') as f:
        json.dump(summary, f, indent=2)

def plot(ofname, **kwargs):
    json = pd.read_json(ofname)
    json.index.name = "experimet_name"
    json["experiments"] = json["experiments"].apply(lambda x: " ".join(x))
    json = json.set_index("experiments", append=True)
    json = json.T
    experiments = {}
    for col in json.columns:
        experiments[col] = json.pop(col).apply(pd.Series)
    json = pd.concat(experiments.values(), keys=experiments.keys())
    json = json[["mean_end_to_end_latency_ms"]].unstack(2).T
    import matplotlib.pyplot as plt
    plt.tight_layout()
    json = json.droplevel(0)
    json.plot.barh(figsize=(15,10)).figure.savefig("benchmark.png")

    json["speedup"] = (json.iloc[:, 0] / json.iloc[:, 1])
    print(json.round(2))
    print(f"Average speedup: {round(json['speedup'].mean(), 2)}")


if __name__ == "__main__":
    r = subprocess.run("git rev-parse HEAD".split(" "), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

    git_hash = r.stdout.decode("UTF-8")

    parser = argparse.ArgumentParser(prog='time_ort')

    parser.add_argument('--onnx_files', nargs="+", type=str)
    parser.add_argument('--ofname', action='store',
                        default=f"benchmark_{git_hash[:-2]}.json")
    args = parser.parse_args()

    main(**vars(args))
    plot(**vars(args))
