import os
import sys

os.environ["KERNEL_EXPLORER_BUILD_DIR"] = "/root/onnxruntime/build/Linux/Release"


import multiprocessing as mp
from multiprocessing import Pool, current_process


def profile(name, *args, **kwargs):
    import kernel_explorer as ke

    ke.set_return_tuning_results()
    ke.set_dispatchable_pattern("*Tunable*")
    # print(os.environ["HIP_VISIBLE_DEVICES"])
    if name == "gemm":
        from gemm_test import profile_with_args as profile

        return profile(*args, **kwargs)
    elif name == "softmax":
        from softmax_test import profile_with_args as profile

        return profile(*args, **kwargs)
    else:
        return []


def init():
    pidx = int(current_process()._identity[0]) - 1
    start_gpu = 4
    num_gpu = 12
    os.environ["HIP_VISIBLE_DEVICES"] = str(pidx % num_gpu + start_gpu)


if __name__ == "__main__":
    # configs = [
    #     ("gemm", "float16", False, False, 1, 8912, 8912),
    #     ("gemm", "float16", False, False, 8, 8912, 8912),
    #     # ("gemm", "float16", False, False, 16, 8912, 8912),
    #     # ("gemm", "float16", False, False, 24, 8912, 8912),
    #     # ("gemm", "float16", False, False, 32, 8912, 8912),
    #     # ("gemm", "float16", False, False, 40, 8912, 8912),
    #     # ("gemm", "float16", False, False, 48, 8912, 8912),
    #     ("softmax", 1, 1024, False, "float16"),
    #     ("softmax", 2, 1024, False, "float16"),
    # ]

    Ms = list(range(4096))
    # Ms = [i for i in Ms if i % 8 != 0] + [4096]
    Ms = [4096]
    NKs = [[12288, 4096], [22016, 4096], [4096, 11008], [4096, 4096]]

    tuning_result = "/ws/ort-tuning-results.json"
    with open(tuning_result) as f:
        import json
        tuning_results = json.load(f)[0]["results"]
        configs = []
        for value_dict in tuning_results.values():
            for key in value_dict:
                _, M, N, K = key.split("_")
                M, N, K = int(M), int(N), int(K)
                configs.append(("gemm", "float16", False, False, M, N, K))
        print(len(configs))
        # exit()

    # configs = []
    # for M in Ms:
    #     for NK in NKs:
    #         configs.append(("gemm", "float16", False, False, M, NK[0], NK[1]))

    mp.set_start_method("spawn")

    with Pool(processes=12, initializer=init) as pool:
        ret = pool.starmap(profile, configs, chunksize=1)

    from pprint import pprint
    from onnxruntime.tools.offline_tuning import Merger

    m = Merger()
    for tr in ret:
        m.merge(tr)

    # pprint(m.get_merged())
    with open("merged.json", "w") as f:
        import json
        json.dump(m.get_merged(), f, indent=2)
