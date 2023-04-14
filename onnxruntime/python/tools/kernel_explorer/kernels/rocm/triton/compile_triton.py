# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import os
import shutil

import triton
from log_softmax import logsoftmax_function_table
from softmax import softmax_funcion_table

kernel_table = [
    softmax_funcion_table,  # softmax
    logsoftmax_function_table,  # log_softmax
]


"""
function_table = [
        {'name': xx,
         'func': func,
         'sig': sig,
         'kwargs': kwargs
        }
]
"""


def compile(function_table, lib_dir):
    def compile_one(func, sig, **kwargs):
        ret = triton.compile(func, signature=sig, **kwargs)
        return ret

    metadata = []
    for func_desc in function_table:
        name = func_desc["name"]
        group = func_desc["group"]
        sig = func_desc["sig"]
        func = func_desc["func"]
        kwargs = func_desc["kwargs"]

        print("compile func: ", func_desc)

        ret = compile_one(func, sig, **kwargs)

        compile_res = {}
        compile_res["name"] = name
        compile_res["group"] = group
        compile_res["func_name"] = ret.metadata["name"]
        compile_res["num_warps"] = ret.metadata["num_warps"]
        compile_res["shared"] = ret.metadata["shared"]
        if "constants" in kwargs:
            compile_res["constants"] = kwargs["constants"]

        # move tmp hsaco file into current dir
        lib_name = f"{name}.hsaco"
        shutil.copyfile(ret.asm["hsaco_path"], f"{lib_dir}/{lib_name}")
        compile_res["lib_file"] = lib_name
        metadata.append(compile_res)

    return metadata


def main():
    lib_dir = "./libs"
    if not os.path.exists(lib_dir):
        os.mkdir(lib_dir)

    metadata = []
    for k in kernel_table:
        func_tb = k()
        m = compile(func_tb, lib_dir)
        metadata.extend(m)

    print("compile done.")

    # save metadata into json file
    with open(f"{lib_dir}/meta.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
    print("save into file done.")


if __name__ == "__main__":
    main()
