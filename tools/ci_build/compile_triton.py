# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import importlib.util
import os
import shutil

import triton


def compile(function_table, out_dir):
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

        # print("compile func: ", func_desc)

        ret = compile_one(func, sig, **kwargs)

        compile_res = {}
        compile_res["name"] = name
        compile_res["group"] = group
        compile_res["func_name"] = ret.metadata["name"]
        compile_res["num_warps"] = ret.metadata["num_warps"]
        compile_res["shared"] = ret.metadata["shared"]
        if "constants" in kwargs:
            compile_res["constants"] = kwargs["constants"]

        # move tmp kernel file into current dir
        if "hsaco_path" in ret.asm and os.path.exists(ret.asm["hsaco_path"]):
            # is rocm
            lib_name = f"{name}.hsaco"
            shutil.copyfile(ret.asm["hsaco_path"], f"{out_dir}/{lib_name}")
        elif "cubin" in ret.asm:
            # is cuda
            lib_name = f"{name}.cubin"
            # need to write cubin into file
            with open(f"{out_dir}/{lib_name}", "wb") as fp:
                fp.write(ret.asm["cubin"])
        else:
            raise Exception("not find rocm or cuda compiled kernel")

        compile_res["lib_file"] = lib_name
        metadata.append(compile_res)

    return metadata


def convert_lib_to_obj(lib_file, out_dir):
    obj_file = lib_file.split(".")[0] + ".o"
    command = f"cd {out_dir}; objcopy -I binary -O elf64-x86-64 -B i386:x86-64 {lib_file} {obj_file}; cd -"
    ret = os.system(command)

    if ret != 0:
        raise Exception(f"exec convert command: {command} failed.")
    # check file exist
    if not os.path.exists(f"{out_dir}/{obj_file}"):
        raise Exception(f"the output file not exist, after exec comamnd: {command}")

    return obj_file


def archive_obj_files(obj_files, out_dir, out_obj_file):
    obj_files = " ".join(obj_files)
    command = f"cd {out_dir}; ar rcs {out_obj_file} {obj_files}; cd -"
    ret = os.system(command)

    if ret != 0:
        raise Exception(f"exec convert command: {command} failed.")
    # check file exist
    if not os.path.exists(f"{out_dir}/{out_obj_file}"):
        raise Exception(f"the output file not exist, after exec comamnd: {command}")


def convert_and_save(metadata, header_file, out_dir, out_obj_file):
    c_metadata = []
    binary_files = []
    for m in metadata:
        meta_ele = []
        obj_file = convert_lib_to_obj(m["lib_file"], out_dir)
        binary_files.append(obj_file)

        lib_name = m["lib_file"].replace(".", "_")
        meta_ele.append(f'"_binary_{lib_name}_start"')
        meta_ele.append(f"\"{m['func_name']}\"")
        meta_ele.append(f"\"{m['group']}\"")
        meta_ele.append(f"\"{m['name']}\"")
        meta_ele.append(str(m["num_warps"]))
        meta_ele.append(str(m["shared"]))

        # convert constants
        constants = []
        for k, v in m["constants"].items():
            constants.append(f'{{ "{k}", {v!s}}}')
        meta_ele.append(f"{{ { ', '.join(constants) } }}")

        c_metadata.append(f"{{ { ', '.join(meta_ele) } }}")

    archive_obj_files(binary_files, out_dir, out_obj_file)

    code = f"""
#include <unordered_map>

struct _TritonKernelInfo {{
  const char* name_start;
  const char* func_name;
  const char* group_name;
  const char* name;
  int num_warps;
  int shared;
  std::unordered_map<std::string, int> constants;
}};

const _TritonKernelInfo kernel_infos[] = {{
  { ', '.join(c_metadata) },
}};
    """

    with open(header_file, "w") as fp:
        fp.write(code)


def main(args):
    out_obj_file = args.obj_file
    out_dir = os.path.dirname(out_obj_file)
    out_obj_file = os.path.basename(out_obj_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    metadata = []
    print("[triton kernel] start compile triton kernel.")
    for i, f in enumerate(args.script_files):
        # import module in f, and call function
        spec = importlib.util.spec_from_file_location(f"module_{i}", f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func_tb = module.get_function_table()
        m = compile(func_tb, out_dir)
        metadata.extend(m)

    print("[triton kernel] compile triton kernel done.")

    # save metadata into header file
    convert_and_save(metadata, args.header, out_dir, out_obj_file)
    print("[triton kernel] save into file done.")


def get_arges():
    parser = argparse.ArgumentParser(description="PyTorch Template Finetune Example")
    parser.add_argument(
        "--header", type=str, default="triton_kernel_infos.h", help="the header file that should be generated."
    )
    parser.add_argument("--ort_root", type=str, default="onnxruntime", help="the root dir of onnxruntime.")
    parser.add_argument("--script_files", type=str, nargs="+", help="the root dir of onnxruntime.")
    parser.add_argument("--obj_file", type=str, default="triton_kernel_infos.a", help="output target object files.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arges()
    main(args)
