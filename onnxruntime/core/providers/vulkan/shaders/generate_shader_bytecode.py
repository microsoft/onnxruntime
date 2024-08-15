import argparse
import os
import platform
import shutil
import subprocess
import sys

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert shaders from GLSL to bytecode",
        usage="""
         Run the script to convert shaders from various sources to compiled bytecode.
         The convert_shaders.py script from kompute is used.
         All output directories are relative to the location of this script.

         Shaders from various sources can be pre-processed and added to 'compiled_shaders'.
         glslangValidator is used to compile all .comp files in 'compiled_shaders'.

         Headers containing the compiled shader byte code are written to the 'compiled_headers' folder.
         """,
    )

    parser.add_argument("--build_dir", type=Path, required=True,
                        help="The ORT build output directory that contains the '_deps' folder with 3rd party source")
    parser.add_argument("--shader_compiler", type=Path, required=False,
                        help="The path to the binary that will compile the shaders. "
                             "Default is glslangValidator from PATH with fallback to the Vulkan SDK "
                             "(if available via VULKAN_SDK environment variable).",
                        default="glslangValidator")

    args = parser.parse_args()

    if shutil.which(str(args.shader_compiler)) is None:
        sdk_compiler = os.environ.get("VULKAN_SDK", "") + "/bin/glslangValidator"
        if shutil.which(str(sdk_compiler.resolve())) is None:
            raise RuntimeError(f"Could not find shader compiler '{args.shader_compiler}' in PATH or VULKAN SDK")

        args.shader_compiler = sdk_compiler

    args.shader_dir = Path(__file__).resolve().parent
    # change to shader_dir so we can use relative paths for the files. the path used when calling convert_shaders.py
    # ends up in the name of the struct with the bytecode in the compiled header file, so we want it to be short and
    # meaningful.
    os.chdir(args.shader_dir)

    # working directory to assemble pre-processed .comp shader files to compile.
    # convert_shaders.py outputs the spv to the same directory, and as this directory name will end up in the struct
    # containing bytecode in the header file we use 'compiled_shaders' rather than 'shaders_to_compile'.
    args.shaders_to_compile_dir = Path("compiled_shaders")

    # location that the headers containing the compiled shader bytecode will be written to.
    # each kernel includes one or more shaders from here.
    args.output_dir = Path("include")

    # TODO: decide if we want to clean these up first. It may be useful to leave the to_compile directory as-is so we
    # can manually/change things there.
    args.shaders_to_compile_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(exist_ok=True)

    args.convert_script = ((args.build_dir / "_deps" / "kompute-src" / "scripts" / "convert_shaders.py")
                           .resolve(strict=True))

    return args


def preprocess_ncnn_shaders(args):

    # for each .comp
    ncnn_shader_dir = args.shader_dir / "ncnn"

    with open(ncnn_shader_dir / "preamble.comp.in", "r") as preamble_file:
        preamble = preamble_file.read()

    includes = {"vulkan_activation.comp"}

    for file in ncnn_shader_dir.glob(f'*.comp'):
        if file.name in includes:
            continue

        with (open(file, "r") as shader_file,
              open((args.shaders_to_compile_dir / ("ncnn." + file.name)), "w") as combined_file):
            for line in shader_file:
                # manually insert #include as glslangValidator does not support it
                if "#include" in line:
                    include_file = line.split()[1].strip("\"")
                    if not include_file in includes:
                        raise RuntimeError(f"Missing include file: {include_file}")

                    with open(ncnn_shader_dir / include_file, "r") as include:
                        combined_file.write(include.read())

                    continue

                combined_file.write(line)
                if line.startswith("#version"):
                    combined_file.write(preamble)


def convert_shaders():
    args = parse_args()

    if platform.system() == "Windows":
        xxd_path = args.build_dir / "_deps" / "kompute-src" / "external" / "bin" / "xxd.exe"
        if not xxd_path.exists():
            print("xxd needs to be manually compiled.")
            print(f"From a Visual Studio developer command prompt, in the {xxd_path.parent} directory, run:")
            print("cl.exe /DWIN32 .\\xxd.c")
            print("NOTE: manually defining WIN32 is required.")
            exit(-1)

    preprocess_ncnn_shaders(args)

    cmd = ["python", str(args.convert_script),
           "--shader-path", str(args.shaders_to_compile_dir),
           "--shader-binary", str(args.shader_compiler),
           "--header-path", str(args.output_dir),
           "--verbose"
           ]

    try:
        print(f"Running conversion script: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running shader conversion script: {e}")
        print("To see the actual error, run the shader binary with `-V <shader_file> -o <output_file>` manually.")
        sys.exit(-1)


if __name__ == "__main__":
    convert_shaders()
