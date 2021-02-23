import os
import subprocess
import argparse
import tarfile
from perf_utils import get_latest_commit_hash

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--ort_master_path", required=True, help="ORT master repo")
    parser.add_argument("-t", "--tensorrt_home", required=True, help="TensorRT home directory")
    parser.add_argument("-c", "--cuda_home", required=True, help="CUDA home directory")
    parser.add_argument("-v", "--commit_hash", required = False, help="Github commit to test perf off of")
    parser.add_argument("-s", "--save", required=False, help="Directory to archive wheel file")
    parser.add_argument("-a", "--use_archived", required=False, help="Archived wheel file")
    args = parser.parse_args()
    return args

def archive_wheel_file(save_path, ort_wheel_file):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    p1 = subprocess.Popen(["cp", ort_wheel_file, save_path])
    p1.wait()

def install_new_ort_wheel(ort_master_path):
    ort_wheel_path = os.path.join(ort_master_path, "build", "Linux", "Release", "dist") 
    p1 = subprocess.Popen(["find", ort_wheel_path, "-name", "*.whl"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("utf-8").strip()
    ort_wheel = stdout.split("\n")[0]
    p1 = subprocess.Popen(["pip3", "install", "-I", ort_wheel])
    p1.wait()
    return ort_wheel

def main():
    args = parse_arguments()

    cmake_tar = "cmake-3.17.4-Linux-x86_64.tar.gz" 
    if not os.path.exists(cmake_tar):
        p = subprocess.run(["wget", "-c", "https://cmake.org/files/v3.17/" + cmake_tar], check=True)
    tar = tarfile.open(cmake_tar)
    tar.extractall()
    tar.close()
    
    os.environ["PATH"] = os.path.join(os.path.abspath("cmake-3.17.4-Linux-x86_64"), "bin") + ":" + os.environ["PATH"]
    os.environ["CUDACXX"] = os.path.join(args.cuda_home, "bin", "nvcc") 

    ort_master_path = args.ort_master_path 
    pwd = os.getcwd()
    os.chdir(ort_master_path)

    if args.use_archived:
        ort_wheel_file = args.use_archived
        p1 = subprocess.Popen(["pip3", "install", "-I", ort_wheel_file])
        p1.wait()
    
    else:
        if args.commit_hash:
            commit = args.commit_hash    
            p1 = subprocess.Popen(["git", "checkout", commit])
        else: 
            commit = get_latest_commit_hash()
            p1 = subprocess.Popen(["git", "pull", "origin", "master"])
        p1.wait()

        p1 = subprocess.Popen(["./build.sh", "--config", "Release", "--use_tensorrt", "--tensorrt_home", args.tensorrt_home, "--cuda_home", args.cuda_home, "--cudnn", "/usr/lib/x86_64-linux-gnu", "--build_wheel", "--skip_tests", "--parallel"])
        p1.wait()

        ort_wheel_file = install_new_ort_wheel(ort_master_path)
    
        if args.save:
            save_path = os.path.join(args.save, commit)
            archive_wheel_file(save_path, ort_wheel_file)

    os.chdir(pwd)

if __name__ == "__main__":
    main()
