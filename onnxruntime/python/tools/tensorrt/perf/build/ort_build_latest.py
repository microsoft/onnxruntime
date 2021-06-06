import os
import subprocess
import argparse
import tarfile

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--ort_master_path", required=True, help="ORT master repo")
    parser.add_argument("-t", "--tensorrt_home", required=True, help="TensorRT home directory")
    parser.add_argument("-c", "--cuda_home", required=True, help="CUDA home directory")
    parser.add_argument("-b", "--branch", required=False, default="master", help="Github branch to test perf off of")
    parser.add_argument("-s", "--save", required=False, help="Directory to archive wheel file")
    parser.add_argument("-a", "--use_archived", required=False, help="Archived wheel file")
    args = parser.parse_args()
    return args

def archive_wheel_file(save_path, ort_wheel_file):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    subprocess.run(["cp", ort_wheel_file, save_path], check=True)

def install_new_ort_wheel(ort_master_path):
    ort_wheel_path = os.path.join(ort_master_path, "build", "Linux", "Release", "dist") 
    p1 = subprocess.run(["find", ort_wheel_path, "-name", "*.whl"], stdout=subprocess.PIPE, check=True)
    stdout = p1.stdout.decode("utf-8").strip()
    ort_wheel = stdout.split("\n")[0]
    subprocess.run(["pip3", "install", "--force-reinstall", ort_wheel], check=True)
    return ort_wheel

def main():
    args = parse_arguments()

    cmake_tar = "cmake-3.18.4-Linux-x86_64.tar.gz" 
    if not os.path.exists(cmake_tar):
        p = subprocess.run(["sudo", "wget", "-c", "https://cmake.org/files/v3.18/" + cmake_tar], check=True)
    tar = tarfile.open(cmake_tar)
    tar.extractall()
    tar.close()
    
    os.environ["PATH"] = os.path.join(os.path.abspath("cmake-3.18.4-Linux-x86_64"), "bin") + ":" + os.environ["PATH"]
    os.environ["CUDACXX"] = os.path.join(args.cuda_home, "bin", "nvcc") 

    ort_master_path = args.ort_master_path 
    pwd = os.getcwd()
    os.chdir(ort_master_path)

    if args.use_archived:
        ort_wheel_file = args.use_archived
        subprocess.run(["pip3", "install", "--force-reinstall", ort_wheel_file], check=True)
    
    else:
        subprocess.run(["git", "fetch"], check=True)
        subprocess.run(["git", "checkout", args.branch], check=True)
        subprocess.run(["git", "pull", "origin", args.branch], check=True)
        subprocess.run(["./build.sh", "--config", "Release", "--use_tensorrt", "--tensorrt_home", args.tensorrt_home, "--cuda_home", args.cuda_home, "--cudnn", "/usr/lib/x86_64-linux-gnu", "--build_wheel", "--skip_tests", "--parallel"], check=True)

        ort_wheel_file = install_new_ort_wheel(ort_master_path)
    
        if args.save:
            archive_wheel_file(args.save, ort_wheel_file)

    os.chdir(pwd)

if __name__ == "__main__":
    main()
