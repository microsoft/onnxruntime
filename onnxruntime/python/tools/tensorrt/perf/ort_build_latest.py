import os
import subprocess

def main():
    ort_master_path = "/home/hcsuser/repos/onnxruntime/"
    pwd = os.getcwd()
    os.chdir(ort_master_path)

    p1 = subprocess.Popen(["git", "pull"])
    p1.wait()

    p1 = subprocess.Popen(["git", "log"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("utf-8").strip().split('\n')
    print(stdout[0])
    p1.wait()

    os.environ["PATH"] = "/home/hcsuser/repos/cmake-3.17.4-Linux-x86_64/bin/:" + os.environ["PATH"]
    os.environ["CUDACXX"] = "/usr/local/cuda-11.0/bin/nvcc"

    p1 = subprocess.Popen(["./build.sh", "--config", "Release", "--use_tensorrt", "--tensorrt_home", "/home/hcsuser/tensorrt/TensorRT-7.1.3.4", "--cuda_home", "/usr/local/cuda-11.0/", "--cudnn", "/usr/lib/x86_64-linux-gnu", "--build_wheel", "--skip_tests"])
    p1.wait()

    ort_wheel_path = "/home/hcsuser/repos/onnxruntime/build/Linux/Release/dist/"
    p1 = subprocess.Popen(["find", ort_wheel_path, "-name", "*.whl"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("utf-8").strip()
    ort_wheel = stdout.split("\n")
    p1 = subprocess.Popen(["pip3", "install", ort_wheel[0]])
    p1.wait()

    os.chdir(pwd)

if __name__ == "__main__":
    main()
