import shlex
import subprocess
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    cmd = ["/workspace/onnxruntime_training_bert", *argv]
    print(shlex.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
