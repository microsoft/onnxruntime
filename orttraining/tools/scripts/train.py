import subprocess
import sys

cmd = ["/workspace/onnxruntime_training_bert", *sys.argv[1:]]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
