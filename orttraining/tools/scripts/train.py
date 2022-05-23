import os
import sys

cmd = f"/workspace/onnxruntime_training_bert {' '.join(sys.argv[1:])}"
print(cmd)
os.system(cmd)
