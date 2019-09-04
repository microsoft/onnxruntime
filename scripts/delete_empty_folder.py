import sys
import shutil
import glob
import os

folder = sys.argv[1]
for d in os.listdir(folder):
    if not os.path.isdir(d):
        continue;
    models = glob.glob(os.path.join(d,'*.onnx'))
    if len(models) == 0:
        print(d)
        shutil.rmtree(d)
    