import os
import sys

cmd = '/workspace/onnxruntime_training_bert {}'.format(' '.join(sys.argv[1:]))
print(cmd)
os.system(cmd)
