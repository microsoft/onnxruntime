import os
import sys
import os.path
from pathlib import Path

import xml.etree.cElementTree as ET
dir = sys.argv[1]

rootdir = Path(dir)
file_list = [f for f in rootdir.glob('**/*.onnx') if f.is_file()]
test_index = 0
for modelfile in file_list:
    test_folder = modelfile.parent
    relative_dir =  test_folder.relative_to(rootdir)
    #TODO: replace all '-' and '.' to '_'
    test_name = test_folder.name.replace('-','_').replace('.','_')
    print("TEST_METHOD(%s){" % test_name)
    print("  SessionFactory sf(LotusIR::kCpuExecutionProvider, true, true);")
    print("  run(sf,L\"%s\");" % str(relative_dir).replace('\\','\\\\'))
    print("}")    
    test_index+=1