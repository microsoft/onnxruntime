# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import sys
import subprocess

selfdir = os.path.dirname(os.path.realpath(__file__))

for testpath in glob.glob(os.path.join(selfdir, '*')):
  if not os.path.basename(testpath).startswith('_') and \
     not (sys.platform.startswith("win") and os.path.basename(testpath).startswith('linux_only_')):
    print(f'Running tests for {testpath} ...')
    subprocess.check_call([sys.executable, testpath])
    print()