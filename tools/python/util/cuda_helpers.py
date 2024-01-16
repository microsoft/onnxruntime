#  // Copyright (c) Microsoft Corporation. All rights reserved.
#  // Licensed under the MIT License.
import os
from .BuildError import BuildError
from .platform_helpers import is_windows

def setup_cuda_vars(args):
    cuda_home = ""
    cudnn_home = ""

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else os.getenv("CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv("CUDNN_HOME")

        cuda_home_valid = cuda_home is not None and os.path.exists(cuda_home)
        cudnn_home_valid = cudnn_home is not None and os.path.exists(cudnn_home)

        if not cuda_home_valid or (not is_windows() and not cudnn_home_valid):
            raise BuildError(
                "cuda_home and cudnn_home paths must be specified and valid.",
                "cuda_home='{}' valid={}. cudnn_home='{}' valid={}".format(
                    cuda_home, cuda_home_valid, cudnn_home, cudnn_home_valid
                ),
            )

    return cuda_home, cudnn_home
