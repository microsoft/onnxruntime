# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import textwrap


def rewrite_target_file(target):
    with open(target, 'a') as f:
        f.write(textwrap.dedent(
            """
            import warnings

            try:
                # This import is necessary in order to delegate the loading of libtvm.so to TVM.
                import tvm
            except ImportError as e:
                warnings.warn(
                    f"WARNING: Failed to import TVM, libtvm.so was not loaded. More details: {e}"
                )
            try:
                # Working between the C++ and Python parts in TVM EP is done using the PackedFunc and
                # Registry classes. In order to use a Python function in C++ code, it must be registered in
                # the global table of functions. Registration is carried out through the JIT interface,
                # so it is necessary to call special functions for registration.
                # To do this, we need to make the following import.
                import onnxruntime.providers.tvm
            except ImportError as e:
                warnings.warn(
                    f"WARNING: Failed to register python functions to work with TVM EP. More details: {e}"
                )
            """
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_file", type=str, required=True, help="Path to the file to be expanded.")
    args = parser.parse_args()
    rewrite_target_file(args.target_file)


if __name__ == '__main__':
    main()
