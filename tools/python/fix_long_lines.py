import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile


# look for long lines in the file, and if found run clang-format on just those lines
def _process_file(path, clang_exe, tmpdir):
    print(f'Processing {path}')

    bad_lines = []
    with open(path, 'r', encoding='UTF8') as f:
        line_num = 0
        for line in f:
            line_num += 1  # clang-format line numbers start at 1
            if len(line) > 120:
                bad_lines.append(line_num)

    if bad_lines:
        filename = os.path.basename(path)
        target = os.path.join(tmpdir, filename)
        shutil.copy(path, target)

        # run clang-format to update just the long lines in the file
        cmd = [clang_exe, '-i', ]
        for line in bad_lines:
            cmd.append(f'--lines={line}:{line}')
        cmd.append(target)
        subprocess.run(cmd, cwd=tmpdir, check=True)

        # copy updated file back to original location
        shutil.copy(target, path)


# Walk the path and all subdirectories, hunting files with long lines and correcting their errant ways.
def walker_the_long_line_ranger(path, clang_exe, tmpdir):
    extensions = ['.cc', '.h']
    for root, _, files in os.walk(path):
        for file in files:
            f = file.casefold()
            ext = os.path.splitext(f)[-1]
            if ext in extensions:
                _process_file(os.path.join(root, file), clang_exe, tmpdir)


def main():
    argparser = argparse.ArgumentParser(
        'Script to fix long lines in the source using clang-format. '
        'Only lines that exceed the 120 character maximum are altered in order to minimize the impact. '
        'Checks .cc and .h files under /include/onnxruntime and /onnxruntime/core.')

    argparser.add_argument('--clang-format', type=pathlib.Path, required=True, help='Path to clang-format executable')
    args = argparser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    ort_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    with tempfile.TemporaryDirectory() as tmpdir:
        # create config in tmpdir
        # config is based on .clang-format in the repository root directory, with the ColumnLimit set to 120
        with open(os.path.join(tmpdir, '.clang-format'), 'w') as f:
            f.write('''
            BasedOnStyle: Google
            ColumnLimit: 120
            DerivePointerAlignment: false
            ''')

        clang_format = str(args.clang_format.resolve())

        include_path = os.path.join(ort_root, 'include', 'onnxruntime')
        walker_the_long_line_ranger(include_path, clang_format, tmpdir)

        # src_path = os.path.join(ort_root, 'onnxruntime', 'core')
        # walker_the_long_line_ranger(src_path, clang_format, tmpdir)


if __name__ == "__main__":
    main()
