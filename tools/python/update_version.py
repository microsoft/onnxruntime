import os
import json


def update_version():
    version = ''
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, '..', '..', 'VERSION_NUMBER')) as f:
        version = f.readline().strip()
    lines = []
    current_version = ''
    file_path = os.path.join(cwd, '..', '..', 'docs', 'Versioning.md')
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('|'):
                sections = line.split('|')
                if len(sections) == 8 and sections[1].strip()[0].isdigit():
                    current_version = sections[1].strip()
                    break
    print('Current version of ORT seems to be: ' + current_version)
    if version != current_version:
        with open(file_path, 'w') as f:
            for i, line in enumerate(lines):
                f.write(line)
                if line.startswith('|--'):
                    sections = lines[i+1].split('|')
                    # Make sure there are no 'False Positive' version additions
                    # by making sure the line we are building a new line from
                    # contains the current_version
                    if len(sections) > 1 and sections[1].strip() == current_version:
                        sections[1] = ' ' + version + ' '
                        new_line = '|'.join(sections)
                        f.write(new_line)
    lines = []
    current_version = ''
    file_path = os.path.join(cwd, '..', '..', 'docs', 'python', 'README.rst')
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            sections = line.strip().split('.')
            if len(sections) == 3 and sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
                current_version = line.strip()
                break
    if version != current_version:
        inserted = False
        with open(file_path, 'w') as f:
            for line in lines:
                sections = line.strip().split('.')
                if inserted is False and len(sections) == 3 and \
                        sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
                    f.write(version + '\n')
                    f.write('^^^^^\n\n')
                    f.write('Release Notes : https://github.com/Microsoft/onnxruntime/releases/tag/v'
                            + version.strip() + '\n\n')
                    inserted = True
                f.write(line)
    lines = []
    current_version = ''
    file_path = os.path.join(cwd, '..', '..', 'package', 'rpm', 'onnxruntime.spec')
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Version:'):
                current_version = line.split(':')[1].strip()
                break
    if version != current_version:
        with open(file_path, 'w') as f:
            for line in lines:
                if line.startswith('Version:'):
                    f.write('Version:        ' + version + '\n')
                    continue
                f.write(line)
    lines = []
    current_version = ''
    file_path = os.path.join(cwd, '..', '..', 'onnxruntime', '__init__.py')
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('__version__'):
                current_version = line.split('=')[1].strip()[1:-1]
                break
    if version != current_version:
        with open(file_path, 'w') as f:
            for line in lines:
                if line.startswith('__version__'):
                    f.write('__version__ = "' + version + '"\n')
                    continue
                f.write(line)

    # update version for NPM packages
    current_version = ''
    js_root = os.path.join(cwd, '..', '..', 'js')
    file_paths = [
        os.path.join(js_root, 'common', 'package.json'),
        os.path.join(js_root, 'common', 'package-lock.json'),
        os.path.join(js_root, 'node', 'package.json'),
        os.path.join(js_root, 'node', 'package-lock.json'),
        os.path.join(js_root, 'web', 'package.json'),
        os.path.join(js_root, 'web', 'package-lock.json'),
        os.path.join(js_root, 'react_native', 'package.json'),
        os.path.join(js_root, 'react_native', 'package-lock.json'),
    ]
    for file_path in file_paths:
        with open(file_path) as f:
            content = json.load(f)
            current_version = content['version']
        if version != current_version:
            content['version'] = version
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)


if __name__ == "__main__":
    update_version()
