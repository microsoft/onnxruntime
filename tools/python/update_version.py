import os

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
                if len(sections) == 6 and sections[1].strip()[0].isdigit() :
                    current_version = sections[1].strip()
                    break
    if version != current_version:
        with open(file_path, 'w') as f:
            for i,line in enumerate(lines):
                f.write(line)
                if line.startswith('|--'):
                    sections = lines[i+1].split('|')
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
        with open(file_path,'w') as f:
            for line in lines:
                sections = line.strip().split('.')
                if inserted == False and len(sections) == 3 and sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
                    f.write(version + '\n')
                    f.write('^^^^^\n\n')
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

if __name__ == "__main__":
    update_version()
