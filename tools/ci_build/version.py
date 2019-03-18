version = ''
with open('../../VERSION_NUMBER') as f:
    version = f.readline().strip()
lines = []
current_version = ''
with open('../../docs/Versioning.md') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('|'):
            sections = line.split('|')
            if len(sections) == 6 and sections[1].strip()[0].isdigit() :
               current_version = sections[1].strip()
               break
if version != current_version:
    with open('../../docs/Versioning.md','w') as f:
        for i,line in enumerate(lines):
            f.write(line)
            if line.startswith('|--'):
                sections = lines[i+1].split('|')
                sections[1] = ' ' + version + ' '
                new_line = '|'.join(sections)
                f.write(new_line)
        f.close()
lines = []
current_version = ''
with open('../../docs/python/README.rst') as f:
    lines = f.readlines()
    for line in lines:
        sections = line.strip().split('.')
        if len(sections) == 3 and sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
            current_version = line.strip()
            break
if version != current_version:
    inserted = False
    with open('../../docs/python/README.rst','w') as f:
        for line in lines:
            sections = line.strip().split('.')
            if inserted == False and len(sections) == 3 and sections[0].isdigit() and sections[1].isdigit() and sections[2].isdigit():
                f.write(version + '\n')
                f.write('^^^^^\n\n')
                inserted = True
            f.write(line)
        f.close()
lines = []
current_version = ''
with open('../../package/rpm/onnxruntime.spec') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Version:'):
            current_version = line.split(':')[1].strip()
            break
if version != current_version:
    with open('../../package/rpm/onnxruntime.spec','w') as f:
        for line in lines:
            if line.startswith('Version:'):
                f.write('Version:        ' + version + '\n')
                continue
            f.write(line)
        f.close()
print (current_version) 
print (version)
