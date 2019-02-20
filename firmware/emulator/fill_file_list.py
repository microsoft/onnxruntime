import argparse
from os import listdir, environ
from os.path import isfile, join, dirname
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--firmware_name", type=str)
parser.add_argument("--template_prop_file", type=str)
parser.add_argument("--target_prop_file", type=str)
parser.add_argument("--target_folder", type=str)
args= parser.parse_args()

bond_file_list = "    <Firmware Include=\""+ args.firmware_name +"\" />\n"

def get_imported_bond(file_path, found_bonds):
    '''check all imported *.bond files in the specified file, and add all found bond file full path into found_bonds.'''
    with open(file_path, 'r') as myfile:
        content = myfile.read()
        imports = re.findall('import\s+"([a-zA-Z]+.bond)"', content)
        current_dir = dirname(file_path)
        for bond in imports:
            if bond in found_bonds:
                continue
            
            full_path = join(current_dir, bond)
            if isfile(full_path): # check file existence
                print("find bond file " + bond + " in folder " + current_dir)
                pass
            else:
                if bond in ["CommonFunctions.bond", "Lstm.bond"]:
                    full_path = environ['DevKit'] + "\\schemas\\" + bond
                    print("find bond file " + full_path)

            if full_path:
                found_bonds.append(full_path)
                # find the imported bond files recursively.
                get_imported_bond(full_path, found_bonds)
            else:
                print("not found " + bond + ", skip")

found_bonds = []
for file_name in listdir(args.target_folder):
    file_path = join(args.target_folder, file_name)
    if not isfile(file_path):
        continue

    print("found file ", file_path)
    if file_name.endswith(".bond"):
        found_bonds.append(args.firmware_name + "\\" + file_name)
        get_imported_bond(file_path, found_bonds)

found_bonds.reverse()
for bond in found_bonds:
    bond_file_list += "    <BondCodegen Include=\"" + bond + "\" />\n"

with open(args.template_prop_file, "r") as fin:
    with open(args.target_prop_file, "w") as fout:
        fout.write(fin.read().replace('PLACEHOLDER_FOR_BOND_FILES', bond_file_list))
        
print("Successfully generate .prop file " + args.target_prop_file)
