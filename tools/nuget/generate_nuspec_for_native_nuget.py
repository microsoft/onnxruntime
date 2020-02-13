# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX Runtime create nuget spec script (for hosting native shared library artifacts)",
                                     usage='')
    # Main arguments
    parser.add_argument("--package_name", required=True, help="ORT package name. Eg: Microsoft.ML.OnnxRuntime.Gpu")
    parser.add_argument("--package_version", required=True, help="ORT package version. Eg: 1.0.0")
    parser.add_argument("--target_architecture", required=True, help="Eg: x64")
    parser.add_argument("--native_build_path", required=True, help="Path where the generated nuspec is to be placed.")        
    parser.add_argument("--sources_path", required=True, help="Path where the generated nuspec is to be placed.")        
    parser.add_argument("--commit_id", required=True, help="The last commit id included in this package.")        

    return parser.parse_args()

def generate_id(list, package_name):
    list.append('<id>' + package_name + '</id>')
    
def generate_version(list, package_version):
    list.append('<version>' + package_version + '</version>')
    
def generate_authors(list, authors):
    list.append('<authors>' + authors + '</authors>')
    
def generate_owners(list, owners):
    list.append('<owners>' + owners + '</owners>')
    
def generate_description(list, description):
    list.append('<description>' + description + '</description>')
    
def generate_copyright(list, copyright):
    list.append('<copyright>' + copyright + '</copyright>')

def generate_tags(list, tags):
    list.append('<tags>' + tags + '</tags>')

def generate_icon_url(list, icon_url):
    list.append('<iconUrl>' + icon_url + '</iconUrl>')

def generate_license(list):
    list.append('<license type="file">LICENSE.txt</license>')
    
def generate_project_url(list, project_url):
    list.append('<projectUrl>' + project_url + '</projectUrl>')

def generate_repo_url(list, repo_url, commit_id):
    list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')
    
def generate_dependencies(list, version):
    return
    # list.append('<dependencies>' + '<dependency id="Microsoft.ML.OnnxRuntime.Managed"' + ' version="' + version + '"/> </dependencies>')
    
def generate_metadata(list, args):
    metadata_list = ['<metadata>']
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, 'Microsoft')
    generate_owners(metadata_list, 'Microsoft')
    generate_description(metadata_list, 'This package contains native shared library artifacts for all supported platforms of ONNX Runtime.')
    generate_copyright(metadata_list, '\xc2\xa9 ' + 'Microsoft Corporation. All rights reserved.')
    generate_tags(metadata_list, 'ONNX;ONNX Runtime;Machine Learning')
    generate_icon_url(metadata_list, 'https://go.microsoft.com/fwlink/?linkid=2049168')
    generate_license(metadata_list)
    generate_project_url(metadata_list, 'https://github.com/Microsoft/onnxruntime')
    generate_repo_url(metadata_list, 'https://github.com/Microsoft/onnxruntime.git', args.commit_id)  
    generate_dependencies(metadata_list, args.package_version)
    metadata_list.append('</metadata>')
    
    list += metadata_list

def generate_files(list, args):
    files_list = ['<files>']
    
    # Process headers
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'include\\onnxruntime\\core\\session\\onnxruntime_*.h') + '" target="build\\native\\include" />')
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'include\\onnxruntime\\core\\providers\\cpu\\cpu_provider_factory.h') + '" target="build\\native\\include" />')
    
    if (args.package_name != 'Microsoft.ML.OnnxRuntime.DirectML'):
        files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'include\\onnxruntime\\core\\providers\\cuda\\cuda_provider_factory.h') + '" target="build\\native\\include" />')
    else: # it is a DirectML package
        files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'include\\onnxruntime\\core\\providers\\dml\\dml_provider_factory.h') + '" target="build\\native\\include" />')
        # Process DirectML dll
        if os.path.exists(os.path.join(args.native_build_path, 'DirectML.dll')):
            files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'DirectML.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
        
        
    # Process onnxruntime import lib, dll, and pdb
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.lib') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.pdb') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    
    # Process windows.ai.machinelearning dll, and pdb
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.native_build_path, 'windows.ai.machinelearning.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'windows.ai.machinelearning.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
        
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.native_build_path, 'windows.ai.machinelearning.pdb')):        
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'windows.ai.machinelearning.pdb') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    
    # Process dnll.dll    
    if os.path.exists(os.path.join(args.native_build_path, 'dnnl.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'dnnl.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    
    # Process mklml.dll
    if os.path.exists(os.path.join(args.native_build_path, 'mklml.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'mklml.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')

    # Process libiomp5md.dll
    if os.path.exists(os.path.join(args.native_build_path, 'libiomp5md.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'libiomp5md.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')

    # Process tvm.dll
    if os.path.exists(os.path.join(args.native_build_path, 'tvm.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'tvm.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    
    # Process License, ThirdPartyNotices, Privacy, README
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'LICENSE.txt') + '" target="LICENSE.txt" />')    
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'ThirdPartyNotices.txt') + '" target="ThirdPartyNotices.txt" />')    
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'docs', 'Privacy.md') + '" target="Privacy.md" />')
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'docs', 'CSharp_API.md') + '" target="README.md" />')
    
    # Process props file
    source_props = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', 'props.xml')
    target_props = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', args.package_name + '.props')    
    os.system('copy ' + source_props + ' ' + target_props)
    files_list.append('<file src=' + '"' + target_props + '" target="build\\native" />')
        
    # Process targets file
    source_targets = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', 'targets.xml')
    target_targets = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', args.package_name + '.targets')    
    os.system('copy ' + source_targets + ' ' + target_targets)
    files_list.append('<file src=' + '"' + target_targets + '" target="build\\native" />')    
    
    files_list.append('</files>')
    
    list += files_list
        
def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append('<package>')
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append('</package>')
    return lines

def main():
    # Parse arguments
    args = parse_arguments()
    lines = generate_nuspec(args)
        
    with open('NativeNuget.nuspec', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

if __name__ == "__main__":
    sys.exit(main())    