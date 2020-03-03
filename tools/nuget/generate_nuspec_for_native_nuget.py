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
    parser.add_argument("--build_config", required=True, help="Eg: RelWithDebInfo")
    parser.add_argument("--ort_build_path", required=True, help="ORT build directory.")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--packages_path", required=True, help="Nuget packages output directory.")
    parser.add_argument("--sources_path", required=True, help="OnnxRuntime source code root.")
    parser.add_argument("--commit_id", required=True, help="The last commit id included in this package.")
    parser.add_argument("--is_release_build", required=False, default=None, type=str, help="Flag indicating if the build is a release build. Accepted values: true/false.")        

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
    list.append('<dependencies>')

    # Support .Net Core
    list.append('<group targetFramework="NETCOREAPP">')
    list.append('<dependency id="Microsoft.ML.OnnxRuntime.Managed"' + ' version="' + version + '"/>')
    list.append('</group>')

    # Support .Net Standard
    list.append('<group targetFramework="NETSTANDARD">')
    list.append('<dependency id="Microsoft.ML.OnnxRuntime.Managed"' + ' version="' + version + '"/>')
    list.append('</group>')

    # Support .Net Framework
    list.append('<group targetFramework="NETFRAMEWORK">')
    list.append('<dependency id="Microsoft.ML.OnnxRuntime.Managed"' + ' version="' + version + '"/>')
    list.append('</group>')

    list.append('</dependencies>')

def get_env_var(key):
    return os.environ.get(key)

def generate_release_notes(list):
    list.append('<releaseNotes>')    
    list.append('Release Def:')
    
    branch = get_env_var('BUILD_SOURCEBRANCH')
    list.append('\t' + 'Branch: ' + (branch if branch != None else ''))
    
    version = get_env_var('BUILD_SOURCEVERSION')
    list.append('\t' + 'Commit: ' + (version if version != None else ''))

    build_id = get_env_var('BUILD_BUILDID')
    list.append('\t' + 'Build: https://aiinfra.visualstudio.com/Lotus/_build/results?buildId=' + (build_id if build_id != None else ''))

    list.append('</releaseNotes>')

def generate_metadata(list, args):
    metadata_list = ['<metadata>']
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, 'Microsoft')
    generate_owners(metadata_list, 'Microsoft')
    generate_description(metadata_list, 'This package contains native shared library artifacts for all supported platforms of ONNX Runtime.')
    generate_copyright(metadata_list, '\xc2\xa9 ' + 'Microsoft Corporation. All rights reserved.')
    generate_tags(metadata_list, 'ONNX ONNX Runtime Machine Learning')
    generate_icon_url(metadata_list, 'https://go.microsoft.com/fwlink/?linkid=2049168')
    generate_license(metadata_list)
    generate_project_url(metadata_list, 'https://github.com/Microsoft/onnxruntime')
    generate_repo_url(metadata_list, 'https://github.com/Microsoft/onnxruntime.git', args.commit_id)  
    generate_dependencies(metadata_list, args.package_version)
    generate_release_notes(metadata_list)
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
            files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'DirectML.pdb') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
            files_list.append('<file src=' + '"' + os.path.join(args.packages_path, 'DirectML.0.0.1\\LICENSE.txt') + '" target="DirectML_LICENSE.txt" />')
        
    # Process onnxruntime import lib, dll, and pdb
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.lib') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime.pdb') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
    
    # Process Windows.AI.MachineLearning lib, dll, and pdb
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.native_build_path, 'windows.ai.machinelearning.lib')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'windows.ai.machinelearning.lib') + '" target="runtimes\\win-' + args.target_architecture + '\\native\\Windows.AI.MachineLearning.lib" />')

    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.native_build_path, 'windows.ai.machinelearning.dll')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'windows.ai.machinelearning.dll') + '" target="runtimes\\win-' + args.target_architecture + '\\native\\Windows.AI.MachineLearning.dll" />')
        
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.native_build_path, 'windows.ai.machinelearning.pdb')):        
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'windows.ai.machinelearning.pdb') + '" target="runtimes\\win-' + args.target_architecture + '\\native\\Windows.AI.MachineLearning.pdb" />')

    # Process windows.ai.machinelearning.winmd
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.winmd')):
        files_list.append('<file src=' + '"' + os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.winmd') + '" target="build\\native\\metadata\\Windows.AI.MachineLearning.winmd" />')

    # Process windows.ai.machinelearning headers
    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.h')):
        files_list.append('<file src=' + '"' + os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.h') + '" target="build\\native\\include\\Windows.AI.MachineLearning.h" />')

    if (args.package_name == 'Microsoft.ML.OnnxRuntime.DirectML' or args.package_name == 'Microsoft.ML.OnnxRuntime') and os.path.exists(os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.native.h')):
        files_list.append('<file src=' + '"' + os.path.join(args.ort_build_path, args.build_config, 'windows.ai.machinelearning.native.h') + '" target="build\\native\\include\\Windows.AI.MachineLearning.Native.h" />')

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
    files_list.append('<file src=' + '"' + os.path.join(args.sources_path, 'docs', 'C_API.md') + '" target="README.md" />')
    
    # Process props file
    source_props = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', 'props.xml')
    target_props = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', args.package_name + '.props')    
    os.system('copy ' + source_props + ' ' + target_props)
    files_list.append('<file src=' + '"' + target_props + '" target="build\\native" />')
    files_list.append('<file src=' + '"' + target_props + '" target="build\\netstandard1.1" />')

    # Process targets file
    source_targets = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', 'targets.xml')
    target_targets = os.path.join(args.sources_path, 'csharp', 'src', 'Microsoft.ML.OnnxRuntime', args.package_name + '.targets')    
    os.system('copy ' + source_targets + ' ' + target_targets)
    files_list.append('<file src=' + '"' + target_targets + '" target="build\\native" />')   
    files_list.append('<file src=' + '"' + target_targets + '" target="build\\netstandard1.1" />')
    
    # Some tools to be packaged in nightly build only, should not be released 
    # These are copied to the runtimes folder for convenience of loading with the dlls    
    if args.is_release_build.lower() != 'true' and args.target_architecture == 'x64' and os.path.exists(os.path.join(args.native_build_path, 'onnxruntime_perf_test.exe')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnxruntime_perf_test.exe') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')

    if args.is_release_build.lower() != 'true' and args.target_architecture == 'x64' and os.path.exists(os.path.join(args.native_build_path, 'onnx_test_runner.exe')):
        files_list.append('<file src=' + '"' + os.path.join(args.native_build_path, 'onnx_test_runner.exe') + '" target="runtimes\\win-' + args.target_architecture + '\\native" />')
        
    files_list.append('</files>')
    
    list += files_list

def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append('<package>')
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append('</package>')
    return lines

def is_windows():
    return sys.platform.startswith("win")
    
def main():
    if not is_windows():
        raise Exception('Native Nuget generation is currently supported only on Windows')
    
    # Parse arguments
    args = parse_arguments()
    if (args.is_release_build.lower() != 'true' and args.is_release_build.lower() != 'false'):
        raise Exception('Only valid options for IsReleaseBuild are: true and false')        
    
    # Generate nuspec 
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget       
    with open(os.path.join(args.native_build_path, 'NativeNuget.nuspec'), 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

if __name__ == "__main__":
    sys.exit(main())    