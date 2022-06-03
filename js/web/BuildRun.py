import os
import shutil
import argparse


HR_Model_Path_ONNX = ".\\test\\data\\ONNX\\"
HR_Model_Path_ORT = ".\\test\\data\\ORT\\"

models = {"HR_ONNX":HR_Model_Path_ONNX,"HR_ORT":HR_Model_Path_ORT}


Build_Path = "..\\..\\build\\Windows\\Release\\"
Dist_Path = ".\\dist\\"
Binding_Path = ".\\lib\\wasm\\binding\\"

# Needed Arguments
# -b - build WASM
#      - Need to add partial build for only WASM artifacts - reduces the amount of time to build ~100x
#      - Need to add full build for the entire WASM + Web App
#      - configuration - Release/Debug
# -r - run the test app
#      - Which model
#      - Single / Multi inference

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--build", choices=["Release","Debug"], help='build WASM - {Release,Debug}')
parser.add_argument("-p", "--partial", help='build WASM - {full,partial}', action="store_true")
parser.add_argument("-r", "--run", choices=["HR_ONNX","HR_ORT"],help='run test app')

args = parser.parse_args()

if(args.build):
    configuration = args.build
    #print(configuration)
    if(args.partial):
        ## Building WASM - Release - WASM_SIMD
        command = "cmake --build ..\\..\\build\Windows\Release\ --target onnxruntime_webassembly -- -j 6"
        #print(command)
        os.system(command)
    else:
        ## Building WASM - Release - WASM_SIMD
        ## --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit
        command = "..\\..\\build.bat --build_wasm --skip_tests --skip_submodule_sync --config "+configuration+" --wasm_malloc dlmalloc --enable_wasm_simd"
        #print(command)
        os.system(command)
        ## Building WASM - Release - WASM
        #command = "..\\..\\build.bat --build_wasm --skip_tests --skip_submodule_sync --config "+configuration+" --wasm_malloc dlmalloc --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit"
        ## print(command)
        #os.system(command)   
        

    ## Copying WASM artifacts - WASM-SIMD
    file_name = "ort-wasm-simd.wasm"
    shutil.copyfile(Build_Path+file_name,Dist_Path+file_name)
    ## Copying WASM artifacts - WASM
    #file_name = "ort-wasm.wasm"
    #shutil.copyfile(Build_Path+file_name,Dist_Path+file_name)

    file_name_source = "ort-wasm-simd.js"
    #test for Yulong - seem to not work...
    file_name = "ort-wasm-simd.js"
    shutil.copyfile(Build_Path+file_name_source,Binding_Path+file_name)  
    file_name = "ort-wasm.js"
    shutil.copyfile(Build_Path+file_name_source,Binding_Path+file_name)  

    #file_name = "ort-wasm.js"
    #shutil.copyfile(Build_Path+file_name,Binding_Path+file_name)

      

if(args.run):
    model_name = args.run
    model_path = models[model_name]
    #print(model_name)
    ## Running the tests
    command = "npm test -- model "+model_path+" -b=wasm --wasm-enable-simd=true --wasm-number-threads=1"
    os.system(command)

exit()



