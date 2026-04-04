import sys
import os

# Prevent loading the unbuilt source 'onnxruntime' folder in the repo root
# Since we are now in the "测试目录" subdirectory, repo_root is one level up
repo_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path = [p for p in sys.path if p != '' and os.path.abspath(p) != repo_root]

import onnxruntime as ort

def test_load_error():
    print("Attempting to load missing dependency DLL cascade...")
    
    # Needs absolute path for loading local dir DLL as custom ops library
    dll_path = os.path.abspath("test_main.dll")
    
    if not os.path.exists(dll_path):
        print(f"File not found: {dll_path}. Please run the batch script first to build it.")
        return 1

    try:
        # This will internally invoke LoadLibrary.
        # Since 'missing_dep.dll' is missing, it will fail, and ORT will use DetermineLoadLibraryError
        # to parse the import table of test_main.dll to find 'missing_dep.dll'.
        # The fix addresses the narrow-to-wide string conversion of 'missing_dep.dll'.
        ort.SessionOptions().register_custom_ops_library(dll_path)
    except Exception as e:
        error_msg = str(e)
        print("\n====== Caught expected error ======")
        print(error_msg)
        print("===================================\n")
        
        # We verify that 'missing_dep.dll', which is embedded in the DLL import table as ANSI,
        # was properly converted and correctly appears in the wide-char error cascade message.
        if "missing_dep.dll" in error_msg:
            print("✅ SUCCESS: The fix works! 'missing_dep.dll' is correctly identified and reported in the error message.")
            return 0
        else:
            print("❌ FAILED: The missing dependent DLL was not found in the error message.")
            return 1
    
    print("\n❌ FAILED: DLL loaded successfully, which shouldn't happen because \"missing_dep.dll\" is missing.")
    return 1

if __name__ == "__main__":
    sys.exit(test_load_error())