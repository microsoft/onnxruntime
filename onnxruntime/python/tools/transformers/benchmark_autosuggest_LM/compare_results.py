import os
import sys

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        counter = 0
        for x, y in zip(f1, f2):
            x = x.split("\t")
            y = y.split("\t")
            check_columns = [1, 4]
            for column_no in check_columns:
                x_temp = x[column_no].replace('"', '').strip()
                y_temp = y[column_no].replace('"', '').strip()
                if x_temp != y_temp:
                    counter += 1
                    print(x_temp + " is different from," + y_temp)
                    break

        print("Total different: " + str(counter))

def compare(type: str):
    file1 = "10KPrefixes_RandomSet_WithRepeat_set1_result_" + type + "_top_8.tsv"
    file2 = "10KPrefixes_RandomSet_WithRepeat_set1_result_" + type + "_original_top_8.tsv"
    compare_files(file1, file2)

if __name__  ==  "__main__":
    if len(sys.argv) < 2:
        print("Usage python compare_results.py onnx/dlis")
        sys.exit(1)
    compare(sys.argv[1])