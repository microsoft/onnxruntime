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

def compare_results(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        counter = 0
        line = 0
        for x, y in zip(f1, f2):
            #print(x + y)
            line += 1
            x = x.split("\t")
            y = y.split("\t")
            if x[4] != y[4]:
                counter += 1
                #print(str(line) + "     :       " + x[4] + "is diff from" + y[4])

        print("Total different: " + str(counter))


def compare(type: str, number:str):
    if type == "dlis":
        return
    
    if number == "10K":
        file1 = "10KPrefixes_RandomSet_WithRepeat_set1_post_fused_orig.tsv"
        file2 = "10KPrefixes_RandomSet_WithRepeat_set1_post_fused_now.tsv"
    elif number == "1K":
        file1 = "1KPrefixes_RandomSet_WithNoRepeat_onnx_post_fused_orig.tsv"
        file2 = "1KPrefixes_RandomSet_WithNoRepeat_onnx_post_fused_now.tsv"

    compare_results(file1, file2)

if __name__  ==  "__main__":
    if len(sys.argv) < 2:
        print("Usage python compare_results.py onnx/dlis")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])