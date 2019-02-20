This folder is copied from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist.

Command to train the model
> python fully_connected_feed.py --input_data_dir data --log_dir log --max_steps 5000 --learning_rate 0.02

It should reach an precision around 88% at the end of training.


Output Sample:

Step 0: loss = 0.12 (0.142 sec)
Step 100: loss = 0.09 (0.001 sec)
Step 200: loss = 0.08 (0.001 sec)
Step 300: loss = 0.07 (0.001 sec)
Step 400: loss = 0.07 (0.001 sec)
Step 500: loss = 0.07 (0.001 sec)
Step 600: loss = 0.06 (0.001 sec)
Step 700: loss = 0.06 (0.001 sec)
Step 800: loss = 0.05 (0.001 sec)
Step 900: loss = 0.05 (0.002 sec)
Training Data Eval:
Num examples: 55000  Num correct: 41463  Precision @ 1: 0.7539
Validation Data Eval:
Num examples: 5000  Num correct: 3809  Precision @ 1: 0.7618
Test Data Eval:
Num examples: 10000  Num correct: 7620  Precision @ 1: 0.7620
Step 1000: loss = 0.05 (0.023 sec)
Step 1100: loss = 0.05 (0.105 sec)
Step 1200: loss = 0.05 (0.001 sec)
Step 1300: loss = 0.05 (0.001 sec)
Step 1400: loss = 0.05 (0.002 sec)
Step 1500: loss = 0.05 (0.002 sec)
Step 1600: loss = 0.05 (0.001 sec)
Step 1700: loss = 0.04 (0.001 sec)
Step 1800: loss = 0.04 (0.001 sec)
Step 1900: loss = 0.04 (0.000 sec)
Training Data Eval:
Num examples: 55000  Num correct: 46050  Precision @ 1: 0.8373
Validation Data Eval:
Num examples: 5000  Num correct: 4209  Precision @ 1: 0.8418
Test Data Eval:
Num examples: 10000  Num correct: 8439  Precision @ 1: 0.8439
Step 2000: loss = 0.04 (0.019 sec)
Step 2100: loss = 0.04 (0.001 sec)
Step 2200: loss = 0.04 (0.105 sec)
Step 2300: loss = 0.04 (0.001 sec)
Step 2400: loss = 0.04 (0.001 sec)
Step 2500: loss = 0.04 (0.001 sec)
Step 2600: loss = 0.03 (0.001 sec)
Step 2700: loss = 0.04 (0.001 sec)
Step 2800: loss = 0.04 (0.001 sec)
Step 2900: loss = 0.03 (0.001 sec)
Training Data Eval:
Num examples: 55000  Num correct: 47552  Precision @ 1: 0.8646
Validation Data Eval:
Num examples: 5000  Num correct: 4376  Precision @ 1: 0.8752
Test Data Eval:
Num examples: 10000  Num correct: 8717  Precision @ 1: 0.8717
Step 3000: loss = 0.04 (0.022 sec)
Step 3100: loss = 0.04 (0.001 sec)
Step 3200: loss = 0.03 (0.001 sec)
Step 3300: loss = 0.03 (0.108 sec)
Step 3400: loss = 0.03 (0.001 sec)
Step 3500: loss = 0.03 (0.001 sec)
Step 3600: loss = 0.03 (0.001 sec)
Step 3700: loss = 0.03 (0.001 sec)
Step 3800: loss = 0.03 (0.001 sec)
Step 3900: loss = 0.03 (0.001 sec)
Training Data Eval:
Num examples: 55000  Num correct: 48306  Precision @ 1: 0.8783
Validation Data Eval:
Num examples: 5000  Num correct: 4441  Precision @ 1: 0.8882
Test Data Eval:
Num examples: 10000  Num correct: 8871  Precision @ 1: 0.8871
Step 4000: loss = 0.03 (0.018 sec)
Step 4100: loss = 0.03 (0.002 sec)
Step 4200: loss = 0.03 (0.001 sec)
Step 4300: loss = 0.03 (0.002 sec)
Step 4400: loss = 0.03 (0.100 sec)
Step 4500: loss = 0.03 (0.001 sec)
Step 4600: loss = 0.03 (0.001 sec)
Step 4700: loss = 0.03 (0.002 sec)
Step 4800: loss = 0.03 (0.001 sec)
Step 4900: loss = 0.03 (0.001 sec)
Training Data Eval:
Num examples: 55000  Num correct: 48885  Precision @ 1: 0.8888
Validation Data Eval:
Num examples: 5000  Num correct: 4492  Precision @ 1: 0.8984
Test Data Eval:
Num examples: 10000  Num correct: 8970  Precision @ 1: 0.8970