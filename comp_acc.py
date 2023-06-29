import csv
import sys
import numpy as np
method_type = sys.argv[1]

suffixes = [".dev.ctx_eval_long.csv", ".dev.ctx_eval_short.csv", ".train.ctx_eval_long.csv"]

total_cnt = 0
total_correct = 0
for suffix in suffixes:
    fname = f"./output/{method_type}" + suffix
    this_cnt = 0
    this_correct = 0
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_cnt += 1
            this_cnt += 1
            if 'map-answer' in row:
                if row['map-answer'] == row['gold']:
                    total_correct += 1
                    this_correct += 1
            else:
                if row['answer'] == row['gold']:
                    total_correct += 1
                    this_correct += 1

    print(f"File: {fname}, accuracy: {np.round(this_correct / this_cnt, 3)*100}")
print(f"Total accuracy: {np.round(total_correct / total_cnt, 3)*100}")