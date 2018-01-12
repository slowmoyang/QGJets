import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from evaluation import evaluate_all

entries = os.listdir("./logs/")
entries = map(lambda each: os.path.join("./logs", each), entries)
for each in entries:
    evaluate_all(each)

