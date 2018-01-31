import os
from keras.models import load_model
import pandas as pd
import sys
sys.path.append("..")
from keras4jet.utils import get_size_of_model

def parse_dname(dname):
    print(dname)
    parsed = dname.split("_")
    print(parsed)
    channels = parsed[0]
    kernel_size = parsed[1].split("-")[0][1:]
    kernel_size = int(kernel_size)
    return [channels, kernel_size]

if __name__ == "__main__":
    logs = "./logs"
    entries = os.listdir(logs)
    model_paths = map(lambda each: os.path.join(logs, each, "saved_models", "model_final.h5"), entries)
    setups = map(parse_dname, entries)

    csv = []
    for p, s in zip(model_paths, setups):
        model = load_model(p)
        num_params = get_size_of_model(model)
        csv.append(s + [num_params])

#    csv = np.array(csv)
    df = pd.DataFrame(csv)
    df.to_csv("./num_params.csv", header=["channels", "kernel", "params"])
        
