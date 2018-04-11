import sys
import os
from ROOT import TFile
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np

path = str(sys.argv[1])
root_file = TFile(path)
tree = root_file.jetAnalyser

label = []
pt = [] 

for entry in tree:
    label.append(int(entry.label))
    pt.append(float(entry.pt))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.plot(label, label="label")
ax2.scatter(np.arange(len(pt)), pt, label="pt")

name = os.path.split(path)[-1]
name = os.path.splitext(name)[0]
plot_path = os.path.join(".", name+".png")
fig.savefig(plot_path)


