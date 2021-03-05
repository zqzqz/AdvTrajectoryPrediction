import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset import ApolloscapeDataset
from prediction.visualize import draw_traces
import numpy as np

dataset = ApolloscapeDataset(6, 6)
data = np.genfromtxt("traj.txt", delimiter=" ")
input_data = dataset.format_data(data)
print(input_data)
draw_traces(input_data[0], "test.png")
