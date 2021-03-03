import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from TrafficPredict import TrafficPredictInterface
from visualize import *

if __name__ == "__main__":
    api = TrafficPredictInterface()
    for input_data in api.data():
        output_data = api.run(input_data)
        draw_traces(output_data, "test.png")
        break