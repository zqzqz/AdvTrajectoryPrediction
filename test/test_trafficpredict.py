import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.TrafficPredict import TrafficPredictInterface
from prediction.visualize import draw_traces

if __name__ == "__main__":
    api = TrafficPredictInterface("apolloscape", 4, 6)
    for input_data in api.data():
        output_data = api.run(input_data)
        draw_traces(output_data, "test.png")
        break