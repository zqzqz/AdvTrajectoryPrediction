import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from TrafficPredict import TrafficPredictInterface

if __name__ == "__main__":
    api = TrafficPredictInterface()
    for input_data in api.data():
        print(api.run(input_data))
        break