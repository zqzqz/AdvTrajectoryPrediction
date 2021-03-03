from TrafficPredict import TrafficPredictInterface
from common.eval_util import ade, fde

def evaluate_error(AlgorithmInterface):
    api = AlgorithmInterface
    for input_data in api.data():
        output_data = api.run(input_data)
        for _, obj in output_data["objects"].items():
            pass