from .base import Evaluator
from .utils import ade, fde

import numpy as np


class SingleFrameEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.metric_map = {
            "ade": self.ade,
            "fde": self.fde
        }

    def get_obj_ids(self, data):
        obj_ids = []

        if "obj_id" in data:
            obj_ids.append(data["obj_id"])
        else:
            for obj_id, obj in data["objects"].items():
                if obj["complete"] and np.min(obj["predict_trace"]) > 0:
                    obj_ids.append(obj_id)
        
        return obj_ids

    def ade(self, data):
        result = []

        obj_ids = self.get_obj_ids(data)

        for obj_id in obj_ids:
            result.append(ade(data["objects"][obj_id]["predict_trace"],
                              data["objects"][obj_id]["future_trace"]))

        return result

    def fde(self, data):
        result = []

        obj_ids = self.get_obj_ids(data)

        for obj_id in obj_ids:
            result.append(fde(data["objects"][obj_id]["predict_trace"],
                              data["objects"][obj_id]["future_trace"]))

        return result


class MultiFrameEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.metric_map = {
            "ade": self.ade,
            "fde": self.fde
        }
    
    def ade(self, data):
        result = []

        return result

    def fde(self, data):
        result = []

        return result