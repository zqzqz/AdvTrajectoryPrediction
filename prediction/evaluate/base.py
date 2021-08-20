import json
import numpy as np

class Evaluator:
    def __init__(self):
        self.metric_map = {
            # TODO
        }

    def evaluate_data_on_one_metric(self, data, metric):
        return self.metric_map[metric](data)

    def evaluate_data_on_all_metrics(self, data):
        result = {}
        for metric in self.metric_map:
            result[metric] = self.evaluate_data_on_one_metric(data, metric)
        return result

    def evalute_metric(self, data_generator, metric):
        report = {}

        for name, data in data_generator:
            result = self.evaluate_data_on_one_metric(data, metric)
            report[name] = result

        return report

    def evaluate(self, data_generator):
        report = {}
        for metric in self.metric_map:
            report[metric] = {}
        
        for name, data in data_generator:
            result = self.evaluate_data_on_all_metrics(data)
            for metric in result:
                report[metric][name] = result[metric]

        return report
