from prediction.base.interface import Interface
from .dataloader import BaselineDataLoader

class BaselineInterface(Interface):
    def __init__(self, dataset_name, obs_length, pred_length):
        super().__init__(dataset_name, obs_length, pred_length)
        self.dataloader = BaselineDataLoader(self.dataset, self.obs_length, self.pred_length)

    def data(self):
        return self.dataloader.generate_data()

    def run(self, input_data):
        for _, obj in input_data["objects"].items():
            v = obj["observe_trace"][self.obs_length-1, :] -  obj["observe_trace"][self.obs_length-2, :]
            v2 = obj["observe_trace"][self.obs_length-2, :] -  obj["observe_trace"][self.obs_length-3, :]
            a = v - v2
            latest_x = obj["observe_trace"][self.obs_length-1, :]
            for i in range(self.pred_length):
                latest_v = v + a
                obj["predict_trace"][i,:] = latest_x + latest_v
                latest_x = obj["predict_trace"][i,:]
        return input_data