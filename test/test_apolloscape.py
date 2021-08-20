import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.apolloscape import ApolloscapeDataset


def test_apolloscape():
    dataset = ApolloscapeDataset(6, 6, 0.5)
    dataset.generate_data("train")
    dataset.generate_data("test")
    print(dataset.data_size("test"))
    for input_data in dataset.data_generator("train"):
        print(input_data)
        break

if __name__ == "__main__":
    test_apolloscape()