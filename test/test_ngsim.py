import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.dataset.ngsim import NGSIMDataset


def test_ngsim():
    dataset = NGSIMDataset(6, 6, 0.5)
    for input_data in dataset.test_data_generator():
        print(input_data)
        break

if __name__ == "__main__":
    test_ngsim()