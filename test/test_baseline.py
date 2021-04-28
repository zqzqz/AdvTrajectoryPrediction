import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prediction.baseline import BaselineInterface
from prediction.evaluate import *
from prediction.visualize import *

if __name__ == "__main__":
    api = BaselineInterface("apolloscape", 6, 6)
    ade, fde = evaluate_error(api, online=True)
    draw_error_distribution(ade, fde, "error.png")
    print("ade:", sum(ade)/len(ade), "fde:", sum(fde)/len(fde))

    # for input_data in api.data():
    #     output_data = api.run(input_data)
    #     print(output_data)
    #     draw_traces(output_data, "trace.png")
    #     break