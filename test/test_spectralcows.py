import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from SpectralCows import SpectralCowsInterface
from visualize import *

if __name__ == "__main__":
    api = SpectralCowsInterface()
    api.run(None)
