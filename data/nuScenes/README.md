* Clone git repository [Trajectron++](git@github.com:StanfordASL/Trajectron-plus-plus.git)
* Download nuScenes dataset following instructions from Trajectron++ repo.
* Execute `preprocessing.py` to produce `prediction_train`, `prediction_val`, `prediction_test` (in Apolloscape format), as well as preprocessed map patches.

Example:
```
# At the current folder (data/nuScenes)
python preprocessing.py --data="." --version="v1.0-trainval" --output_path="."
```
