* Download NGSIM data from the [website](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm). Make sure `I-80/vehicle-trajectory-data` and `US-101/vehicle-trajectory-data` is correctly placed.
* Run matlab script `preprocessing.m`. It generates `TrainSet.mat`, `ValSet.mat`, and `TestSet.mat`. The code is modified from [conv-social-pooling](https://github.com/nachiket92/conv-social-pooling/blob/master/preprocess_data.m).
* Run python script `format.py`, which generates `prediction_train`, `prediction_val`, and `prediction_test` in Apolloscape format.
