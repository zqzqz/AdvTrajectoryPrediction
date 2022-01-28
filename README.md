# Adversarial Robustness Analysis of Trajectory Prediction

## Requirements

* Python 3.6+

The dependency packages are listed in `/requirements.txt` which supports all three models (GRIP++, FQA, and Trajectron++) and other analysis code.

```
pip install -r requirements.txt
```

In fact, the `requirements.txt` include packages required by [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) and a few tools, e.g., `SetGPU`, `matplotlib`.

## Directory Structure and Definitions

Parameters:
- `dataset_name`: apolloscape, ngsim, nuscenes
- `model_name`: grip, fqa, trajectron, trajectron_map
- `predict_mode`: single_frame, multi_frame (3-second)
- `attack_mode`: original (white box), augment (data augmentation), smooth (train-time trajectory smoothing), augment_smooth (data augmentation plus train-time trajectory smoothing) smooth2 (test-time trajectory smoothing), smooth3 (conditional test-time trajectory smoothing with detection), blackbox
- `metric`: ade, fde, left, right, front, rear.

Directories:
- `/data`: place for raw datasets.
- `/prediction`: Python module including the implementation of data processing, attacks, and utility tools.
- `/test`: Scripts for running the adversarial robustness analysis.
- `/test/data`: Location of results (by default).
    - `dataset/${dataset_name}/${predict_mode}/`: Formulated trajectory data (universial for all models).
        - `raw`: JSON format trajectory data. File names are like `${case ID}.json`.
        - `visualize`: PNG format visualization of trajectories. File names are like `${case ID}.png`.
    - `${model_name}_${dataset_name}/`
        - `model/${attack_mode}`: pretrained models.
        - `${predict_mode}/${normal or attack}/${attack_mode}`: Prediction results under various modes.
            - `raw`: JSON format result data. File names are like `${case ID}-${object ID}.json` (normal) or `${case ID}-${object ID}-${metric}.json` (attack).
            - `visualize`: PNG format visualization of prediction results. File names are like `${case ID}-${object ID}.png` (normal) or `${case ID}-${object ID}-${metric}.png` (attack).
            - `evaluate`: Summary of prediction performance. `loss_${metric}.txt` lists case ID, object ID, and prediction error. `loss_${metric}.png` draw distribution of prediction error.

Format of JSON-format input trajectory data
```
{
    "observe_length": int,
    "predict_length": int,
    "time_step": float,
    "feature_dimension": int, // extra features other than x-y location coordinates
    "objects": {
        "string object id": {
            "type": int,  // 1: small vehicle 2: large vehicle 3: pedestrian 4: unknown
            "complete": bool, // all time frames are filled
            "visible": bool, // the last frame of history is filled
            "observe_trace": [observe_length * 2],
            "observe_feature": [observe_length * feature_dimension],
            "observe_mask": [observe_length],
            "observe_trace": [observe_length * 2],
            "observe_feature": [observe_length * feature_dimension],
            "observe_mask": [observe_length],
            "future_trace": [predict_length * 2],
            "future_feature": [predict_length * feature_dimension],
            "future_mask": [predict_length],
            "predict_trace": [predict_length * 2] // Empty before inference
        }, ...
    }
}
```

Format of JSON-format output result data
```
{
    "perturbation": [observe_length+attack_length-1 * 2];
    "loss": number or dict,
    "obj_id": string,
    "attack_opts": dict, // other configuration or options of the attack
    "output_data": {
        "0": { // ID of the time frame (string)
            // The content is the same as the input trajectory data
        }, ...
    }
}
```

## Steps to Reproduce

### Prepare datasets

Place datasets in directory `/data` following `README.md` in `data/apolloscape`, `data/NGSIM`, and `data/nuScenes`.

### Prepare models

The models are trained seperatedly for each dataset following the instructions from model authors. The training code is not in this repo and we will provide hyperparameters for training and pretrained models in the future.

The models are placed in `/test/data/${model_name}_${dataset_name}/model/${attack_mode}`.

### Generate universial JSON-format testing data

This is done by using APIs we provide. Here we show code samples for Apolloscape datasets. The translation on various datasets is implemented in `/prediction/dataset`. In directory `test`:

```
from test_utils import multiframe_data
from prediction.dataset import ApolloscapeDataset

output_dir = "data/dataset/apolloscape/multi_frame/raw"
multiframe_data(output_dir, ApolloscapeDataset, obs_length=6, pred_length=6, time_step=0.5)
```

### Run normal prediction as well as the attack

Normal prediction, adversarial attack, and evaluation are done through API `normal_test`, `adv_attack`, and `evaluate_loss` implemented in `test_utils.py`. As a quick start, we can execute `test.py` to run the whole pipeline.

```
python test.py ${model_name} ${dataset_name} 0  # The last parameter is "overwrite". If it is 1, the generated result will overwrite existing files.
```

Tune parameters in `test/test.py:L284-287` to switch prediction modes and training/testing modes.

