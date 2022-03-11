# Adversarial Robustness Analysis of Trajectory Prediction

## Requirements

* Python 3.6+

The dependency packages are listed in `/requirements.txt` which supports all three models (GRIP++, FQA, and Trajectron++) and other analysis code.

```
pip install -r requirements.txt
```

The `requirements.txt` include packages required by [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) and a few tools e.g., `matplotlib` for visualization and `pyswarm` for PSO implementation.

* We assume the user has GPU access. The code is tested on CUDA 10.2 and RTX 2080. The code is based on `PyTorch`.

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

## Steps to reproduce

### Prepare datasets

First of all, we provide formulated test cases via [Google Drives](https://drive.google.com/file/d/1WxFotCnTC6oSqFTtc7PAkBHM6CNrTZJ4/view?usp=sharing). Download the ZIP file and unzip it into directory `test/data`. By doing so, you can skip the following steps in this subsection.

First, place datasets in directory `/data` following `README.md` in `data/apolloscape`, `data/NGSIM`, and `data/nuScenes`.

Second, this codebase translate raw dataset into JSON-format testing data. This is done by using APIs we provide. Here we show code samples for Apolloscape datasets. The translation on various datasets is implemented in `/prediction/dataset`. 

To quickly generate the JSON-format test cases, run scripts in directory `test`:

```
python generate_data.py ${dataset_name}
```


### Prepare models

The models are trained seperatedly for each dataset following the instructions from model authors. 
The models should be placed in `/test/data/${model_name}_${dataset_name}/model/${attack_mode}`.

The training code is not in this repo but we provide pretrained models via [Google Drives](https://drive.google.com/file/d/18240VaDBKSwa5TzZjCnU99EVqEXv9uiG/view?usp=sharing). Download the ZIP file and unzip it into directory `test/data`


### Run normal prediction as well as the attack

Normal prediction, adversarial attack, and evaluation are done through API `normal_test`, `adv_attack`, and `evaluate_loss` implemented in `test_utils.py`. As a quick start, we can execute `test.py` to run the whole pipeline.

```
python test.py --help
```

The script contains following parameters:

* `dataset`: the dataset's name, by default `apolloscape`.
* `model`: the model's name, by default `grip`.
* `mode`: the prediction mode, by default `single_frame`.
* `augment`: boolean flag; adding the option enables data augmentation.
* `smooth`: integer flag; 0 disable trajectory smoothing; 1 enable train-time smoothing; 2 enable test-time smoothing.
* `blackbox`: boolean flag; adding the option enables blackbox attack instead of whitebox.
* `overwrite`: boolean flag; if adding the option, generated data will overwrite existing data. False by default.

For executing normal tests or attacks on specific test case, see function `normal_sample` and `attack_sample` in `test/test_utils.py`

## For developer

### Add custom datasets

Similar to `prediction/dataset/apolloscape.py`, the developer should write a class inheriting `prediction.dataset.base.BaseDataset` and implement interface `format_data`.
`format_data` should be a generator and  use `yield` to output test cases in the JSON-format defined before.

### Add custom prediction models

Similar to `prediction/model/GRIP/interface.py`, the developer should write a class inheriting `prediction.model.base.interface.Interface`.
The developer should implement the `run` interface, which accept the JSON-format test case (defined before) and a dictionary called `perturbation`.

The perturbation structure is defined as follows. For more details, please see the implementation of `prediction.attack.gradient.GradientAttacker`.
```
{
  "obj_id": str - the vehicle id whose trajectory is to be perturbed,
  "loss": function instance, e.g., prediction.attack.loss.attack_loss,
  "value": {obj_id: torch tensor of perturbation},
  "ready_value": {obj_id: torch tensor of perturbation after the constraint checking},
  "attack_opts": {
    "type": str - evalaution metric in ["ade", "fde", "left", "right", "front", "rear"],
    ... other parameters used in loss function
  }
}
```
