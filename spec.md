
**Input/Output data of prediction**

```(json)
{
    "observe_length": int,
    "predict_length": int,
    "objects": {
        string (object id): {
            "type": str in {1:"small vehicle", 2:"big vehicle", 3:"pedestrian", 4:"bicycle", 5:"other"},
            "observe_trace": numpy array (observe_length * 2: pos_x, pos_y),
            "future_trace": numpy array (predict_length * 2),
            "predict_trace": numpy array (predict_length * 2),
            "full_observe_trace": numpy array (observe_length * 7: pos_x, pos_y, pos_z, obj_length, obj_width, obj_height, heading),
            "full_future_trace": numpy array (future_length * 7)
        }, ...
    }
}
```

**Perturbation data**

```(json)
{
    "value": {obj_id: torch tensor variable (attack_step * 2)},
    "ready_value": {obj_id: torch tensor variable (attack_step * 2: processed from "value" e.g., set bounds)},
    "loss": function (loss function accepting predict_trace, future_trace, and perturbation value as inputs),
    "obj_id": int (the target object id),
    "mode": str ("single": perturb the target object only; "all": perturb all surrounding objects; "select": specify the objects for perturbation),
}
```

**Interfaces of dataloaders**

* `generate_data`. Raw data from dataset to json-format input data. Implemented as a data generator using `yield`.
    * Parameters: None
    * Returns: generator instance
* `preprocess`. Transform json-format input data to data accepted by the prediction algorithm.
    * Parameters: input_data (json)
    * Returns: undecided
* `postprocess`. Transform the outputs of the prediction algorithm to json-format output data.
    * Parameters: undecided
    * Returns output_data (json)


**Interfaces of prediction algorithms**

* `data`. Return the data generator. Delegate of `dataloader.generate_data`.
* `run`. One pass of prediction algorithm.
    * Parameters: input_data (json)
    * Returns: output_data (json)

