
**Input/Output data of prediction**

```(json)
{
    "observe_length": int,
    "predict_length": int,
    "objects": {
        int (object id): {
            "type": str in ["vehicle", "pedestrian", "bicycle"],
            "observe_trace": numpy array (observe_length * 2),
            "future_trace": numpy array (predict_length * 2),
            "predict_trace": numpy array (predict_length * 2),
        }, ...
    }
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

