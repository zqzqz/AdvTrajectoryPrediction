- http://apolloscape.auto/trajectory.html
- https://github.com/ApolloScapeAuto/dataset-api/blob/master/trajectory_prediction/readme.md

At least download `prediction_train.zip` and unzip it in this directory.

The meaing of each line:

- `sample_trajectory.zip`: timestamp, id, width, length, type, position_x, position_y, velocity_x, velocity_y
- `prediction_train.zip` and `prediction_test.zip`: frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading

Divide files in `prediction_train.zip` in three seperate directories: `prediction_train`, `prediction_val`, and `prediction_test`.
