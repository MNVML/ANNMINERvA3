# ANNMINERvA3

This is a Python3 TF framework.

* `eager_hadmult_simple.py` -  Run classification in Eager mode. This is not
meant for "production," but rather for debugging model code.
* `estimator_hadmult_simple.py` - Run classification using the `Estimator` API.
* `mnvtf/`
  * `data_readers.py` - collection of functions for ingesting data using the
  `tf.data.Dataset` API.
  * `estimator_fns.py` - collection of functions supporting the `Estimator`s.
  * `hdf5_readers.py` - collection of classes for reading HDF5 (used by
    `data_readers.py`).
  * `model_classes.py` - collection of (Keras) models used here (Eager code
    relies on Keras API).
* `run_eager_hadmult_simple.sh` - Runner script for `eager_hadmult_simple.py`
meant for short, interactive tests.
* `run_estimator_hadmult_simple.sh` - Runner script for
`estimator_hadmult_simple.py` meant for short, interactive tests.
* `test_data_readers.py` - Exercise the data reader classes.
* `test_models.py` - Exercise model creation code.
