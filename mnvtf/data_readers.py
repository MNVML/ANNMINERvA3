from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf


def _make_generator_fn(file_name, batch_size):
    """
    make a generator function that we can query for batches
    """
    from mnvtf.hdf5_readers import SimpleCategorialHDF5Reader as HDF5Reader
    reader = HDF5Reader(file_name)
    nevents = reader.openf()

    def example_generator_fn():
        start_idx, stop_idx = 0, batch_size
        while True:
            if start_idx >= nevents:
                reader.closef()
                return
            yield reader.get_samples(start_idx, stop_idx)
            start_idx, stop_idx = start_idx + batch_size, stop_idx + batch_size

    return example_generator_fn


def make_dset(
    file_name, batch_size, num_epochs=1, shuffle=False
):
    # make a generator function - read from HDF5
    dgen = _make_generator_fn(file_name, batch_size)

    # make a Dataset from a generator
    x_shape = [None, 127, 94, 2]
    uv_shape = [None, 127, 47, 2]
    labels_shape = [None, 6]
    evtids_shape = [None]
    # TF doesn't support uint{32,64}, but leading bit should be zero for us
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.int64, tf.int32, tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(evtids_shape),
         tf.TensorShape(labels_shape),
         tf.TensorShape(x_shape),
         tf.TensorShape(uv_shape),
         tf.TensorShape(uv_shape))
    )
    # we are grabbing an entire "batch", so don't call `batch()`, etc.
    # also, note, there are issues with doing more than one epoch for
    # `from_generator` - so do just one epoch at a time for now.
    ds = ds.prefetch(10)
    if shuffle:
        ds = ds.shuffle(10)

    return ds


def make_iterators(
        file_name, batch_size, num_epochs=1, shuffle=False,
):
    '''
    estimators require an input fn returning `(features, labels)` pairs, where
    `features` is a dictionary of features.
    '''
    ds = make_dset(
        file_name, batch_size, num_epochs, shuffle
    )

    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    eventids, labels, x_img, u_img, v_img = itrtr.get_next()
    features = {}
    features['x_img'] = x_img
    features['u_img'] = u_img
    features['v_img'] = v_img
    features['eventids'] = eventids
    return features, labels


def get_data_files_dict(path='path_to_data'):
    data_dict = {}
    data_dict['train'] = os.path.join(
        path, 'hadmultkineimgs_mnvvtx_train.hdf5'
    )
    data_dict['test'] = os.path.join(
        path, 'hadmultkineimgs_mnvvtx_test.hdf5'
    )
    return data_dict
