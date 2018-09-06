import argparse
import numpy as np
import tensorflow as tf
from mnvtf.model_classes import ConvModel

tf.enable_eager_execution()


def test_conv_model(nout):
    model = ConvModel()
    x = tf.random_normal((1, 127, 94, 1))
    u = tf.random_normal((1, 127, 47, 1))
    v = tf.random_normal((1, 127, 47, 1))
    out = model(x, u, v)
    print(model.summary())
    assert np.any(tf.is_nan(out).numpy()) == False, "output contains nans"
    assert out.get_shape().as_list() == [1, nout], "bad model output shape"
    assert model.count_params() == 3320274, "model parameter number is wrong"


def main(nout):
    test_conv_model(nout)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nout', default=6, type=int, help='num outputs')
    args = parser.parse_args()
    main(**vars(args))
