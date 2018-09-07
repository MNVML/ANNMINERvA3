'''
Note - multi-epoch training for HDF5 inputs will fail with this structure.
Need to make a `train_one_epoch` function that regens the dataset each time.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import time
import datetime
import tensorflow as tf

from mnvtf.model_classes import ConvModel
from mnvtf.data_readers import make_dset

tfe = tf.contrib.eager
tf.logging.set_verbosity(tf.logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--num-epochs', default=1, type=int,
                    help='number of training epochs')
parser.add_argument('--train-file', default='', type=str,
                    help='full path to train file')
parser.add_argument('--eval-file', default='', type=str,
                    help='full path to evaluation file')
parser.add_argument('--model-dir', default='fashion', type=str,
                    help='model dir')

tf.logging.set_verbosity(tf.logging.INFO)
logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
    + str(int(time.time())) + '.txt'
logging.basicConfig(
    filename=logfilename, level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Starting...")
LOGGER.info(__file__)


def loss(model, x, u, v, y):
    prediction = model(x, u, v)
    return tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=prediction
    )


def grad(model, x, u, v, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, u, v, targets)
    return loss_value, tape.gradient(loss_value, model.variables)


def train(
    model, optimizer, dataset, global_step, checkpoint, checkpoint_prefix
):
    # must include a record_summaries_method
    with tf.contrib.summary.record_summaries_every_n_global_steps(20):
        for (i, (_, labels, x, u, v)) in enumerate(dataset):
            global_step.assign_add(1)
            train_loss, grads = grad(model, x, u, v, labels)
            optimizer.apply_gradients(
                zip(grads, model.variables), global_step=global_step
            )
            tf.contrib.summary.scalar('loss', train_loss)
            if i % 20 == 0:
                LOGGER.info(
                    'loss at step {:03d}: {:.3f}'.format(i, train_loss)
                )
                checkpoint.save(file_prefix=checkpoint_prefix)
            if i > 40:
                break   # short test for now...
        checkpoint.save(file_prefix=checkpoint_prefix)


def test(model, dataset):
    avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

    for (_, labels, x, u, v) in dataset:
        logits = model(x, u, v)
        avg_loss(loss(model, x, u, v, labels))
        accuracy(
            tf.argmax(logits, axis=1, output_type=tf.int32),
            tf.argmax(labels, axis=1, output_type=tf.int32)
        )

    LOGGER.info('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
                (avg_loss.result(), 100 * accuracy.result()))
    # need a separate writer (either `with`or as default) to keep distinct
    # with tf.contrib.summary.always_record_summaries():
    #     tf.contrib.summary.scalar('loss', avg_loss.result())
    #     tf.contrib.summary.scalar('accuracy', accuracy.result())


def main(batch_size, num_epochs, train_file, eval_file, model_dir):
    if num_epochs != 1:
        print(__doc__)
        import sys
        sys.exit(1)
    tf.enable_eager_execution()

    model = ConvModel()
    dataset = make_dset(
        train_file, batch_size, num_epochs=num_epochs, shuffle=True
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    # writer _can_ make its own log directory
    writer = tf.contrib.summary.create_file_writer(model_dir)
    global_step = tf.train.get_or_create_global_step()
    # writer.set_as_default()  # use a scope instead...

    # os.makedirs(model_dir)   # model dir must exist
    checkpoint_prefix = os.path.join(model_dir, 'ckpt')
    checkpoint = tfe.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=global_step
    )
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))

    _, labels, x, u, v = iter(dataset).next()
    LOGGER.info('initial loss: {:.3f}'.format(loss(model, x, u, v, labels)))

    # training loop
    with writer.as_default():
        t0 = time.perf_counter()
        train(
            model, optimizer, dataset, global_step,
            checkpoint, checkpoint_prefix
        )
        t1 = time.perf_counter()
        LOGGER.info(' epoch train time: {}'.format(
            str(datetime.timedelta(seconds=t1-t0))
        ))

    test_dataset = make_dset(eval_file, batch_size, num_epochs=1)
    test(model, test_dataset)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
