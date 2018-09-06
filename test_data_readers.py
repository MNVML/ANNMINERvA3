from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging
import argparse

import tensorflow as tf

from mnvtf.data_readers import make_dset
from mnvtf.data_readers import make_iterators

# Get path to data
TFILE = os.path.join(
    os.environ['HOME'],
    'Dropbox/Data/RandomData/hdf5',
    'hadmultkineimgs_mnvvtx_test.hdf5'
)

logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] + '.txt'
logging.basicConfig(
    filename=logfilename, level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting...")
logger.info(__file__)
logger.info(' Examining file {}'.format(TFILE))


def test_graph_one_shot_iterator_read(
    hdf5_file=TFILE, batch_size=25, num_epochs=1
):
    feats, labs = make_iterators(hdf5_file, batch_size, num_epochs)
    with tf.Session() as sess:
        total_batches = 0
        total_examples = 0
        try:
            while True:
                xs, ev, ls = sess.run([
                    feats['x_img'], feats['eventids'], labs
                ])
                logger.info('{}, {}, {}, {}, {}, {}'.format(
                    xs.shape, xs.dtype,
                    ev.shape, ev.dtype,
                    ls.shape, ls.dtype
                ))
                total_batches += 1
                total_examples += ls.shape[0]
        except tf.errors.OutOfRangeError:
            logger.info('end of dataset at total_batches={}'.format(
                total_batches
            ))
        except Exception as e:
            logger.error(e)
    logger.info('saw {} total examples'.format(total_examples))


def test_eager_one_shot_iterator_read(
    hdf5_file=TFILE, batch_size=25, num_epochs=1
):
    tfe = tf.contrib.eager
    tf.enable_eager_execution()
    targets_and_labels = make_dset(
        hdf5_file, batch_size, num_epochs
    )

    total_examples = 0
    for i, (ev, ls, xs, us, vs) in enumerate(tfe.Iterator(targets_and_labels)):
        logger.info('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
            ev.shape, ev.dtype,
            ls.shape, ls.dtype,
            xs.shape, xs.dtype,
            us.shape, us.dtype,
            vs.shape, vs.dtype,
        ))
        total_examples += ls.shape[0]
        if i % 10 == 9:
            logger.info(str(ls))
    logger.info('saw {} total examples'.format(total_examples))


def main(eager, batch_size):
    if eager:
        test_eager_one_shot_iterator_read(batch_size=batch_size)
    else:
        test_graph_one_shot_iterator_read(batch_size=batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eager', default=False, action='store_true',
        help='Use Eager execution'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10,
        help='Batch size'
    )
    args = parser.parse_args()
    main(**vars(args))
