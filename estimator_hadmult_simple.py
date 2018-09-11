from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import time
import datetime
import tensorflow as tf

from mnvtf.data_readers import make_iterators
from mnvtf.estimator_fns import est_model_fn
from mnvtf.recorder_text import MnvCategoricalTextRecorder as Recorder


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=100, type=int, help='batch size')
parser.add_argument('--train-steps', default=None, type=int,
                    help='number of training steps')
parser.add_argument('--num-epochs', default=1, type=int,
                    help='number of epochs')
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


def predict(classifier, data_files, hyper_pars):
    # predictions is a generator - evaluation is lazy
    predictions = classifier.predict(
        input_fn=lambda: make_iterators(
            data_files['test'], hyper_pars['batch_size']
        ),
    )
    recorder = Recorder(hyper_pars['predictions_file'])
    for p in predictions:
        recorder.write_data(p)
    recorder.close()


def evaluate(classifier, data_files, hyper_pars):
    eval_result = classifier.evaluate(
        input_fn=lambda: make_iterators(
            data_files['test'], hyper_pars['batch_size']
        ),
        steps=1000,
    )
    LOGGER.info('\nEval:')
    LOGGER.info('acc: {accuracy:0.3f},'
                ' loss: {loss:0.3f},'
                ' MPCA {mpca:0.3f}'.format(
                    **eval_result
                ))


def train_one_epoch(classifier, data_files, hyper_pars):
    classifier.train(
        input_fn=lambda: make_iterators(
            data_files['train'], hyper_pars['batch_size'], shuffle=True
        ),
        steps=hyper_pars['train_steps']
    )


def train(classifier, data_files, hyper_pars):
    for i in range(hyper_pars['num_epochs']):
        LOGGER.info('training epoch {}'.format(i))
        t0 = time.perf_counter()
        train_one_epoch(classifier, data_files, hyper_pars)
        t1 = time.perf_counter()
        LOGGER.info(' epoch train time: {}'.format(
            str(datetime.timedelta(seconds=t1-t0))
        ))
        LOGGER.info('evaluation after epoch {}'.format(i))
        evaluate(classifier, data_files, hyper_pars)


def main(
    batch_size, train_steps, num_epochs, train_file, eval_file, model_dir
):
    import os

    data_files = {}
    data_files['train'] = train_file
    data_files['test'] = eval_file
    hyper_pars = {}
    hyper_pars['batch_size'] = batch_size
    hyper_pars['num_epochs'] = num_epochs
    hyper_pars['train_steps'] = train_steps
    # TODO - pass in predictions output path
    hyper_pars['predictions_file'] = os.path.join(
        model_dir, 'predictions'
    )

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_summary_steps=10,
        keep_checkpoint_max=3,
        model_dir=model_dir,
        tf_random_seed=None,
    )
    classifier = tf.estimator.Estimator(
        model_fn=est_model_fn,
        params={},
        config=run_config
    )
    t0 = time.perf_counter()
    train(classifier, data_files, hyper_pars)
    t1 = time.perf_counter()
    LOGGER.info(' total train time: {}'.format(
        str(datetime.timedelta(seconds=t1-t0))
    ))
    predict(classifier, data_files, hyper_pars)
    t1 = time.perf_counter()
    LOGGER.info(' total run time: {}'.format(
        str(datetime.timedelta(seconds=t1-t0))
    ))


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
