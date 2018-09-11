#!/usr/bin/env python
"""
Do persistence
"""
import os
import logging
import gzip
import shutil

from mnvtf.evtid_utils import decode_eventid

LOGGER = logging.getLogger(__name__)


class MnvCategoricalTextRecorder:
    """
    record segments or planecodes in a text file

    NOTE - user should call `close()` to compress the output file at the end
    of the job.
    """

    def __init__(self, db_base_name):
        LOGGER.info('Setting up {}...'.format(
            self.__class__.__name__
        ))
        self.db_name = db_base_name + '.txt'
        if os.path.isfile(self.db_name):
            LOGGER.info('found existing record file {}, removing'.format(
                self.db_name
            ))
            os.remove(self.db_name)
        LOGGER.info('using record file {}'.format(self.db_name))

    def write_data(self, predictions):
        '''
        predictions is a dict with keys, 'classes', 'probabilities', and
        'eventids'. 'probabilities' is a numpy array.
        '''
        with open(self.db_name, 'ab+') as f:
            evtid_string = ','.join(
                decode_eventid(predictions['eventids'])
            )
            probs_string = ','.join([
                str(i) for i in predictions['probabilities']
            ])
            msg = evtid_string + ',' + str(predictions['classes']) + ',' \
                + probs_string + '\n'
            f.write(bytes(msg, 'utf8'))
        return None

    def read_data(self):
        """
        do not call this on large files
        NOTE: we are assuming gzip compression has occurred!
        """
        with gzip.open(self.db_name + '.gz', 'rb') as f:
            content = f.readlines()
            content = [x.decode('utf8').strip() for x in content]
        return content

    def close(self):
        gzf = self.db_name + '.gz'
        with open(self.db_name, 'rb') as f_in, gzip.open(gzf, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        if os.path.isfile(gzf) and (os.stat(gzf).st_size > 0):
            os.remove(self.db_name)
        else:
            raise IOError('Compressed file not produced!')
