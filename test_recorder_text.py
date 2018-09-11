from mnvtf.recorder_text import MnvCategoricalTextRecorder as Recorder
from mnvtf.evtid_utils import encode_eventid
import numpy as np
import time
import math

p1 = np.asarray([
    0.05629708, 0.07719567, 0.05056403, 0.06280191, 0.71456647, 0.03857485
], dtype=np.float32)
pd1 = {'classes': 4, 'probabilities': p1, 'eventids': 1141500001000301}
p2 = np.asarray([
    0.04713702, 0.09934037, 0.02083675, 0.02786209, 0.39107746, 41374636
], dtype=np.float32)
pd2 = {'classes': 5, 'probabilities': p2, 'eventids': 1141500001000801}

predictions = [pd1, pd2]


def test_write(tfile):
    recorder = Recorder(tfile)
    for p in predictions:
        recorder.write_data(p)
    recorder.close()


def test_read(tfile):
    recorder = Recorder(tfile)
    data = recorder.read_data()
    for i, d in enumerate(data):
        print(d)
        elems = d.split(',')
        eventid = int(encode_eventid(elems[0], elems[1], elems[2], elems[3]))
        assert eventid == predictions[i]['eventids']
        predicted_class = int(elems[4])
        assert predicted_class == predictions[i]['classes']
        probabilities = [float(x) for x in elems[5:]]
        for j, p in enumerate(probabilities):
            assert math.fabs(p - predictions[i]['probabilities'][j]) < 0.0001


if __name__ == '__main__':
    tfile = '/tmp/reader_test' + str(int(time.time()))
    test_write(tfile)
    test_read(tfile)
