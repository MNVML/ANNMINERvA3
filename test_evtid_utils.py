import os
import h5py
from mnvtf.evtid_utils import compare_evtid_encodings


if __name__ == '__main__':

    HDF5FILE = os.path.join(
        os.environ['HOME'],
        'Dropbox/Data/RandomData/hdf5',
        'hadmultkineimgs_mnvvtx_test.hdf5'
    )
    f = h5py.File(HDF5FILE, 'r')
    n_events = f['event_data']['eventids'].shape[0]
    print('n_events = {}'.format(n_events))

    for i in range(n_events):
        evtid = f['event_data']['eventids'][i: i + 1]
        evtia = f['event_data']['eventids_a'][i: i + 1]
        evtib = f['event_data']['eventids_b'][i: i + 1]
        check = compare_evtid_encodings(evtid, evtia, evtib)
        if not check:
            print('found a mismatch!')
            print(i, evtid, evtia, evtib)
