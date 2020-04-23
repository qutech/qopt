import qutil.storage
import qutip.control_2.data_container

import h5py
import os
import time
import unittest

import pandas as pd


class TestStorage(unittest.TestCase):
    def test_read_write(self):
        file = r"Z:\SimulationData\Qutip\Tests\storage_test\test1"
        if os.path.isdir(file):
            try:
                os.remove(file)
            except PermissionError:
                pass

        infidelities = pd.DataFrame(
            {'fid1': pd.Series(data=[1, 2]), 'fid2': pd.Series(data=[1, 2])})
        a = qutip.control_2.data_container.DataContainer(
            costs=infidelities,
            final_costs=infidelities,
            init_parameters=infidelities,
            final_parameters=infidelities,
            storage_path=file,
            file_name='test',
            append_time_to_path=False)

        try:
            a.write_to_hdf5()
        except ValueError:
            pass

        time.sleep(1)

        hdf5_handel = h5py.File(os.path.join(file, 'test' + r".hdf5"), mode='r')
        b = qutil.storage.from_hdf5(hdf5_handel, reserved=dict())

        assert (infidelities.equals(b['DataContainer'].infidelities))
