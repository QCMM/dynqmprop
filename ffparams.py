import warnings
import time

from iodata import load_one
from denspart.adapters.horton3 import prepare_input
from denspart.mbis import partition
from denspart.properties import compute_rcubed

import numpy as np

# avoid warnings
#warnings.filterwarnings('ignore')
np.seterr(all='warn')

class ForceFieldParams(object):

    def __init__(self, infile):
        self.data = load_one(infile)  # data from the orca mkl file
        self.grid = None
        self.rho = None
        self.pro_model = None

    # set molgrid
    def set_molgrid(self):
        self.grid, self.rho = prepare_input(self.data, 150, 194)  # molgrid
        return

    # do partitioning depending on method
    def do_partitioning(self, method='mbis'):
        begin_time = time.time()
        if method == 'hi':
            print('Pending')
            return
        elif method == 'mbis':
            # do MBIS partitioning
            print('MBIS partitioning ...')
            pro_model = partition(self.data.atnums, self.data.atcoords, self.grid, self.rho)
        else:
            print('Invalid method')
            return
        self.pro_model = pro_model
        end_time = time.time()
        time_hours = (end_time - begin_time) / 60
        print(f'Time: {round(time_hours, 2)} minutes')
        return

    def get_charges(self):
        pro_model, localgrids = self.pro_model
        self.data.atffparams['charges'] = pro_model.charges
        _charges = [round(c, 6) for c in pro_model.charges]  # rounded charges
        # print(f'Charges:\n{chrgs}\n')
        # print(f'Total charge: {self.pro_model.charges.sum()}')
        return _charges

    def get_rcubed(self):
        self.data.atffparams['rcubed'] = compute_rcubed(self.pro_model, self.grid, self.rho)
        rc = [round(r, 6) for r in self.data.atffparams['rcubed']]
        # print(f'R^3 moments:\n {self.data.atffparams["rcubed"]}')
        return rc
