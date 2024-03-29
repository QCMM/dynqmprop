import sys

from dynqmprop import DynQMProp

top_file = sys.argv[1]
coords_file = sys.argv[2]


def main():
    # creating objects for validation
    # standard parameters:
    #   radius: 10
    #   sampling: 25
    #   n_charge_updates: 3
    radius_param = DynQMProp(top_file, coords_file, qm_charge=+1, ligand_selection=':2', receptor_selection=':1')
    # sampling_param = DynQMProp(top_file, coords_file, qm_charge=+1, ligand_selection=':2', receptor_selection=':1', radius=12, n_charge_updates=1)
    # n_charge_updates_param = DynQMProp(top_file, coords_file, qm_charge=0, ligand_selection=f':1', radius=10, n_charge_updates=2, sampling_time=1, total_qm_calculations=5)
    # basis_param = DynQMProp(top_file, coords_file, qm_charge=0, ligand_selection=f':1', radius=10, n_charge_updates=2, sampling_time=1, total_qm_calculations=5, method='HF')
    # parameter validation run
    radius_param.validation('radius', [12, 14], overwrite=True, compl=True)
    # sampling_param.validation('sampling_time', [15, 20, 25, 30, 35, 40], overwrite=True)
    # n_charge_updates_param.validation(
    #    'n_charge_updates', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], overwrite=True)
    # basis_param.validation(
    #    'basis', ['def2-SVP', 'def2-TZVP', 'cc-pVDZ', 'cc-pVTZ'], overwrite=True)


if __name__ == '__main__':
    main()
