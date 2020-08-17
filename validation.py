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
    basis_param = DynQMProp(top_file, coords_file, qm_charge=0, parmed_selection=f':1',
                            radius=10, n_charge_updates=2, sampling_time=1, total_qm_calculations=5, method='HF')
    # parameter validation run
    basis_param.validation(
        'basis', ['def2-SVP', 'def2-TZVP', 'cc-pVDZ', 'cc-pVTZ'], overwrite=True)


if __name__ == '__main__':
    main()
