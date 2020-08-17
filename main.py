import sys

from dynqmprop import DynQMProp

top_file = sys.argv[1]
coords_file = sys.argv[2]


def main():
    charges_param = DynQMProp(top_file, coords_file, qm_charge=0, parmed_selection=f':1',
                              n_charge_updates=2, sampling_time=1, total_qm_calculations=5)
    charges_out, charges_std_out, epol_out = charges_param.set_output_files()
    charges_param.run(charges_out, charges_std_out, epol_out)


if __name__ == '__main__':
    main()
