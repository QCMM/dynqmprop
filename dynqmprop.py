import time
import os
import shutil

import dynqmprop.mdtools as mdt
import parmed as pmd

from simtk.openmm import app

class DynQMProp(object):

    def __init__(self, top_file, coords_file, qm_charge=0, ligand_selection=f':1', receptor_selection=None,
                 radius=10, n_charge_updates=3, sampling_time=25, total_qm_calculations=100, method='B3LYP',
                 basis='def2-TZVP'):

        self.top_file = top_file
        self.coords_file = coords_file
        self.qm_charge = qm_charge  # total qm charge
        # Residue index for molecule to calculate charges
        self.ligand_selection = ligand_selection
        if receptor_selection is not None:
            self.receptor_selection = receptor_selection
        self.radius = radius
        self.n_charge_updates = n_charge_updates
        self.sampling_time = sampling_time  # sampling time in ns
        self.total_qm_calculations = total_qm_calculations
        self.method = method
        self.basis = basis

        if self.top_file.endswith(".top"):
            self.top_format = "gromacs"
            self._gromacs_top_path = '/opt/easybuild/software/GROMACS/2019.3-foss-2019a/share/gromacs/top'
            pmd.gromacs.GROMACS_TOPDIR = self._gromacs_top_path
            self.coords = app.gromacsgrofile.GromacsGroFile(self.coords_file)
            self.box_vectors = self.coords.getPeriodicBoxVectors()
            self.top = app.gromacstopfile.GromacsTopFile(
                self.top_file, periodicBoxVectors=self.box_vectors, includeDir=self._gromacs_top_path)
        elif self.top_file.endswith(".prmtop"):
            self.top_format = "amber"
            self.coords = app.amberinpcrdfile.AmberInpcrdFile(self.coords_file)
            self.box_vectors = self.coords.boxVectors
            self.top = app.amberprmtopfile.AmberPrmtopFile(self.top_file)

        else:
            raise Exception(
                'Topology not implemented ...')

    def set_output_files(self, charges_file='charges.out', epol_file='epol.out'):

        # define output files
        self._charges_file = charges_file
        self._charges_std_file = f'{charges_file.split(".")[0]}_std.out'
        self._epol_file = epol_file

        charges_out = open(self._charges_file, 'w')
        charges_std_out = open(self._charges_std_file, 'w')
        epol_out = open(self._epol_file, 'w')
        charges_out.write('#index \n')
        charges_std_out.write('#index \n')
        return charges_out, charges_std_out, epol_out

    def run(self, charges_out, charges_std_out, epol_out, compl=False):

        begin_time = time.time()
        if compl:
            print('Setting restraints ...')
            restraint = True
            ligand_atom_list, receptor_atom_list = mdt.set_restrained_atoms(
                self.top_file, self.coords_file, self.ligand_selection, self.receptor_selection)
        shutil.copyfile(self.top_file, f'{self.top_file}.old')
        if self.top_format == 'amber':
            b_vectors = self.box_vectors
        else:
            b_vectors = None
        for update in range(self.n_charge_updates):
            # list to store charges and polarization energies for each update
            charge_list = []
            epol_list = []
            # creating OpenMM simulation class
            if update == 0:
                positions = self.coords.positions
            if compl:
                simulation, system = mdt.setup_simulation(
                    self.top, positions, update, b_vectors, restraint, ligand_atom_list, receptor_atom_list)
            else:
                simulation, system = mdt.setup_simulation(
                    self.top, positions, update, b_vectors)
            # starting loop to calculate atomic charges from different conformations
            qm_calculations = int(
                self.total_qm_calculations / self.n_charge_updates) * (update + 1)
            for i in range(qm_calculations):
                step = int(self.sampling_time * 500000 / qm_calculations)
                simulation.step(step)
                # calculate charges and polarization energy for current configuration
                print('Calculating charges ...')
                positions, epol, charges = mdt.calculate_charges(simulation, system, self.ligand_selection,
                                                                 self.qm_charge, self.radius, self.method, self.basis)
                epol_list.append(epol)
                charge_list.append(charges)
            new_charges, new_charges_std = mdt.charge_stats(charge_list)
            epol_mean, epol_std = mdt.epol_stats(epol_list)
            print('Creating new topology ...')
            self.top = mdt.make_new_top(
                self.top_file, self.box_vectors, new_charges, self.ligand_selection)
            for i in range(len(new_charges)):
                charges_out.write(f'{new_charges[i]:.6f}  ')
                charges_std_out.write(f'{new_charges_std[i]:.6f}  ')
            charges_out.write('\n')
            charges_std_out.write('\n')
            epol_out.write(f'{epol_mean:.3f}  {epol_std:.3f}\n')

        end_time = time.time()
        time_hours = (end_time - begin_time) / 3600
        print(f'Time {time_hours} hours')
        charges_out.close
        epol_out.close
        return

    def validation(self, parm=None, parm_vals=[], overwrite=False, compl=False):

        parm_opt = ['radius', 'sampling_time',
                    'n_charge_updates', 'total_qm_calculations', 'method', 'basis']

        if str(parm) in parm_opt and isinstance(parm_vals, list):
            parm_dict = dict({parm: parm_vals})
        else:
            raise Exception(
                '''Usage:
                        parm: str ('radius', 'sampling_time', 'n_charge_updates', 'total_qm_calculations', 'method', 'basis')
                        parm_vals: list
                        ''')

        for val in parm_dict[parm]:
            if parm == 'radius':
                self.radius = float(val)
            elif parm == 'sampling_time':
                self.sampling_time = float(val)
            elif parm == 'total_qm_calculations':
                self.total_qm_calculations = int(val)
            elif parm == 'method':
                self.method = str(val)
            elif parm == 'basis':
                self.basis = str(val)
            else:
                self.n_charge_updates = int(val)

            parm_dir = f'{parm[0]}_{val}'
            # create target directory & all intermediate directories if don't exists
            try:
                os.makedirs(parm_dir)
                print(f'Directory {parm_dir} created')
            except FileExistsError:
                print(f'Directory {parm_dir} already exists')
                if not overwrite:
                    print(f'Not overwrite {parm_dir} ...')
                    pass
            shutil.copyfile(self.top_file, f'{parm_dir}/{self.top_file}')
            shutil.copyfile(self.coords_file,
                            f'{parm_dir}/{self.coords_file}')
            shutil.copyfile('calculate_charges.py',
                            f'{parm_dir}/calculate_charges.py')
            os.chdir(parm_dir)
            charges_out, charges_std_out, epol_out = self.set_output_files()
            self.run(charges_out, charges_std_out, epol_out, compl)
            os.chdir('..')
        return
