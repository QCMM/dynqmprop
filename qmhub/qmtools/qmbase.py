import os
import subprocess as sp
import numpy as np

from .. import units

class QMBase(object):

    QMTOOL = None

    def __init__(self, basedir, embed, charge=None, mult=None):
        """
        Creat a QM object.
        """

        self.basedir = basedir
        self._embed = embed

        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1

        self.update()

    def update(self):
        self.get_qm_system(self._embed)
        self.get_mm_system(self._embed)

    @staticmethod
    def get_nproc():
        """Get the number of processes for QM calculation."""
        if 'OMP_NUM_THREADS' in os.environ:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        elif 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1
        return nproc

    @staticmethod
    def load_output(output_file):
        """Load output file."""

        f = open(output_file, 'r')
        output = f.readlines()
        f.close()

        return output

    def get_qm_system(self, embed):
        """Load MM information."""

        self.qm_atoms = embed.qm_atoms

        self._n_qm_atoms = self.qm_atoms.n_atoms
        self._qm_element = self.qm_atoms.element
        self._qm_position = self.qm_atoms.position

    def get_mm_system(self, embed):
        """Load MM information."""

        self.mm_atoms_near = embed.mm_atoms_near
        self.mm_atoms_far = embed.mm_atoms_far

    def get_qm_params(self, calc_forces=None, read_guess=None, addparam=None):
        if calc_forces is not None:
            self.calc_forces = calc_forces
        elif not hasattr(self, 'calc_forces'):
            self.calc_forces = True

        if read_guess is not None:
            self.read_guess = read_guess
        elif not hasattr(self, 'read_guess'):
            self.read_guess = False

        self.addparam = addparam

    def run(self):
        """Run QM calculation."""

        cmdline = self.gen_cmdline()

        if not self.read_guess:
            self.rm_guess()

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    def get_fij_near(self):
        """Get pair-wise forces between QM charges and MM charges."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()

        self.fij_near = -1 * self._qmmm_efield_near * self.qm_charge[np.newaxis, :, np.newaxis] / units.F_AU

        return self.fij_near

    def get_fij_far_qmmm(self):
        """Get pair-wise forces between QM charges and MM charges."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()

        self.fij_far_qmmm = -1 * self._qmmm_efield_far * self.qm_charge[np.newaxis, :, np.newaxis] / units.F_AU

        return self.fij_far_qmmm

    def get_fij_far_qmqm(self):
        """Get pair-wise forces between QM charges and MM charges."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()

        self.fij_far_qmqm = -0.5 * self._qmqm_efield_far * self.qm_charge[np.newaxis, :, np.newaxis] / units.F_AU

        return self.fij_far_qmqm
