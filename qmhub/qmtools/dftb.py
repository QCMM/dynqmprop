from __future__ import division

import os
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class DFTB(QMBase):

    QMTOOL = 'DFTB+'

    def get_mm_system(self, embed):
        """Load MM information."""

        super(DFTB, self).get_mm_system(embed)

        if self.mm_atoms_far.charge_eeq is not None:
            self._pbc = True
            self._mm_atoms = embed.mm_atoms
            self._n_mm_atoms = self._mm_atoms.n_atoms
            self._mm_position = self._mm_atoms.position
            self._mm_charge = self._mm_atoms.charge
            self._cell_basis = embed.cell_basis
        else:
            self._pbc = False
            self._mm_atoms = embed.mm_atoms_near
            self._n_mm_atoms = self._mm_atoms.n_atoms
            self._mm_position = self._mm_atoms.position
            self._mm_charge = self._mm_atoms.charge_eeq

    def get_qm_params(self, skfpath=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(DFTB, self).get_qm_params(**kwargs)

        if skfpath is not None:
            self.skfpath = os.path.join(skfpath, '')
        else:
            raise ValueError("Please set skfpath for DFTB+.")

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        elements = np.unique(self._qm_element).tolist()
        MaxAngularMomentum = "\n    ".join([i+" = "+qmtmpl.MaxAngularMomentum[i] for i in elements])
        HubbardDerivs = "\n    ".join([i+" = "+qmtmpl.HubbardDerivs[i] for i in elements])

        if self._pbc:
            KPointsAndWeights = qmtmpl.KPointsAndWeights
        else:
            KPointsAndWeights = ""

        if self.calc_forces:
            calc_forces = 'Yes'
        else:
            calc_forces = 'No'

        if self.read_guess:
            read_guess = 'Yes'
        else:
            read_guess = 'No'

        if self.addparam is not None:
            addparam = self.addparam
        else:
            addparam = ''

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "dftb_in.hsd"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                charge=self.charge, n_mm_atoms=self._n_mm_atoms,
                read_guess=read_guess, calc_forces=calc_forces,
                skfpath=self.skfpath, addparam=addparam,
                MaxAngularMomentum=MaxAngularMomentum,
                HubbardDerivs=HubbardDerivs,
                KPointsAndWeights=KPointsAndWeights))

        with open(os.path.join(path, "input_geometry.gen"), 'w') as f:
            if self._pbc:
                f.write(str(self._n_qm_atoms) + " S" + "\n")
            else:
                f.write(str(self._n_qm_atoms) + " C" + "\n")
            f.write(" ".join(elements) + "\n")
            for i in range(self._n_qm_atoms):
                f.write("".join(["%6d" % (i+1),
                                 "%4d" % (elements.index(self._qm_element[i])+1),
                                 "%22.14e" % self._qm_position[i, 0],
                                 "%22.14e" % self._qm_position[i, 1],
                                 "%22.14e" % self._qm_position[i, 2], "\n"]))
            if self._pbc:
                f.write("%22.14e%22.14e%22.14e\n" % (0.0, 0.0, 0.0))
                f.write("".join(["%22.14e" % i for i in self._cell_basis[0]]) + "\n")
                f.write("".join(["%22.14e" % i for i in self._cell_basis[1]]) + "\n")
                f.write("".join(["%22.14e" % i for i in self._cell_basis[2]]) + "\n")

        with open(os.path.join(path, "charges.dat"), 'w') as f:
            for i in range(self._n_mm_atoms):
                f.write("".join(["%22.14e" % self._mm_position[i, 0],
                                 "%22.14e" % self._mm_position[i, 1],
                                 "%22.14e" % self._mm_position[i, 2],
                                 " %22.14e" % self._mm_charge[i], "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "export OMP_NUM_THREADS=%d; dftb+ > dftb.out" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        qmsave = os.path.join(self.basedir, "charges.bin")
        if os.path.isfile(qmsave):
            os.remove(qmsave)

    def parse_output(self):
        """Parse the output of QM calculation."""

        output = self.load_output(os.path.join(self.basedir, "results.tag"))

        self.get_qm_energy(output)
        self.get_qm_charge(output)
        self.get_qm_force(output)
        self.get_mm_force(output)

        self.qm_atoms.qm_energy = self.qm_energy * units.E_AU
        self.qm_atoms.qm_charge = self.qm_charge
        self.qm_atoms.force = self.qm_force * units.F_AU
        self._mm_atoms.force = self.mm_force * units.F_AU

    def get_qm_energy(self, output=None):
        """Get QM energy from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "results.tag"))

        self.qm_energy = np.loadtxt(output[1:2], dtype=float)

        return self.qm_energy

    def get_qm_charge(self, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "results.tag"))

        self.qm_charge = np.array([], dtype=float)

        n_lines = int(np.ceil(self._n_qm_atoms / 3))
        start = self._n_qm_atoms + self._n_mm_atoms + n_lines + 14
        stop = start + n_lines

        for line in output[start:stop]:
            self.qm_charge = np.append(self.qm_charge, np.fromstring(line, sep=' '))

        return self.qm_charge

    def get_qm_force(self, output=None):
        """Get QM forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "results.tag"))

        self.qm_force = np.loadtxt(output[5:(self._n_qm_atoms + 5)], dtype=float)

        return self.qm_force

    def get_mm_force(self, output=None):
        """Get external point charge forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "results.tag"))

        start = self._n_qm_atoms + 6
        stop = start + self._n_mm_atoms

        self.mm_force = np.loadtxt(output[start:stop], dtype=float)

        return self.mm_force
