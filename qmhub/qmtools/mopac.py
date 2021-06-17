from __future__ import division

import os
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class MOPAC(QMBase):

    QMTOOL = 'MOPAC'

    def get_qm_system(self, embed):
        """Load QM information."""

        super(MOPAC, self).get_qm_system(embed)

        self._n_qm_atoms = self.qm_atoms.n_atoms
        self._n_real_qm_atoms = self.qm_atoms.n_real_atoms
        self._n_virt_qm_atoms = self.qm_atoms.n_virt_atoms

    def get_mm_system(self, embed):
        """Load MM information."""

        super(MOPAC, self).get_mm_system(embed)

        self._n_mm_atoms = self.mm_atoms_near.n_atoms
        self._mm_position = self.mm_atoms_near.position

        self._qmmm_esp_near = embed.qmmm_esp_near
        self._qmmm_efield_near = embed.qmmm_efield_near

        self._qmmm_esp_far = embed.qmmm_esp_far
        self._qmmm_efield_far = embed.qmmm_efield_far

        self._qmqm_esp_far = embed.qmqm_esp_far
        self._qmqm_efield_far = embed.qmqm_efield_far

        self._qm_esp = np.zeros(self._n_qm_atoms, dtype=float)

        if self._qmmm_esp_near is not None:
            self._qm_esp += self._qmmm_esp_near.sum(axis=0)

        if self._qmmm_esp_far is not None:
            self._qm_esp += self._qmmm_esp_far.sum(axis=0)

        if self._qmqm_esp_far is not None:
            self._qm_esp += self._qmqm_esp_far.sum(axis=0)

        if np.all(self._qm_esp == 0.0):
            self._qm_esp = None

    def get_qm_params(self, method=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(MOPAC, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for MOPAC.")

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if self._qm_esp is not None:
            qm_mm = 'QMMM '
        else:
            qm_mm = ''

        if self.calc_forces:
            calc_forces = 'GRAD '
        else:
            calc_forces = ''

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join([" %s" % i for i in self.addparam])
            else:
                addparam = " " + self.addparam
        else:
            addparam = ''

        nproc = self.get_nproc()

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "mopac.mop"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                method=self.method, charge=self.charge,
                qm_mm=qm_mm, calc_forces=calc_forces,
                addparam=addparam, nproc=nproc))
            f.write("NAMD QM/MM\n\n")
            for i in range(self._n_qm_atoms):
                f.write(" ".join(["%6s" % self._qm_element[i],
                                  "%22.14e 1" % self._qm_position[i, 0],
                                  "%22.14e 1" % self._qm_position[i, 1],
                                  "%22.14e 1" % self._qm_position[i, 2], "\n"]))

        if self._qm_esp is not None:
            with open(os.path.join(path, "mol.in"), 'w') as f:
                f.write("\n")
                f.write("%d %d\n" % (self._n_real_qm_atoms, self._n_virt_qm_atoms))

                for i in range(self._n_qm_atoms):
                    f.write(" ".join(["%6s" % self._qm_element[i],
                                    "%22.14e" % self._qm_position[i, 0],
                                    "%22.14e" % self._qm_position[i, 1],
                                    "%22.14e" % self._qm_position[i, 2],
                                    " %22.14e" % (self._qm_esp[i]), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.basedir + "; "
        cmdline += "mopac mopac.mop 2> /dev/null"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        pass

    def parse_output(self):
        """Parse the output of QM calculation."""

        output = self.load_output(os.path.join(self.basedir, "mopac.aux"))

        self.get_qm_energy(output)
        self.get_qm_charge(output)
        self.get_qm_force(output)

        if self._qmmm_efield_near is not None:
            self.get_mm_force_near()

        if self._qmmm_efield_far is not None:
            self.get_mm_force_far()

        self.qm_atoms.qm_energy = self.qm_energy * units.E_AU
        self.qm_atoms.qm_charge = self.qm_charge
        self.qm_atoms.force = self.qm_force * units.F_AU

        if self._qmmm_efield_near is not None:
            self.mm_atoms_near.force = self.mm_force_near * units.F_AU

        if self._qmmm_efield_far is not None:
            self.mm_atoms_far.force += self.mm_force_far * units.F_AU

    def get_qm_energy(self, output=None):
        """Get QM energy from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "mopac.aux"))

        for line in output:
            if "TOTAL_ENERGY" in line:
                self.qm_energy = float(line[17:].replace("D", "E")) / units.EH_TO_EV
                break

        if self._qmqm_esp_far is not None:
            if not hasattr(self, 'qm_charge'):
                self.get_qm_charge()

            self.qm_energy -= 0.5 * (self._qmqm_esp_far * self.qm_charge[np.newaxis, :]).sum() / units.E_AU

        return self.qm_energy

    def get_qm_force(self, output=None):
        """Get QM forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "mopac.aux"))

        n_lines = int(np.ceil(self._n_qm_atoms * 3 / 10))

        for i in range(len(output)):
            if "GRADIENTS" in output[i]:
                gradients = np.empty(self._n_qm_atoms * 3, dtype=float)

                grad_lines = []
                for line in output[(i + 1):(i + 1 + n_lines)]:
                    grad_lines.append(line.rstrip())
                grad_lines = "".join(grad_lines)

                for i in range(self._n_qm_atoms * 3):
                    gradients[i] = grad_lines[(i * 18):(i * 18 + 18)]
                break

        self.qm_force = -1 * gradients.reshape(self._n_qm_atoms, 3) / units.F_AU

        if self._qmmm_efield_near is not None:
            if not hasattr(self, 'fij_near'):
                self.get_fij_near()

            self.qm_force -= self.fij_near.sum(axis=0)

        if self._qmmm_efield_far is not None:
            if not hasattr(self, 'fij_far_qmmm'):
                self.get_fij_far_qmmm()

            self.qm_force -= self.fij_far_qmmm.sum(axis=0)

        if self._qmqm_efield_far is not None:
            if not hasattr(self, 'fij_far_qmqm'):
                self.get_fij_far_qmqm()

            self.qm_force -= self.fij_far_qmqm.sum(axis=0)
            self.qm_force += self.fij_far_qmqm.sum(axis=1)

        return self.qm_force

    def get_mm_force_near(self):
        """Get MM forces from QM charges in the near field."""

        if not hasattr(self, 'fij_near'):
            self.get_fij_near()

        self.mm_force_near = self.fij_near.sum(axis=1)

        return self.mm_force_near

    def get_mm_force_far(self):
        """Get MM forces from QM charges in the far field."""

        if not hasattr(self, 'fij_far_qmmm'):
            self.get_fij_far_qmmm()

        self.mm_force_far = self.fij_far_qmmm.sum(axis=1)

        return self.mm_force_far

    def get_qm_charge(self, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "mopac.aux"))

        n_lines = int(np.ceil(self._n_qm_atoms / 10))

        for i in range(len(output)):
            if "ATOM_CHARGES" in output[i]:
                self.qm_charge = np.array([])
                for line in output[(i + 1):(i + 1 + n_lines)]:
                    self.qm_charge = np.append(self.qm_charge, np.fromstring(line, sep=' '))
                break

        return self.qm_charge
