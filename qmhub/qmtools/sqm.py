import os
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class SQM(QMBase):

    QMTOOL = 'SQM'

    def get_mm_system(self, embed):
        """Load MM information."""

        super(SQM, self).get_mm_system(embed)

        self._n_mm_atoms = self.mm_atoms_near.n_atoms
        self._mm_position = self.mm_atoms_near.position
        self._mm_charge = self.mm_atoms_near.charge_eed

        self._qmmm_esp_near = embed.qmmm_esp_near
        self._qmmm_efield_near = embed.qmmm_efield_near

        self._qmmm_esp_far = embed.qmmm_esp_far
        self._qmmm_efield_far = embed.qmmm_efield_far

        self._qmqm_esp_far = embed.qmqm_esp_far
        self._qmqm_efield_far = embed.qmqm_efield_far

        self._qm_esp = np.zeros(self._n_qm_atoms, dtype=float)

        if self._qmmm_esp_near is not None:
            self._qm_esp += embed.qmmm_esp_near.sum(axis=0) / units.E_AU

        if self._qmmm_esp_far is not None:
            self._qm_esp += embed.qmmm_esp_far.sum(axis=0) / units.E_AU

        if self._qmqm_esp_far is not None:
            self._qm_esp += embed.qmqm_esp_far.sum(axis=0) / units.E_AU

        if np.all(self._qm_esp == 0.0):
            self._qm_esp = None

    def get_qm_params(self, method=None, skfpath=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(SQM, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for SQM.")

        if self.method.lower() in ['dftb2', 'dftb3']:
            if self._mm_charge is not None:
                raise ValueError("Electrostatic embedding with electron density is not supported in DFTB.")

            if skfpath is not None:
                self.skfpath = " dftb_slko_path= '" + os.path.join(skfpath, '') + "',\n"
            else:
                self.skfpath = ''
        else:
            self.skfpath = ''

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        element_num = []
        for element in self._qm_element:
            element_num.append(qmtmpl.Elements.index(element))

        if self._mm_charge is not None:
            qm_mm = 1
        else:
            qm_mm = 0

        if self.calc_forces:
            verbosity = 4
        else:
            verbosity = 1

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join(["%s,\n" % i for i in self.addparam])
            else:
                addparam = self.addparam + ',\n'
        else:
            addparam = ''

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "sqm.inp"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                method=self.method, verbosity=verbosity,
                charge=self.charge, mult=self.mult, qm_mm=qm_mm,
                skfpath=self.skfpath, addparam=addparam))

            for i in range(self._n_qm_atoms):
                f.write("".join(["%4d" % element_num[i],
                                 "%4s " % self._qm_element[i],
                                 "%22.14e" % self._qm_position[i, 0],
                                 "%22.14e" % self._qm_position[i, 1],
                                 "%22.14e" % self._qm_position[i, 2], "\n"]))

            if self._mm_charge is not None:
                f.write("#EXCHARGES\n")
                for i in range(self._n_mm_atoms):
                    f.write("".join(["   1   H ",
                                        "%22.14e" % self._mm_position[i, 0],
                                        "%22.14e" % self._mm_position[i, 1],
                                        "%22.14e" % self._mm_position[i, 2],
                                        " %22.14e" % self._mm_charge[i], "\n"]))
                f.write("#END" + "\n")

            if self._qm_esp is not None:
                f.write("\n")
                f.write("#ATOM_SITE_POTENTIAL\n")
                for i in range(self._n_qm_atoms):
                    f.write("%22.14e\n" % self._qm_esp[i])
                f.write("#END" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "sqm -O -i sqm.inp -o sqm.out"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        pass

    def parse_output(self):
        """Parse the output of QM calculation."""

        output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        self.get_qm_energy(output)
        self.get_qm_charge(output)
        self.get_qm_force(output)

        if self._mm_charge is not None:
            self.get_mm_force_eed(output)

        if self._qm_esp is not None:
            self.get_qm_force_eeq()

        if self._qmmm_efield_near is not None:
            self.get_mm_force_near()

        if self._qmmm_efield_far is not None:
            self.get_mm_force_far()

        if self._mm_charge is not None:
            self.get_mm_esp_eed()

        self.qm_atoms.qm_energy = self.qm_energy * units.E_AU
        self.qm_atoms.qm_charge = self.qm_charge
        self.qm_atoms.force = self.qm_force * units.F_AU

        if self._qm_esp is not None:
            self.qm_atoms.force += self.qm_force_eeq * units.F_AU

        if self._mm_charge is not None:
            self.mm_atoms_near.force += self.mm_force_eed * units.F_AU
            self.mm_atoms_near.esp_eed = self.mm_esp_eed * units.E_AU

        if self._qmmm_efield_near is not None:
            self.mm_atoms_near.force += self.mm_force_near * units.F_AU

        if self._qmmm_efield_far is not None:
            self.mm_atoms_far.force += self.mm_force_far * units.F_AU

    def get_qm_energy(self, output=None):
        """Get QM energy from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        for line in output:
            line = line.strip().expandtabs()

            if "Heat of formation" in line:
                scf_energy = line.split()[-5]
                break

        self.qm_energy = float(scf_energy) / units.E_AU

        if self._qmqm_esp_far is not None:
            if not hasattr(self, 'qm_charge'):
                self.get_qm_charge()

            self.qm_energy -= 0.5 * (self._qmqm_esp_far * self.qm_charge[np.newaxis, :]).sum() / units.E_AU

        return self.qm_energy

    def get_qm_charge(self, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        for i in range(len(output)):
            if "Atomic Charges" in output[i]:
                self.qm_charge = np.empty(self._n_qm_atoms, dtype=float)
                for j in range(self._n_qm_atoms):
                    line = output[i + 2 + j]
                    self.qm_charge[j] = float(line.split()[-1])
                break

        return self.qm_charge

    def get_qm_force(self, output=None):
        """Get QM forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        for i in range(len(output)):
            if "Forces on QM atoms from SCF calculation" in output[i]:
                self.qm_force = np.empty((self._n_qm_atoms, 3), dtype=float)
                for j in range(self._n_qm_atoms):
                    line = output[i + 1 + j]
                    self.qm_force[j][0] = -1 * float(line[18:38])
                    self.qm_force[j][1] = -1 * float(line[38:58])
                    self.qm_force[j][2] = -1 * float(line[58:78])
                break

        self.qm_force /= units.F_AU

        return self.qm_force

    def get_qm_force_eeq(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force_eeq = np.zeros((self._n_qm_atoms, 3), dtype=float)

        if self._qmmm_efield_near is not None:
            if not hasattr(self, 'fij_near'):
                self.get_fij_near()

            self.qm_force_eeq -= self.fij_near.sum(axis=0)

        if self._qmmm_efield_far is not None:
            if not hasattr(self, 'fij_far_qmmm'):
                self.get_fij_far_qmmm()

            self.qm_force_eeq -= self.fij_far_qmmm.sum(axis=0)

        if self._qmqm_efield_far is not None:
            if not hasattr(self, 'fij_far_qmqm'):
                self.get_fij_far_qmqm()

            self.qm_force_eeq -= self.fij_far_qmqm.sum(axis=0)
            self.qm_force_eeq += self.fij_far_qmqm.sum(axis=1)

        return self.qm_force_eeq

    def get_mm_force_eed(self, output=None):
        """Get external point charge forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        for i in range(len(output)):
            if "Forces on MM atoms from SCF calculation" in output[i]:
                self.mm_force_eed = np.empty((self._n_mm_atoms, 3), dtype=float)
                for j in range(self._n_mm_atoms):
                    line = output[i + 1 + j]
                    self.mm_force_eed[j] = [-1 * float(n) for n in line.split()[-3:]]
                break

        self.mm_force_eed /= units.F_AU

        return self.mm_force_eed

    def get_mm_force_near(self, output=None):
        """Get external point charge forces from output of QM calculation."""

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

    def get_mm_esp_eed(self, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "sqm.out"))

        for i in range(len(output)):
            if "Electrostatic Potential on MM atoms from QM Atoms" in output[i]:
                self.mm_esp_eed = np.empty(self._n_mm_atoms, dtype=float)
                for j in range(self._n_mm_atoms):
                    line = output[i + 1 + j]
                    self.mm_esp_eed[j] = float(line.split()[-1])
                break

        self.mm_esp_eed = np.nan_to_num(self.mm_esp_eed)
        self.mm_esp_eed /= units.E_AU

        return self.mm_esp_eed
