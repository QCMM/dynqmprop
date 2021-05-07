from __future__ import absolute_import

import os
import numpy as np
import psi4

from .. import units

from .qmbase import QMBase


class PSI4(QMBase):

    QMTOOL = 'PSI4'

    def get_mm_system(self, embed):
        """Load MM information."""

        super(PSI4, self).get_mm_system(embed)

        self._n_mm_atoms = self.mm_atoms_near.n_atoms
        self._mm_position = self.mm_atoms_near.position
        self._mm_charge = self.mm_atoms_near.charge_eed

        if self.mm_atoms_far.charge_eeq is not None:
            self._qm_esp_near = embed.qm_esp_near
            self._qm_efield_near = embed.qm_efield_near
            raise NotImplementedError()

    def parse_output(self):
        """Parse the output of QM calculation."""

        self.get_qm_energy()
        self.get_qm_charge()

        self.get_mm_esp_eed()

        self.qm_atoms.qm_energy = self.qm_energy * units.E_AU
        self.qm_atoms.qm_charge = self.qm_charge
        if self.calc_forces is True:
            self.get_qm_force()
            self.get_mm_force()
            self.qm_atoms.force = self.qm_force * units.F_AU
            self.mm_atoms_near.force = self.mm_force * units.F_AU


        self.mm_atoms_near.esp_eed = self.mm_esp_eed * units.E_AU

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(PSI4, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for PSI4.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for PSI4.")

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        #psi4.set_options({'dft_functional' : '{0}'.format(self.method), 'basis' : '{0}'.format(self.basis)})
        psi4.set_options({'basis' : '{0}'.format(self.basis)})
        psi4.set_options({'e_convergence': 1e-9})
        if self.calc_forces:
            psi4.set_options({'e_convergence': 1e-8,
                              'd_convergence': 1e-8})

        # if self.read_guess:
        #     psi4.set_options({'guess': 'read'})

        if self.addparam is not None:
            addparam = dict(self.addparam)
            psi4.set_options(addparam)

        geom = []
        for i in range(self._n_qm_atoms):
            geom.append("".join(["%3s" % self._qm_element[i],
                                 "%22.14e" % self._qm_position[i, 0],
                                 "%22.14e" % self._qm_position[i, 1],
                                 "%22.14e" % self._qm_position[i, 2], "\n"]))
        geom.append("symmetry c1\n")
        geom.append("no_reorient\n")
        geom.append("no_com\n")
        geom = "".join(geom)

        molecule = psi4.geometry(geom)
        molecule.set_molecular_charge(self.charge)
        molecule.set_multiplicity(self.mult)
        molecule.fix_com(True)
        molecule.fix_orientation(True)
        molecule.update_geometry()
        mm_charge = psi4.QMMM()

        for i in range(self._n_mm_atoms):
            mm_charge.addChargeAngstrom(self._mm_charge[i],
                                        self._mm_position[i, 0],
                                        self._mm_position[i, 1],
                                        self._mm_position[i, 2])
        mm_charge.populateExtern()
        psi4.core.set_global_option_python('EXTERN', mm_charge.extern)

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "grid.dat"), 'w') as f:
            for i in range(self._n_mm_atoms):
                f.write("".join(["%22.14e" % self._mm_position[i, 0],
                                 "%22.14e" % self._mm_position[i, 1],
                                 "%22.14e" % self._mm_position[i, 2], "\n"]))

    def run(self):
        """Run QM calculation."""

        psi4.core.set_output_file(os.path.join(self.basedir, "psi4.out"), False)

        psi4_io = psi4.core.IOManager.shared_object()
        psi4_io.set_default_path(self.basedir)
        psi4_io.set_specific_retention(32, True)
        psi4_io.set_specific_path(32, self.basedir)

        nproc = self.get_nproc()
        psi4.core.set_num_threads(nproc, True)

        oldpwd = os.getcwd()
        os.chdir(self.basedir)
        method = self.method + "/" + self.basis
        print(method)
        scf_e, self.scf_wfn = psi4.energy(method, return_wfn=True)

        #write molden file for analysis, NOT FUNTIONING because waiting for iodata to handle properly
        #psi4.molden(self.scf_wfn, 'orca.molden.input', density_a=self.scf_wfn.Da())
        if self.calc_forces:
            psi4.gradient('scf', ref_wfn=self.scf_wfn)

        self.oeprop = psi4.core.OEProp(self.scf_wfn)
        self.oeprop.add("MULLIKEN_CHARGES")
        self.oeprop.add("GRID_ESP")
        self.oeprop.add("GRID_FIELD")
        self.oeprop.compute()

        os.chdir(oldpwd)

        self.exitcode = 0
        return self.exitcode

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        raise NotImplementedError()

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        self.qm_energy = self.scf_wfn.energy()

        return self.qm_energy

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force = -1 * np.array(self.scf_wfn.gradient())

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        self.mm_force = (np.column_stack([self.oeprop.Exvals(),
                                          self.oeprop.Eyvals(),
                                          self.oeprop.Ezvals()])
                         * self._mm_charge[:, np.newaxis])

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""
        self.qm_charge = np.array(self.scf_wfn.atomic_point_charges())

        return self.qm_charge

    def get_mm_esp_eed(self):
        """Get ESP at external point charges in the near field from QM density."""

        self.mm_esp_eed = np.array(self.oeprop.Vvals())

        return self.mm_esp_eed
