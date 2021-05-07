import os
import sys
import numpy as np

from . import embedtools
from . import mmtools
from . import qmtools


class QMMM(object):
    def __init__(self, qmatoms, mmatoms, baseDir=None, mmSoftware=None,
                 qmCharge=None, qmMult=None, qmSoftware=None,
                 qmEmbedNear=None, qmEmbedFar=None,
                 qmElement=None, qmRefCharge=True,
                 qmSwitchingType='shift',
                 qmCutoff=None, qmSwdist=None,
                 qmReadGuess=False, postProc=False):
        """
        Create a QMMM object.
        """
        self.baseDir = baseDir
        self.mmSoftware = mmSoftware
        self.qmSoftware = qmSoftware
        self.qmCharge = qmCharge
        self.qmMult = qmMult
        self.qmEmbedNear = qmEmbedNear
        self.qmEmbedFar = qmEmbedFar
        self.qmElement = qmElement
        self.qmRefCharge = qmRefCharge
        self.qmSwitchingType = qmSwitchingType
        self.qmCutoff = qmCutoff
        self.qmSwdist = qmSwdist
        self.qmReadGuess = qmReadGuess
        self.postProc = postProc

        # Initialize the system
        if self.mmSoftware is None:
            self.mmSoftware = sys.argv[2]

        #if fin is None:
        #    fin = sys.argv[1]

        self.system = mmtools.choose_mmtool(self.mmSoftware)(qmatoms, mmatoms)

        if self.qmElement is not None:
            self.system.qm_atoms.element = np.asarray(self.qmElement)

        if self.qmCharge is None:
            self.qmCharge = self.system.qm_charge

        if self.qmMult is None:
            self.qmMult = self.system.qm_mult

        # Set up embedding scheme
        #if self.system.mm_atoms is not None:
        self.embed = embedtools.choose_embedtool(self.qmEmbedNear, self.qmEmbedFar)(
                self.system, self.qmRefCharge, self.qmSwitchingType, self.qmCutoff, self.qmSwdist)

        # Initialize the QM system
        if self.baseDir is None:
            self.baseDir = os.getcwd() + "/"
        self.qm = qmtools.choose_qmtool(self.qmSoftware)(
            self.baseDir, self.embed, self.qmCharge, self.qmMult)

        if self.qmReadGuess and not self.system.step == 0:
            self.qm.read_guess = True
        else:
            self.qm.read_guess = False

        if self.postProc:
            self.qm.calc_forces = False
        else:
            self.qm.calc_forces = True

    def run_qm(self, **kwargs):
        """Run QM calculation."""
        self.qm.get_qm_params(**kwargs)
        self.qm.gen_input()
        self.qm.run()
        if self.qm.exitcode != 0:
            sys.exit(self.qm.exitcode)

    def dry_run_qm(self, **kwargs):
        """Generate input file without running QM calculation."""
        if self.postProc:
            self.qm.get_qm_params(**kwargs)
            self.qm.gen_input()
        else:
            raise ValueError("dryrun_qm() can only be used with postProc=True.")

    def parse_output(self):
        """Parse the output of QM calculation."""
        if self.postProc:
            pass
        elif hasattr(self.qm, 'exitcode'):
            if self.qm.exitcode == 0:
                self.system.parse_output(self.qm)
                self.system.apply_corrections(self.embed)
            else:
                raise ValueError("QM calculation did not finish normally.")
        else:
            raise ValueError("Need to run_qm() first.")

    def save_results(self):
        """Save the results of QM calculation to file."""
        if hasattr(self.system, 'qm_energy'):
            self.system.save_results()
        else:
            raise ValueError("Need to parse_output() first.")

    def save_charges(self):
        """Save the QM and MM charges to file (for debugging only)."""
        system_scale = np.ones(self.system.n_atoms)
        system_dij_min = np.zeros(self.system.n_atoms)
        system_charge = np.zeros(self.system.n_atoms)

        system_scale.flat[self.system.mm_atoms.real_atoms.index] = self.embed.charge_scale

        system_dij_min[self.system.mm_atoms.real_atoms.index] = self.system.mm_atoms.real_atoms.dij_min

        system_charge[self.system.mm_atoms.real_atoms.index] = self.system.mm_atoms.real_atoms.charge
        system_charge.flat[self.embed.mm_atoms_near.real_atoms.index] = self.embed.mm_atoms_near.charge_near
        system_charge[self.system.qm_atoms.real_atoms.index] = self.system.qm_atoms.real_atoms.charge

        np.save(self.baseDir + "system_scale", system_scale)
        np.save(self.baseDir + "system_dij_min", system_dij_min)
        np.save(self.baseDir + "system_charge", system_charge)
