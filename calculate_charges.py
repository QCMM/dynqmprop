#!/usr/bin/env python2

from horton import part, BeckeMolGrid, grid
import numpy as np
from horton import *
import sys

# We want no output on the screen
log.set_level(log.warning)


if len(sys.argv) <= 1:
   print "You have to provide an input file"
   exit()
else:
   input_file = sys.argv[1]
   if sys.argv[2] == "hi" or sys.argv[2] == "mbis":
       method = sys.argv[2]
   else:
       print "Please provide a method: hi or mbis"
       exit()

# Load the data from the orca mkl file
mol = IOData.from_file(input_file)

# Create the molecular grid
agspec = AtomicGridSpec('insane')
grid = BeckeMolGrid(mol.coordinates, mol.numbers,mol.pseudo_numbers, agspec, mode='only')
dm_full = mol.get_dm_full()
moldens = mol.obasis.compute_grid_density_dm(dm_full, grid.points,epsilon=1e-8)

# do partitioning depending on method
if method == "hi":
    try:
        with open('atoms.h5') as f: pass
    except IOError as e:
        print 'You have to create a pro atomic density database '
        exit()
    atom_dens = ProAtomDB.from_file('atoms.h5')
    atom_dens.normalize()
    # Define the scheme to be used in the charge derivation:
    WPartClass = HirshfeldIWPart
    # Do HirshfeldI partioning
    wpart = WPartClass(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid,
                       moldens, atom_dens, None, local=True, lmax=3)
    wpart.do_charges()

if method == "mbis":
    # Do Mbis partioning
    WPartClass = MBISWPart
    wpart = WPartClass(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens)
    wpart.do_charges()

charges = wpart['charges']

out = open('charges.dat','w+')
for i in range(0,np.size(charges)):
  out.write("%1.6f " % (charges[i]))
