import subprocess
import sys


def get_charges(fn_input, method):
    ''' Calculates HI or MBIS atomic charges
    **Arguments:**
           
           fn_input
                The filename of the file with molden or fhck format

            method
                Atom partitioning to be used: Current either hi or mbis
    
    '''
    
    command = '/opt/anaconda/anaconda3/envs/horton2/bin/python calculate_charges.py {0} {1}'.format(fn_input, method)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # Read in the charges in file charges.dat
    charges = [] 
    f = open('charges.dat','r')
    line = f.readline()
    data = line.split()
    for i in range(len(data)):
        charges.append(float(data[i]))
    return charges
    
