import importlib

MMCLASS = {'namd': "NAMD", 'sander': "Sander", 'openmm': "OpenMM"}

MMMODULE = {'NAMD': ".namd", 'Sander': ".sander", 'OpenMM': ".openmm"}


def choose_mmtool(mmSoftware):
    try:
        mm_class = MMCLASS[mmSoftware.lower()]
        mm_module = MMMODULE[mm_class]
    except:
        raise ValueError("Please choose 'namd', 'sander', or 'openmm' for mmSoftware.")

    mmtool = importlib.import_module(mm_module, package='qmhub.mmtools').__getattribute__(mm_class)

    return mmtool
