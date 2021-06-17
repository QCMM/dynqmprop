try:
    import numba
    from . import elec_core_qmqm_numba as elec_core_qmqm
    from . import elec_core_qmmm_numba as elec_core_qmmm
except ImportError:
    from . import elec_core_qmqm_numpy as elec_core_qmqm
    from . import elec_core_qmmm_numpy as elec_core_qmmm
