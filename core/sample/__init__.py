from .cornernet import cornernet
from .cornernet_saccade import cornernet_saccade
from .cornernet_ifp_saccade import cornernet_ifp_saccade
from .cornernet_ifp_squeeze import cornernet_ifp_squeeze
from .linenet import linenet

def data_sampling_func(sys_configs, db, k_ind, data_aug=True, debug=False):
    return globals()[sys_configs.sampling_function](sys_configs, db, k_ind, data_aug, debug)
