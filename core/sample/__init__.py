from .cornernet import cornernet
from .cornernet_saccade import cornernet_saccade
from .cornernet_ifp_saccade import cornernet_ifp_saccade
from .cornernet_ifp_squeeze3 import cornernet_ifp_squeeze3
from .cornernet_ifp_squeeze2 import cornernet_ifp_squeeze2
from .linenet import linenet

def data_sampling_func(sys_configs, db, k_ind, data_aug=True, debug=False):
    return globals()[sys_configs.sampling_function](sys_configs, db, k_ind, data_aug, debug)
