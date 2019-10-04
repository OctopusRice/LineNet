from .cornernet import cornernet
from .linenet import linenet
from .cornernet_saccade import cornernet_saccade
from .cornernet_ifp_saccade import cornernet_ifp_saccade
from .cornernet_ifp_squeeze3 import cornernet_ifp_squeeze3
from .cornernet_ifp_squeeze2 import cornernet_ifp_squeeze2

def test_func(sys_config, db, nnet, result_dir, debug=False):
    return globals()[sys_config.sampling_function](db, nnet, result_dir, debug=debug)
