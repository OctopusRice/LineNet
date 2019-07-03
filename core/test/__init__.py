from .cornernet import cornernet
from .linenet import linenet
from .cornernet_saccade import cornernet_saccade
from .cornernet_ifp_saccade import cornernet_ifp_saccade

def test_func(sys_config, db, nnet, result_dir, debug=False):
    return globals()[sys_config.sampling_function](db, nnet, result_dir, debug=debug)
