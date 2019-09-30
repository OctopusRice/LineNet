from .base import Base, load_cfg, load_nnet
from .paths import get_file_path
from .config import SystemConfig
from .dbs.coco import COCO

class CornerNet(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet import model

        cfg_path   = get_file_path("..", "configs", "CornerNet.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet", "CornerNet_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class LineNet(Base):
    def __init__(self):
        from .test.linenet import linenet_inference
        from .models.LineNet import model

        cfg_path   = get_file_path("..", "configs", "LineNet.json")
        model_path = get_file_path("..", "cache", "nnet", "LineNet", "LineNet_10.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(LineNet, self).__init__(coco, cornernet, linenet_inference, model=model_path)

class LineNet_tlbr(Base):
    def __init__(self):
        from .test.linenet import linenet_inference
        from .models.LineNet_tlbr import model

        cfg_path   = get_file_path("..", "configs", "LineNet_tlbr.json")
        model_path = get_file_path("..", "cache", "nnet", "LineNet_tlbr", "LineNet_tlbr_220000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(LineNet_tlbr, self).__init__(coco, cornernet, linenet_inference, model=model_path)

class CornerNet_Squeeze(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet_Squeeze import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Squeeze.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Squeeze, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_Saccade(Base):
    def __init__(self):
        from .test.cornernet_saccade import cornernet_saccade_inference
        from .models.CornerNet_Saccade import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Saccade.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_Saccade", "CornerNet_Saccade_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Saccade, self).__init__(coco, cornernet, cornernet_saccade_inference, model=model_path)

class CornerNet_ifp_Saccade(Base):
    def __init__(self):
        from .test.cornernet_ifp_saccade import cornernet_ifp_saccade_inference
        from .models.CornerNet_ifp_Saccade import model

        cfg_path = get_file_path("..", "configs", "CornerNet_ifp_Saccade.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_ifp_Saccade", "CornerNet_ifp_Saccade_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_ifp_Saccade, self).__init__(coco, cornernet, cornernet_ifp_saccade_inference, model=model_path)

class CornerNet_ifp_Squeeze(Base):
    def __init__(self):
        from .test.cornernet_ifp_squeeze import cornernet_inference
        from .models.CornerNet_ifp_Squeeze import model

        cfg_path = get_file_path("..", "configs", "CornerNet_ifp_Squeeze.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_ifp_Squeeze", "CornerNet_ifp_Squeeze_5000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_ifp_Squeeze, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_ifp_Squeeze2(Base):
    def __init__(self):
        from .test.cornernet_ifp_squeeze2 import cornernet_inference
        from .models.CornerNet_ifp_Squeeze2 import model

        cfg_path = get_file_path("..", "configs", "CornerNet_ifp_Squeeze2.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_ifp_Squeeze2", "CornerNet_ifp_Squeeze2_910000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_ifp_Squeeze2, self).__init__(coco, cornernet, cornernet_inference, model=model_path)