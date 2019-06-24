visualize_jh = False
visualize_sampleFile = False
is_ifpGT = False
is_lineGT = False
# cfg_file = "CornerNet"
cfg_file = "LineNet"
workers = 3
legacy = False

if visualize_sampleFile:
    workers = 1

if cfg_file == "LineNet":
    is_ifpGT = False
    is_lineGT = False