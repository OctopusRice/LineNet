visualize_jh = False
visualize_sampleFile = False

is_ifpGT = True
is_lineGT = False

# cfg_file = "CornerNet_Saccade"
cfg_file = "CornerNet_ifp_Saccade"
# cfg_file = "LineNet"

# cfg_file = "LineNet_tlbr"

workers = 1

if visualize_sampleFile:
    workers = 1

if cfg_file == "LineNet":
    is_ifpGT = False
    is_lineGT = False

if cfg_file == "LineNet_tlbr":
    tlbr = True