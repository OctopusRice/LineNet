#!/usr/bin/env python
import config_debug
import cv2
from core.detectors import CornerNet_Saccade
from core.detectors import CornerNet
from core.detectors import LineNet
from core.detectors import LineNet_tlbr
from core.vis_utils import draw_bboxes

if config_debug.cfg_file == "CornerNet":
    detector = CornerNet()
elif config_debug.cfg_file == "LineNet_tlbr":
    detector = LineNet_tlbr()
elif config_debug.cfg_file == "LineNet":
    detector = LineNet()
else:
    detector = CornerNet_Saccade()

image    = cv2.imread("demo.jpg")

bboxes = detector(image)
image  = draw_bboxes(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
