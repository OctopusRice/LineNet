#!/usr/bin/env python

import cv2
# from core.detectors import CornerNet_Saccade
# from core.detectors import CornerNet
from core.detectors import LineNet
from core.vis_utils import draw_bboxes

# detector = CornerNet_Saccade()
# detector = CornerNet()
detector = LineNet()
image    = cv2.imread("demo.jpg")

bboxes = detector(image)
image  = draw_bboxes(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
