from itertools import count
from xml.etree.ElementTree import tostring
import numpy as np
import cv2
import os
import imutils
from pedsfunc import *


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
cap = cv2.VideoCapture("human.avi")
writer = None
roi0l = (10, 8)
roi0r = (757, 566)
roi1l = (287, 156)
roi1r = (711, 431)
roi2l = (27, 129)
roi2r = (230, 289)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)
fps = 5
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps,frame_size,True)
while True:
    (grabbed, image) = cap.read()

    if not grabbed:
        break


    r0 = cv2.rectangle(image, roi0l, roi0r, (100, 50, 200), 5)
    r1 = cv2.rectangle(image, roi1l, roi1r, (255, 0, 0), 5)
    r2 = cv2.rectangle(image, roi2l, roi2r, (0, 0, 0), 5)
    frame_ROI0 = image[roi0l[1]:roi0r[1], roi0l[0]:roi0r[0]]
    frame_ROI1 = image[roi1l[1]:roi1r[1], roi1l[0]:roi1r[0]]
    frame_ROI2= image[roi2l[1]:roi2r[1], roi2l[0]:roi2r[0]]
    results0, peds_counts0 = pedestrian_detection(frame_ROI0, model, layer_name, personidz=LABELS.index("person"))
    results1, peds_counts1 = pedestrian_detection(frame_ROI1, model, layer_name, personidz=LABELS.index("person"))
    results2, peds_counts2 = pedestrian_detection(frame_ROI2, model, layer_name, personidz=LABELS.index("person"))

    for res in results0:
        cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
        # print(len(results))
    cv2.putText(image, str(peds_counts0), (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for res in results1:
        cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
        # print(len(results))
    cv2.putText(image, str(peds_counts1), (290, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for res in results2:
        cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
        # print(len(results))
    cv2.putText(image, str(peds_counts2), (30, 132),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Text", peds_counts0)
    cv2.imshow("Text", peds_counts1)
    cv2.imshow("Text", peds_counts2)
    cv2.imshow("Detection", image)
    out.write(image)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

    if key == 27:
    #else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()








