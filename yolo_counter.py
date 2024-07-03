
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

import time
from sort import *

persons = 0

parser = argparse.ArgumentParser(
                    prog='yolo_img_detector',
                    description='Detects objects in images',
                    epilog='Text at the bottom of help')
parser.add_argument('weights_file', type=str, help='specify model weights file') 
parser.add_argument('config_file', type=str, help='specify model configs file') 
parser.add_argument('labels_file', type=str, help='specify file for label names')
parser.add_argument('video_path', type=str, help='path to video file')
args = parser.parse_args()



def get_class_names(labels_files):
    my_file = open(labels_file, "r") 
    class_names = my_file.read() 
    class_names = class_names.split("\n") 
    my_file.close() 
    return class_names


def count_persons(video_path, class_names):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter('mot_vid/MOTS20-09-result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (960, 540))
    tracker = Sort()
    i = 1
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                cv2.waitKey(1000)
                continue
            start_time = time.time()
            identify(frame, out, tracker)
            print('Frame: %d, Processing time: %.4f'%(i, time.time() - start_time))
            i += 1
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
    cap.release()
    out.release()

def identify(image, out, tracker):
    global persons
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(input_blob)
    output = network.forward(yolo_layers)
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]
    
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    results = list(filter(lambda x:classes[x] == 0, results.flatten()))
    person_boxes = [bounding_boxes[i] for i in results]
    person_boxes_ = []
    for box in person_boxes:
        tmp_box = [0] * 4
        tmp_box[0:2] = box[0:2]
        tmp_box[2] = box[0] + box[2]
        tmp_box[3] = box[1] + box[3]
        person_boxes_.append(tmp_box)
    tracker_ids = tracker.update(np.asarray(person_boxes_))
    
    if len(results) > 0 and len(tracker_ids > 0):
        z = 0
        for i in range(len(tracker_ids)):
            bounding_box = [int(x) for x in tracker_ids[i][0:4]]
            x_min, y_min = bounding_box[0], bounding_box[1]
            box_width, box_height = bounding_box[2] - x_min, bounding_box[3] - y_min
            colour_box = [int(j) for j in colours[0]]
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box, 5)
            text_box = '%d'%(tracker_ids[i][4])
            persons = max(persons, tracker_ids[i][4])
            cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, .8, colour_box, 2)
            cv2.putText(image, 'People: ' + str(int(persons)), (400,40), cv2.FONT_HERSHEY_SIMPLEX, .8, colour_box, 2)
    #print(image.shape)
    out.write(image)
    #plt.rcParams['figure.figsize'] = (12.0, 12.0)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.savefig('frame_'+ str(frames) + '.png')



yolo_weights_file = args.weights_file
yolo_cfg_file = args.config_file
labels_file = args.labels_file
video_path = args.video_path
class_names = get_class_names(labels_file)

network = cv2.dnn.readNetFromDarknet(yolo_cfg_file, yolo_weights_file)
layers = network.getLayerNames()
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

count_persons(video_path, class_names)
