import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(
                    prog='yolo_img_detector',
                    description='Detects objects in images',
                    epilog='Text at the bottom of help')


yolo_weights_file = ''
yolo_cfg_file = ''
labels_file = ''
images_path = ''
images_list = ''


Inference = False
classes_all = False


network = cv2.dnn.readNetFromDarknet('yolo_files/yolov3.cfg', 'yolo_files/yolov3.weights')
layers = network.getLayerNames()
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']


my_file = open("yolo_files/coco.names", "r") 
# reading the file 
class_names = my_file.read() 
  
# replacing end splitting the text  
# when newline ('\n') is seen. 
class_names = class_names.split("\n") 
#print(class_names) 
my_file.close() 
#%matplotlib inline
image = cv2.imread('imgs/sample.jpg')
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('s1.png')


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
            #print(classes)


results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
coco_labels = 80

np.random.seed(42)
colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

if len(results) > 0:
    for i in results.flatten():
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        colour_box = [int(j) for j in colours[classes[i]]]
        cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                      colour_box, 5)
        text_box = 'conf: {:.4f}, Class: {}'.format(confidences[i], class_names[classes[i]])
        cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_box, 5)


plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('s2.png')
