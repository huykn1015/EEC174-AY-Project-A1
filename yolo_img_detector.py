import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(
                    prog='yolo_img_detector',
                    description='Detects objects in images',
                    epilog='Text at the bottom of help')
parser.add_argument('weights_file', type=str, help='specify model weights file') 
parser.add_argument('config_file', type=str, help='specify model configs file') 
parser.add_argument('labels_file', type=str, help='specify file for label names')
parser.add_argument('--images', type=str, help='specify an image or a list of images to run classification model on')
parser.add_argument('--dir', type=str, help='specify directory to run classification model on')
parser.add_argument('-inf', help='display inference time', action='store_true')
parser.add_argument('-pimg', help='display per image breakdown', action='store_true')
parser.add_argument('-classes_all', help='show number of classes detected', action='store_true')
args = parser.parse_args()
print(args)

def scan_dir(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)
    return files


def get_class_names(labels_files):
    my_file = open(labels_file, "r") 
    class_names = my_file.read() 
    class_names = class_names.split("\n") 
    my_file.close() 
    return class_names


def identify(image_path, class_names, class_count, pimg):
    image = cv2.imread(image_path)
    image_name = image_path.split('.')[0]
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
    class_count_img = [0] * len(class_names)
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box = [int(j) for j in colours[classes[i]]]
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box, 5)
            text_box = 'conf: {:.4f}, Class: {}'.format(confidences[i], class_names[classes[i]])
            cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, .8, colour_box, 2)
            class_count[classes[i]] += 1
            class_count_img[classes[i]] += 1
        if pimg:
            f = True
            res = '%s => '%(image_path)
            for i in range(len(class_names)):
                if class_count_img[i] > 0: 
                    if not f:
                        res += ' | '
                    else:
                        f = False
                    res += '%s: %d'%(class_names[i], class_count_img[i])
            print(res)
            



    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(image_name + '_out.png')
    



yolo_weights_file = args.weights_file
yolo_cfg_file = args.config_file
labels_file = args.labels_file
Inference =  args.inf
classes_all =  args.classes_all
class_names = get_class_names(labels_file)
class_count = [0] * len(class_names)

network = cv2.dnn.readNetFromDarknet(yolo_cfg_file, yolo_weights_file)
layers = network.getLayerNames()
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
inf_times = []
images = []
if args.images != None:
    images.extend(args.images)
if args.dir != None:
    images.extend(scan_dir(args.dir))

if len(images) < 1:
    print('No images specified')
    exit(0)

if args.pimg:
    print('Per Image Breakdown')
for image_path in images:
    start_time = time.time()
    identify(image_path, class_names, class_count, args.pimg)
    if args.inf:
        inf_times.append(time.time() - start_time)

if len(inf_times) > 0:
    print('Average Inference Time: %.4f seconds'%(sum(inf_times) / len(inf_times)))

if args.classes_all:
    print('Total Number of Objects/Classes Detected: %d'%(sum(class_count)))
    print('\nTotal Detection Breakdown')
    for i in range(len(class_names)):
        if class_count[i] > 0: print('%s: %d'%(class_names[i], class_count[i]))

