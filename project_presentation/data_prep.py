import sys
import torchvision
import os
import numpy as np
from tqdm import tqdm
import cv2
import pickle


# need to work save location from string to boolean, or if 
trainning = sys.argv[1]
save_location = sys.argv[2]

max_iou_threshold = 0.7
max_boxes = 50
max_selections = 1000
processed_data_save_path_train = save_location + "/rcnn-processed-pickle/rcnn_train"
processed_data_save_path_val = save_location + "/rcnn-processed-pickle/rcnn_val"
processed_data_save_path_test = save_location + "/rcnn-processed-pickle/rcnn_test"
os.makedirs(processed_data_save_path_train, exist_ok=True)
os.makedirs(processed_data_save_path_val, exist_ok=True)
os.makedirs(processed_data_save_path_test, exist_ok=True)

"""
    Downloads and prepares the PASCAL VOC dataset for training and validation.

    Args:
        trainning (str): True is for training, and False is for testing.
        save_location (str): Where the data is to be stored.
"""

if trainning == "true":

    # loading detection data
    voc_dataset_train = torchvision.datasets.VOCDetection(root= save_location + "/voc",
                                                image_set="train",
                                                download=True,
                                                year="2007")
    voc_dataset_val = torchvision.datasets.VOCDetection(root= save_location + "/voc",
                                                image_set="val",
                                                download=True,
                                                year="2007")

else:
    # loading the training data  
    voc_dataset_test = torchvision.datasets.VOCDetection(root= save_location + "/voc",
                                                image_set="test",
                                                download=True,
                                                year="2007")
    
# Determine the object classes for determination

all_objs = []
for ds in voc_dataset_train:
    obj_annots = ds[1]["annotation"]["object"]
    for obj in obj_annots:
        all_objs.append(obj["name"])

unique_class_labels = set(all_objs)


# manually sets the indices and the labels
label_2_idx = {'pottedplant': 1, 'person': 2,
               'motorbike': 3, 'train': 4,
               'dog': 5, 'diningtable': 6,
               'horse': 7, 'bus': 8,
               'aeroplane': 9, 'sofa': 10,
               'sheep': 11, 'tvmonitor': 12,
               'bird': 13, 'bottle': 14,
               'chair': 15, 'cat': 16,
               'bicycle': 17, 'cow': 18,
               'boat': 19, 'car': 20, 'bg': 0}
idx_2_label = {1: 'pottedplant', 2: 'person',
               3: 'motorbike', 4: 'train',
               5: 'dog', 6: 'diningtable',
               7: 'horse', 8: 'bus',
               9: 'aeroplane', 10: 'sofa',
               11: 'sheep', 12: 'tvmonitor',
               13: 'bird', 14: 'bottle',
               15: 'chair', 16: 'cat',
               17: 'bicycle', 18: 'cow',
               19: 'boat', 20: 'car', 0: 'bg'}



# calculates the intersection over union score between boxes

def calculate_iou_score(box_1, box_2):
    '''
        box_1 = single of ground truth bounding boxes
        box_2 = single of predicted bounded boxes
    '''
    box_1_x1 = box_1[0]
    box_1_y1 = box_1[1]
    box_1_x2 = box_1[2]
    box_1_y2 = box_1[3]

    box_2_x1 = box_2[0]
    box_2_y1 = box_2[1]
    box_2_x2 = box_2[2]
    box_2_y2 = box_2[3]

    x1 = np.maximum(box_1_x1, box_2_x1)
    y1 = np.maximum(box_1_y1, box_2_y1)
    x2 = np.minimum(box_1_x2, box_2_x2)
    y2 = np.minimum(box_1_y2, box_2_y2)

    area_of_intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box_1 = (box_1_x2 - box_1_x1 + 1) * (box_1_y2 - box_1_y1 + 1)
    area_box_2 = (box_2_x2 - box_2_x1 + 1) * (box_2_y2 - box_2_y1 + 1)
    area_of_union = area_box_1 + area_box_2 - area_of_intersection

    return area_of_intersection/float(area_of_union)

# This processes the image sections and the images for the rcnn

def process_data_for_rcnn(image, rects, class_map, boxes_annots, iou_threshold, max_boxes):
    true_classes = []
    image_sections = []
    true_count = 0
    false_count = 0
    for annot in boxes_annots:
        label = annot["name"]
        box = [int(c) for _, c in annot["bndbox"].items()]
        box = np.array(box)
        for rect in rects:
            iou_score = calculate_iou_score(rect, box)
            if iou_score > iou_threshold:
                if true_count < max_boxes//2:
                    true_classes.append(class_map[label])
                    x1, y1, x2, y2 = rect
                    img_section = image[y1: y2, x1: x2]
                    image_sections.append(img_section)
                    true_count += 1
            else:
                if false_count < max_boxes//2:
                    true_classes.append(0)
                    x1, y1, x2, y2 = rect
                    img_section = image[y1: y2, x1: x2]
                    image_sections.append(img_section)
                    false_count += 1
    return image_sections, true_classes

# going through all the images in the dataset for preproceesing for network

def image_processing(process_save_path, voc_dataset):
    all_images = []
    all_labels = []
    count = 0
    if len(os.listdir(process_save_path)) < 79000:
        for image, annot in tqdm(voc_dataset):
            image = np.array(image)
            boxes_annots = annot["annotation"]["object"]
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()[:max_selections]
            rects = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])
            images, classes = process_data_for_rcnn(image,
                                                    rects,
                                                    label_2_idx,
                                                    boxes_annots,
                                                    max_iou_threshold,
                                                    max_boxes)
            count += 1
            all_images += images
            all_labels += classes

        # saving processed data to pickle file
        for idx, (image, label) in enumerate(zip(all_images, all_labels)):
            with open(os.path.join(process_save_path, f"img_{idx}.pkl"), "wb") as pkl:
                pickle.dump({"image": image, "label": label}, pkl)
    else:
        print("Data Already Prepared.")


if trainning == "true":

    image_processing(processed_data_save_path_train, voc_dataset_train)

    image_processing(processed_data_save_path_val, voc_dataset_val)

else: 

    image_processing(processed_data_save_path_test, voc_dataset_test)


