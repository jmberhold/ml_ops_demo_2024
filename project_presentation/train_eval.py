import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Get the current datetime as a string
# this will be used for the model name if new model
# or will be used to differentiate changes in the log for this run on existing models
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

trainning = sys.argv[1]
data_location = sys.argv[2]
model_location = sys.argv[3]
output_location = sys.argv[4]
model_name = sys.argv[5] # tells the system to look for a specific model in model_location

os.makedirs(model_location, exist_ok=True)  # Ensure directory exists

# Determine the file path based on whether the file exists
if os.path.exists(os.path.join(model_location, f"model_{model_name}.txt")):
    # Append to the existing file
    with open(os.path.join(model_location, f"model_{model_name}.txt"), "a") as file:
        sys.stdout = file  # Redirect stdout to the file
        print(f"\n\nUpdating {model_name} at date-time: {current_time}\n\nLog:")
        
else:
    # Create a new file
    with open(os.path.join(model_location, f"model_{current_time}.txt"), "w") as file:
        sys.stdout = file  # Redirect stdout to the file
        print(f"Model information for model {current_time}\n\nLog:")

# RCNNDataset class standardizes the images and resizes to match the neural network input requirements.  
# this is done prior to the data loader
class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data_folder, section_dim=(224, 224)):
        self.section_dim = section_dim
        self.data_files = os.listdir(processed_data_folder)
        self.data_files = list(map(lambda x: os.path.join(processed_data_folder, x), self.data_files))
        self.preprocess = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], "rb") as pkl:
            data = pickle.load(pkl)
        image, label = data["image"], data["label"]
        image = cv2.resize(image, self.section_dim)
        image = np.asarray(image, dtype=np.float32)
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        image = self.preprocess(image)
        label = torch.tensor(label)
        return image, label

# adds finishing layers for the rcnn to match our data

def build_model(backbone, num_classes):
    num_ftrs = backbone.fc.in_features
    # num_classes = number of class categories and +1 for background class
    backbone.fc = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(num_ftrs, 512),
                                nn.Dropout(0.2),
                                nn.Linear(512, num_classes+1))
    return backbone

# eventually this would be loaded from a json file to allow additional flexibility
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

unique_class_labels = {'bird', 'sheep', 'cow', 'chair', 'aeroplane', 'train', 'car', 'person', 'boat', 'bottle', 'motorbike', 'bus', 'cat', 'dog', 'diningtable', 'horse', 'bicycle', 'pottedplant', 'tvmonitor', 'sofa'}

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device: ", device)
print("Model is resnet50 with ImageNET weights trained on VOC dataset")
# Set up the model
resnet_backbone = torchvision.models.resnet50(weights='IMAGENET1K_V2')
# freeze pretrained backbone
for param in resnet_backbone.parameters():
    param.requires_grad = False
model = build_model(backbone=resnet_backbone, num_classes=len(unique_class_labels))
model.to(device)


if trainning == "true":
    
    
    # prepare the processed data into the training and validation dataset
    train_dataset = RCNNDataset(processed_data_folder=data_location + "/rcnn-processed-pickle/rcnn_train", section_dim=(224, 224))
    val_dataset = RCNNDataset(processed_data_folder=data_location + "/rcnn-processed-pickle/rcnn_val", section_dim=(224, 224))

    # Building the dataloader for training and training validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)    

    # if we are using a saved model and modifying that we load that
    if os.path.exists(os.path.join(model_location, f"model_{model_name}.pt")):
        # Load the saved state dictionary
        model.load_state_dict(torch.load(os.path.join(model_location, f"model_{model_name}.pt")))
              
    else:
        model_name = "ResNet_" + current_time
        

    class_weights = [1.0]+[2.0]*len(unique_class_labels) # 1 for bg and 2 for other classes
    class_weights = torch.tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    torch.cuda.empty_cache()
    num_epochs = 5
    best_val_loss = 1000
    epoch_train_losses = []
    epoch_val_losses = []
    train_accuracy = []
    val_accuracy = []
    count = 0
    for idx in range(num_epochs):
        train_losses = []
        total_train = 0
        correct_train = 0
        model.train()
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            predicted = torch.argmax(pred, 1)
            total_train += labels.shape[0]
            correct_train += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        accuracy_train = (100 * correct_train) / total_train
        train_accuracy.append(accuracy_train)
        epoch_train_loss = np.mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        val_losses = []
        total_val = 0
        correct_val = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                loss = criterion(pred, labels)
                val_losses.append(loss.item())
                predicted = torch.argmax(pred, 1)
                total_val += labels.shape[0]
                correct_val += (predicted == labels).sum().item()

        accuracy_val = (100 * correct_val) / total_val
        val_accuracy.append(accuracy_val)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_losses.append(epoch_val_loss)

        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Train Accuracy: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(idx + 1, num_epochs, epoch_train_loss, accuracy_train, epoch_val_loss, accuracy_val))


        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(idx + 1, epoch_val_loss))
            torch.save(model.state_dict(), os.path.join(model_location, f"model_{model_name}.pt"))
            count = 0
        else:
            count += 1

        if count == 5:
            break
    
    torch.save(model.state_dict(), model_location + model_name + "model.pt")

print(f"\n \n Evaluating {model_name} \n \n :")
#Load the model and set to eval
model.load_state_dict(torch.load(os.path.join(model_location, f"model_{model_name}.pt")))
model.eval()  # Set to evaluation mode

# for evaluation we will add helper functions to return information that is visually appealing

#loading the testing dataset
voc_dataset_test = torchvision.datasets.VOCDetection(root= data_location + "/voc",
                                                    image_set="test",
                                                    download=True,
                                                    year="2007")

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

#this creates a single bounding box instead of overlapping ones.
normalized_transform = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
def process_inputs(image, max_selections=300, section_size=(224, 224)):
    images = []
    boxes = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()[:max_selections]
    rects = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])
    for rect in rects:
        x1, y1, x2, y2 = rect
        img_section = image[y1: y2, x1: x2]
        img_section = cv2.resize(img_section, section_size)
        images.append(img_section)
        boxes.append(rect)
    images = np.array(images, dtype=np.float32)
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = normalized_transform(images)
    return images, np.array(boxes)

def non_max_supression(boxes, scores, labels, threshold=0.5, iou_threshold=0.5):
    idxs = np.where(scores>threshold)
    boxes = boxes[idxs]
    scores = scores[idxs]
    labels = labels[idxs]
    idxs = np.argsort(scores)
    chossen_boxes = []
    choosen_boxes_scores = []
    choosen_boxes_labels = []
    while len(idxs):
        last = len(idxs) - 1
        choosen_idx = idxs[last]
        choosen_box = boxes[choosen_idx]
        choosen_box_score = scores[choosen_idx]
        choosen_box_label = labels[choosen_idx]
        chossen_boxes.append(choosen_box)
        choosen_boxes_scores.append(choosen_box_score)
        choosen_boxes_labels.append(choosen_box_label)
        idxs = np.delete(idxs, last)
        i = len(idxs)-1
        while i >= 0:
            idx = idxs[i]
            curr_box = boxes[idx]
            curr_box_score = scores[idx]
            curr_box_label = labels[idx]
            if (curr_box_label == choosen_box_label and
                calculate_iou_score(curr_box, choosen_box) > iou_threshold):
                idxs = np.delete(idxs, i)
            i -= 1
    return chossen_boxes, choosen_boxes_scores, choosen_boxes_labels

def process_outputs(scores, boxes, threshold=0.5, iou_threshold=0.5):
    labels = np.argmax(scores, axis=1)
    probas = np.max(scores, axis=1)
    idxs = labels != 0
    boxes = boxes[idxs]
    probas = probas[idxs]
    labels = labels[idxs]
    assert len(probas) == len(boxes) == len(labels)
    final_boxes, final_boxes_scores, final_boxes_labels = non_max_supression(boxes, probas, labels, threshold, iou_threshold)
    return final_boxes, final_boxes_scores, final_boxes_labels

# defines the bounding box for image gen

# img: image as np array
# boxes: [[xmin, y_min, x_max, y_max]]
# labels: labels present in bounding boxes
# scores: array of probabilities that given object is present in bounding boxes.
# class_map: dictionary that maps index to class names
def draw_boxes(img, boxes, scores, labels, class_map=None):
    nums = len(boxes)
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2])).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4])).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        label = int(labels[i])
        if class_map is not None:
            label_txt = class_map[label]
        else:
            label_txt = str(label)
        img = cv2.putText(
            img,
            "{} {:.4f}".format(label_txt, scores[i]),
            x1y1,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 255),
            2,
        )
    return img

def predict(image, only_boxed_image=False, label_map=None, max_boxes=100, threshold=0.5, iou_threshold=0.5):
    # preprocess input image
    prep_test_images, prep_test_boxes = process_inputs(image, max_selections=max_boxes)
    model.eval()
    with torch.no_grad():
        output = model(prep_test_images.to(device))
    # postprocess output from model
    scores = torch.softmax(output, dim=1).cpu().numpy()
    boxes, boxes_scores, boxes_labels = process_outputs(scores,
                                                        prep_test_boxes,
                                                        threshold=threshold,
                                                        iou_threshold=iou_threshold)
    if only_boxed_image:
        box_image = draw_boxes(image, boxes, boxes_scores, boxes_labels, label_map)
        return box_image
    return boxes, boxes_scores, boxes_labels

# check if the output location exists, and make if not.
os.makedirs(output_location + f"/{model_name}", exist_ok=True)  # Create the directory if it doesn't exist

for i in range(10):
    #Process the image and predict
    image = np.array(voc_dataset_test[i][0])
    final_image = predict(image, only_boxed_image=True,
                          label_map=idx_2_label,
                          threshold=0.5, iou_threshold=0.5)
    # Save the result to a file
    plt.axis("off")
    plt.imshow(final_image)
    output_path = os.path.join(output_location + f"/{model_name}", f"{model_name}_predict_{i}.png")  # Define the file name
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)  # Save the image
    plt.close()  # Close the figure to free memory

print(f"Results saved in '{output_location} / {model_name}'")

# adding the metadata to the models.json file

metadata = {
        "timestamp": current_time,
        "model_name": model_name,
        "model_path": model_location,
        }

# Update the JSON file
log_file = "models.json"
if os.path.exists(os.path.join(model_location, log_file)):
    # Load existing records
    with open(os.path.join(model_location, log_file), "r") as file:
        logs = json.load(file)
else:
    logs = []  # Start fresh if file doesn't exist

# Append the new metadata
logs.append(metadata)

# Save the updated log
with open(os.path.join(model_location, log_file), "w") as file:
    json.dump(logs, file, indent=4)

print(f"Model record added to {log_file}")

# Close the log
sys.stdout.close()
sys.stdout = sys.__stdout__  # Reset stdout to console