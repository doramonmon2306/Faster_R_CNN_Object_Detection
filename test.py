import os
import numpy as np
import torch
from dataset import VOCDataset
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, ColorJitter
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import matplotlib.pyplot as plt

path = "best_model.pth"
test_image = "test.png" #Put your image path here
categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
def test(conf_threshold):
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.float()
    ori_image = cv2.imread(test_image)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))/255.
    image = [torch.from_numpy(image).float()]
    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        bboxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > conf_threshold:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(ori_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                category = categories[label]
                cv2.putText(ori_image, category, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX ,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite("prediction.jpg", ori_image)
        cv2.imshow("Prediction", ori_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    test(conf_threshold = 0.5)

