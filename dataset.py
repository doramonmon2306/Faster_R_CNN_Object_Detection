import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from pprint import pprint

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        boxes = []
        labels = []
        for object in data["annotation"]["object"]:
            xmin = int(object["bndbox"]["xmin"])
            ymin = int(object["bndbox"]["ymin"])
            xmax = int(object["bndbox"]["xmax"])
            ymax = int(object["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.categories.index(object['name']))
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return image, target

if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="VOC", year="2012", image_set="train", download=False, transform=transform)
    image, target = dataset[2000]
    print(target)

