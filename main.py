import os
import numpy as np
import torch
from dataset import VOCDataset
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, ColorJitter, RandomAffine
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import shutil

        
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #------------------------------------------------------------------
    path = "VOC" #Download the dataset and replace your path here
    #------------------------------------------------------------------
    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        ToTensor()
    ])
    
    val_transform = Compose([
        ToTensor()
    ])

    train_dataset = VOCDataset(root=path, year="2012", image_set="train", download=False,
                               transform=train_transform)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_dataset = VOCDataset(root=path, year="2012", image_set="val", download=False,
                             transform=val_transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False
    )

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
                                                  trainable_backbone_layers=3)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_channels, num_classes = len(train_dataset.categories))
    
    model.to(device)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.005, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    bestmap = -1
    num_iter_per_epoch = len(train_dataloader)
    tb = "tensorboard"
    if os.path.isdir(tb):
        shutil.rmtree(tb)
    writer = SummaryWriter(tb)
    
    for epoch in range(50):
        # TRAINING
        model.train()
        progress_bar = tqdm.tqdm(train_dataloader, colour = "cyan")
        losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]

            loss_dict = model(images, labels)
            loss_f = sum([l for l in loss_dict.values()])

            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()
            losses.append(loss_f.item())
            mean_loss = np.mean(losses)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, 50, mean_loss))
            writer.add_scalar(tag="Train/Loss", scalar_value=mean_loss,
                              global_step = num_iter_per_epoch * epoch + iter)
    
        scheduler.step()
        
        # VALIDATION
        model.eval()
        progress_bar = tqdm.tqdm(val_dataloader, colour = "yellow")
        metric = MeanAveragePrecision(iou_type="bbox")

        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            
            all_predictions = []
            with torch.no_grad():  
                outputs = model(images)
            for output in outputs:
                all_predictions.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu")
                })
            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"]
                })
            metric.update(all_predictions, targets)
        result = metric.compute()
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        
        if result["map"] > bestmap:
            bestmap = result["map"]
            torch.save(checkpoint, "best_model.pth")
            print(f"Saved best model with mAP, mAP_50, mAP_75: {bestmap:.4f}, {result["map_50"]:.4f}, {result["map_75"]:.4f}")
            


if __name__ == '__main__':
    train()